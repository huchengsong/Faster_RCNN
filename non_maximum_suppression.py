import numpy as np
import cupy as cp


def non_maximum_suppression_rpn(bbox, thresh, score=None, limit=None):
    """
    apply non maximun suppression
    :param bbox: (N, 4), ndarray
    :param thresh: threshold of IOU for suppression
    :param score: (N, ), score of each bounding box
    :param limit: number of bounding boxes to output
    :return: (K, 4), darray, bounding boxes after nms. Sorting from highest score to lowest score
    """
    bbox_cp = cp.asarray(bbox)
    if score is not None:
        score = cp.asarray(score)
    ind_bbox = _non_maximum_suppression_gpu(bbox_cp, thresh, score, limit)
    ind_bbox = cp.asnumpy(ind_bbox)
    selected_bbox = bbox[ind_bbox]
    return ind_bbox, selected_bbox


def non_maximum_suppression_roi(box_scores, bboxes, class_list, score_threshold, iou_threshold):
    """
    using non maximum suppression to reduce bbox number
    :param box_scores: (N, class_num) ndarray
    :param bboxes: (N, 4 * class_num) ndarray
    :param class_list: list of class ID that NMS apply to
    :param score_threshold: score threshold for box selection
    :param iou_threshold: iou threshold
    :return: label (K, ), score (K, ), box (K, 4)
    """
    bbox_result = []
    score_result = []
    label_result = []

    class_pred = np.argmax(box_scores, axis=1)
    for class_id in class_list:
        index = np.where((class_pred == class_id) &
                         (box_scores[:, class_id] > score_threshold))
        if np.array(index).size == 0:
            continue
        score_candidate = box_scores[index, class_id].flatten()
        box_candidate = bboxes[index, class_id * 4:(class_id + 1) * 4].reshape(-1, 4)

        ind_bbox, selected_bbox = non_maximum_suppression_rpn(box_candidate, iou_threshold, score_candidate)
        selected_score = score_candidate[ind_bbox]
        selected_label = np.full(ind_bbox.shape, class_id)

        bbox_result.append(selected_bbox)
        score_result.append(selected_score)
        label_result.append(selected_label)

    return np.concatenate(bbox_result), np.concatenate(score_result), np.concatenate(label_result)


@cp.util.memoize(for_each_device=True)
def _load_kernel(kernel_name, code, options=()):
    cp.cuda.runtime.free(0)
    assert isinstance(options, tuple)
    kernel_code = cp.cuda.compile_with_cache(code, options=options)
    return kernel_code.get_function(kernel_name)


def _non_maximum_suppression_gpu(bbox, thresh, score=None, limit=None):
    if len(bbox) == 0:
        return cp.zeros((0,), dtype=np.int32)

    n_bbox = bbox.shape[0]

    if score is not None:
        order = score.argsort()[::-1].astype(np.int32)
    else:
        order = cp.arange(n_bbox, dtype=np.int32)

    sorted_bbox = bbox[order, :]
    selec, n_selec = _call_nms_kernel(
        sorted_bbox, thresh)
    selec = selec[:n_selec]
    selec = order[selec]
    if limit is not None:
        selec = selec[:limit]
    return selec


_nms_gpu_code = '''
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long) * 8;

__device__
inline float devIoU(float const *const bbox_a, float const *const bbox_b) {
  float top = max(bbox_a[0], bbox_b[0]);
  float bottom = min(bbox_a[2], bbox_b[2]);
  float left = max(bbox_a[1], bbox_b[1]);
  float right = min(bbox_a[3], bbox_b[3]);
  float height = max(bottom - top, 0.f);
  float width = max(right - left, 0.f);
  float area_i = height * width;
  float area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1]);
  float area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1]);
  return area_i / (area_a + area_b - area_i);
}

extern "C"
__global__
void nms_kernel(const int n_bbox, const float thresh,
                const float *dev_bbox,
                unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  const int row_size =
        min(n_bbox - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_bbox - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_bbox[threadsPerBlock * 4];
  if (threadIdx.x < col_size) {
    block_bbox[threadIdx.x * 4 + 0] =
        dev_bbox[(threadsPerBlock * col_start + threadIdx.x) * 4 + 0];
    block_bbox[threadIdx.x * 4 + 1] =
        dev_bbox[(threadsPerBlock * col_start + threadIdx.x) * 4 + 1];
    block_bbox[threadIdx.x * 4 + 2] =
        dev_bbox[(threadsPerBlock * col_start + threadIdx.x) * 4 + 2];
    block_bbox[threadIdx.x * 4 + 3] =
        dev_bbox[(threadsPerBlock * col_start + threadIdx.x) * 4 + 3];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_bbox + cur_box_idx * 4;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_bbox + i * 4) >= thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_bbox, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}
'''


def _call_nms_kernel(bbox, thresh):
    n_bbox = bbox.shape[0]
    threads_per_block = 64
    col_blocks = np.ceil(n_bbox / threads_per_block).astype(np.int32)
    blocks = (col_blocks, col_blocks, 1)
    threads = (threads_per_block, 1, 1)

    mask_dev = cp.zeros((n_bbox * col_blocks,), dtype=np.uint64)
    bbox = cp.ascontiguousarray(bbox, dtype=np.float32)
    kern = _load_kernel('nms_kernel', _nms_gpu_code)
    kern(blocks, threads, args=(cp.int32(n_bbox), cp.float32(thresh),
                                bbox, mask_dev))

    mask_host = mask_dev.get()
    selection, n_selec = _nms_gpu_post(
        mask_host, n_bbox, threads_per_block, col_blocks)
    return selection, n_selec


def _nms_gpu_post(mask, n_bbox, threads_per_block, col_blocks):
    n_selection = 0
    one_ull = np.array([1],dtype=np.uint64)
    selection = np.zeros((n_bbox,), dtype=np.int32)
    remv = np.zeros((col_blocks,), dtype=np.uint64)

    for i in range(n_bbox):
        nblock = i // threads_per_block
        inblock = i % threads_per_block

        if not (remv[nblock] & one_ull << inblock):
            selection[n_selection] = i
            n_selection += 1

            index = i * col_blocks
            for j in range(nblock, col_blocks):
                remv[j] |= mask[index + j]
    return selection, n_selection


def test():
    # test non_maximum_suppression_rpn()
    bbox = np.array([[0, 0, 100, 100], [0, 0, 75, 75], [100, 100, 200, 200]])
    thresh = 0.5
    score = np.array([0.5, 0.7, 0.8])
    limit = 3
    ind, box = non_maximum_suppression_rpn(bbox, thresh, score, limit=limit)
    print(box)

    # test non_maximum_suppression_roi()
    # def softmax(x, axis=None):
    #     return np.exp(x) / np.exp(x).sum(axis=axis)[:, None]
    # box_scores = softmax(np.random.rand(10, 3), axis=1)
    # print(box_scores)
    box_scores = np.array([[0.35629722, 0.29934438, 0.34435839],
                           [0.33190525, 0.43829933, 0.22979542],
                           [0.35513699, 0.39927991, 0.24558309],
                           [0.43057433, 0.33340436, 0.23602131],
                           [0.33935831, 0.20390023, 0.45674146],
                           [0.37249841, 0.30570373, 0.32179787],
                           [0.47038923, 0.23873957, 0.2908712 ],
                           [0.28352702, 0.31179673, 0.40467625],
                           [0.22726669, 0.47013369, 0.30259961],
                           [0.39092056, 0.27074105, 0.33833839]])

    bboxes = np.array([[0, 0, 100, 100],
                       [0, 0, 110, 110],
                       [0, 0, 125, 125],
                       [0, 0, 150, 150],
                       [0, 0, 100, 100],
                       [0, 0, 110, 110],
                       [0, 0, 125, 125],
                       [0, 0, 150, 150],
                       [500, 500, 600, 600],
                       [500, 500, 650, 650]])
    class_label, box_score, box = \
        non_maximum_suppression_roi(box_scores,
                                    np.concatenate((bboxes, bboxes, bboxes), axis=1),
                                    [0, 1, 2],
                                    score_threshold=0,
                                    iou_threshold=0.5)
    print(class_label, box_score, box)


if __name__ == "__main__":
    test()
