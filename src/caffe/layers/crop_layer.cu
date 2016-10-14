#include <vector>

#include "caffe/vision_layers.hpp"

namespace caffe {

// Copy (one line per thread) from one array to another, with arbitrary
// strides in the last all dimensions.
template <typename Dtype>
__global__ void copy_kernel_forward(const int num_lines, 
    const int line_size, const int num_axes,
    const int* top_shape_data, const int* bottom_shape_data,
    const int* crop_data, const Dtype* bottom, Dtype* top) {
  CUDA_KERNEL_LOOP(line, num_lines) {
    int k = line;
    int top_index = 0;
    int bottom_index = crop_data[num_axes-1];
    int top_offset = top_shape_data[num_axes-1];
    int bottom_offset = bottom_shape_data[num_axes-1];
    // Calcuate top and bottom index
    for (int j = num_axes - 2; j >= 0; --j) {
      int topsub = k % top_shape_data[j];
      top_index += topsub*top_offset;
      top_offset *= top_shape_data[j];

      int bottomsub = topsub+crop_data[j];
      bottom_index += bottomsub*bottom_offset;
      bottom_offset *= bottom_shape_data[j];
      k /= top_shape_data[j];
    }
    // Copy line bottom -> top
    for (int i = 0; i < line_size; ++i) {
      top[top_index + i] = bottom[bottom_index + i];
    }
  }
}

template <typename Dtype>
__global__ void copy_kernel_backward(const int num_lines,
    const int line_size, const int num_axes,
    const int* top_shape_data,const int* bottom_shape_data,
    const int* crop_data, const Dtype* top, Dtype* bottom) {
  CUDA_KERNEL_LOOP(line, num_lines) {
    int k = line;
    int top_index = 0;
    int bottom_index = crop_data[num_axes-1];
    int top_offset = top_shape_data[num_axes-1];
    int bottom_offset = bottom_shape_data[num_axes-1];
    // Calcuate top and bottom index
    for (int j = num_axes - 2; j >= 0; --j) {
      int topsub = k % top_shape_data[j];
      top_index += topsub*top_offset;
      top_offset *= top_shape_data[j];

      int bottomsub = topsub+crop_data[j];
      bottom_index += bottomsub*bottom_offset;
      bottom_offset *= bottom_shape_data[j];
      k /= top_shape_data[j];
    }
    // Copy line top_diff -> bottom_diff
    for (int i = 0; i < line_size; ++i) {
      bottom[bottom_index + i] = top[top_index + i];
    }
    
  }
}

template <typename Dtype>
void CropLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  //const int* crop_data = crop_.cpu_data();
  //for (int i = 0; i < num_axes_-2;++i)
    //printf("%d, ",crop_data[i]);
  //printf("\n");

  copy_kernel_forward<<<CAFFE_GET_BLOCKS(num_lines_), CAFFE_CUDA_NUM_THREADS>>>(
      num_lines_, line_size_, num_axes_,
      top_shape_.gpu_data(), bottom_shape_.gpu_data(),
      crop_.gpu_data(), bottom_data, top_data);
}

template <typename Dtype>
void CropLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  if (propagate_down[0]) {
    caffe_gpu_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
    // NOLINT_NEXT_LINE(whitespace/operators)
    copy_kernel_backward<<<CAFFE_GET_BLOCKS(num_lines_), CAFFE_CUDA_NUM_THREADS>>>(
      num_lines_, line_size_, num_axes_,
      top_shape_.gpu_data(), bottom_shape_.gpu_data(),
      crop_.gpu_data(), top_diff, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CropLayer);

}  // namespace caffe
