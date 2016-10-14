#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void CropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Calculate number of spatial axis
  CropParameter crop_param = this->layer_param_.crop_param();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(crop_param.axis());
  channels_ = bottom[0]->shape(channel_axis_);
  const int first_spatial_axis = channel_axis_ + 1;
  num_axes_ = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes_ - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 1);
  vector<int> dim_blob_shape(1, num_axes_);
  // Construct a map from top blobs to layer inds, skipping over in-place
  // connections.
  CHECK(this->net_!=NULL) << "Crop Layer must be used in a net";
  map<Blob<Dtype>*, int> down_map;
  for (int layer_ind = 0; layer_ind < this->net_->top_vecs().size();
       ++layer_ind) {
    vector<Blob<Dtype>*> tops = this->net_->top_vecs()[layer_ind];
    for (int top_ind = 0; top_ind < tops.size(); ++top_ind) {
      if (down_map.find(tops[top_ind]) == down_map.end()) {
        down_map[tops[top_ind]] = layer_ind;
      }
    }
  }
  // Walk back from the first bottom, keeping track of all the blobs we pass.
  set<Blob<Dtype>*> path_blobs;
  Blob<Dtype>* blob = bottom[0];
  int layer_ind;
  // TODO this logic can be simplified if all blobs are tops
  path_blobs.insert(blob);
  while (down_map.find(blob) != down_map.end()) {
    layer_ind = down_map[blob];
    if (this->net_->bottom_vecs()[layer_ind].size() == 0) {
      break;
    }
    blob = this->net_->bottom_vecs()[layer_ind][0];
    path_blobs.insert(blob);
  }
  // Now walk back from the second bottom, until we find a blob of intersection.
  Blob<Dtype>* inter_blob = bottom[1];
  while (path_blobs.find(inter_blob) == path_blobs.end()) {
    CHECK(down_map.find(inter_blob) != down_map.end())
        << "Cannot align apparently disconnected blobs.";
    layer_ind = down_map[inter_blob];
    CHECK_GT(this->net_->bottom_vecs()[layer_ind].size(), 0)
        << "Cannot align apparently disconnected blobs.";
    inter_blob = this->net_->bottom_vecs()[layer_ind][0];
  }
  // Compute the coord map from the blob of intersection to each bottom.
  vector<DiagonalAffineMap<Dtype> > coord_maps(2,
      DiagonalAffineMap<Dtype>::identity(num_spatial_axes_));
  for (int i = 0; i < 2; ++i) {
    for (Blob<Dtype>* blob = bottom[i]; blob != inter_blob;
         blob = this->net_->bottom_vecs()[down_map[blob]][0]) {
      shared_ptr<Layer<Dtype> > layer = this->net_->layers()[down_map[blob]];
      // printf("[%d] %s\n",i,layer->type());
      coord_maps[i] = coord_maps[i].compose(layer->coord_map(num_spatial_axes_));
      // printf("done [%d] %s\n",i,layer->type());
    }
  }
  // Compute the mapping from first bottom coordinates to second.
  crop_.Reshape(dim_blob_shape);
  top_shape_.Reshape(dim_blob_shape);
  bottom_shape_.Reshape(dim_blob_shape);
  int* crop_data = crop_.mutable_cpu_data();
  int* top_shape_data = top_shape_.mutable_cpu_data();
  int* bottom_shape_data = bottom_shape_.mutable_cpu_data();
 // printf("maps %d %d \n",coord_maps[0].size(), coord_maps[1].size());
  DiagonalAffineMap<Dtype> crop_map =
      coord_maps[1].compose(coord_maps[0].inv());
// printf("Done compute map \n");
// printf("num_axes_ %d \n",num_axes_);
  caffe_set(num_axes_, static_cast<int>(0), crop_data);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    // Check for scale mismatch (unfortunately, CHECK_DOUBLE_EQ does not
    // support a message like the other CHECKs).
    CHECK_DOUBLE_EQ(crop_map.coefs()[i].first, 1);
    CHECK_LE(crop_map.coefs()[i].second, 0) << "Negative crop width.";
    // Check that the crop width is an integer.
    CHECK_DOUBLE_EQ(crop_map.coefs()[i].second,
        round(crop_map.coefs()[i].second));
    crop_data[first_spatial_axis+i] = - round(crop_map.coefs()[i].second);
  }
  // printf("shapes \n");
  for (int i = 0; i < channel_axis_+1; ++i) {
    bottom_shape_data[i] = bottom[0]->shape(i);
    top_shape_data[i] = bottom[0]->shape(i);
  }
    // printf("shapes2 \n");
  for (int i = 0; i < num_spatial_axes_; ++i) {
    bottom_shape_data[first_spatial_axis+i] = bottom[0]->shape(first_spatial_axis+i);
    top_shape_data[first_spatial_axis+i] = bottom[1]->shape(first_spatial_axis+i);
  }
    // printf("line size \n");
  line_size_ = top_shape_data[num_axes_-1];
    // printf("done \n");
}

template <typename Dtype>
void CropLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) { 
  num_ = bottom[0]->count(0, channel_axis_);
  vector<int> reshape = bottom[0]->shape();
  reshape.resize(channel_axis_ + 1);
  for (int i = 0; i < num_spatial_axes_; ++i) {
      reshape.push_back(bottom[1]->shape(channel_axis_+i+1));
  }
  top[0]->Reshape(reshape);
  num_lines_ = top[0]->count()/line_size_;
}

template <typename Dtype>
void CropLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int* crop_data = crop_.cpu_data();
  const int* top_shape_data = top_shape_.cpu_data();
  const int* bottom_shape_data = bottom_shape_.cpu_data();

  for (int line = 0; line < num_lines_; ++line) {
    int k = line;
    int top_index = 0;
    int bottom_index = crop_data[num_axes_-1];
    int top_offset = top_shape_data[num_axes_-1];
    int bottom_offset = bottom_shape_data[num_axes_-1];

    for (int j = num_axes_ - 2; j >= 0; --j) {
      int topsub = k % top_shape_data[j];
      top_index += topsub*top_offset;
      top_offset *= top_shape_data[j];

      int bottomsub = topsub+crop_data[j];
      bottom_index += bottomsub*bottom_offset;
      bottom_offset *= bottom_shape_data[j];
      k /= top_shape_data[j];
    }
    caffe_copy(line_size_,
            bottom_data + bottom_index,
            top_data + top_index);
  }
}

template <typename Dtype>
void CropLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const int* crop_data = crop_.cpu_data();
  const int* top_shape_data = top_shape_.cpu_data();
  const int* bottom_shape_data = bottom_shape_.cpu_data();
  if (propagate_down[0]) {
    caffe_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
      for (int line = 0; line < num_lines_; ++line) {
      int k = line;
      int top_index = 0;
      int bottom_index = crop_data[num_axes_ - 1];
      int top_offset = top_shape_data[num_axes_ - 1];
      int bottom_offset = bottom_shape_data[num_axes_ - 1];

      for (int j = num_axes_ - 2; j >= 0; --j) {
        int topsub = k % top_shape_data[j];
        top_index += topsub*top_offset;
        top_offset *= top_shape_data[j];

        int bottomsub = topsub+crop_data[j];
        bottom_index += bottomsub*bottom_offset;
        bottom_offset *= bottom_shape_data[j];
        if (k)
          k /= top_shape_data[j];
      }
      caffe_copy(line_size_,
              top_diff + top_index,
              bottom_diff + bottom_index);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(CropLayer);
#endif

INSTANTIATE_CLASS(CropLayer);
REGISTER_LAYER_CLASS(Crop);

}  // namespace caffe
