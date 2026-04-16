#include "tensor_view.h"

#include <algorithm>
#include <cassert>
#include <numeric>

namespace infini::rt {

static TensorView::Index GetEffectiveIndex(TensorView::Index index,
                                           TensorView::Size size) {
  return index < 0 ? index + size : index;
}

TensorView::TensorView(void* data, std::initializer_list<Size> shape,
                       const DataType& dtype, const Device& device,
                       std::initializer_list<Stride> strides)
    : TensorView{data, decltype(shape_){shape}, dtype, device,
                 decltype(strides_){strides}} {}

TensorView TensorView::operator[](const Index& index) const {
  return {
      reinterpret_cast<decltype(data_)>(
          reinterpret_cast<decltype(index)>(data_) +
          GetEffectiveIndex(index, shape_[0]) * strides_[0] * element_size()),
      Shape{shape_.cbegin() + 1, shape_.cend()}, dtype_, device_,
      Strides{strides_.cbegin() + 1, strides_.cend()}};
}

void*& TensorView::data() { return data_; }

const void* TensorView::data() const { return data_; }

const TensorView::Shape& TensorView::shape() const { return shape_; }

const DataType& TensorView::dtype() const { return dtype_; }

const Device& TensorView::device() const { return device_; }

const TensorView::Strides& TensorView::strides() const { return strides_; }

TensorView::Size TensorView::size(const Index& index) const {
  return shape_[GetEffectiveIndex(index, shape_.size())];
}

TensorView::Stride TensorView::stride(const Index& index) const {
  return strides_[GetEffectiveIndex(index, strides_.size())];
}

TensorView::Size TensorView::ndim() const { return shape_.size(); }

TensorView::Size TensorView::element_size() const {
  return kDataTypeToSize.at(dtype_);
}

TensorView::Size TensorView::numel() const {
  return std::accumulate(
      shape_.begin(), shape_.end(), static_cast<TensorView::Size>(1),
      [](TensorView::Size a, TensorView::Size b) { return a * b; });
}

TensorView TensorView::T() const {
  return {data_,
          {shape_[1], shape_[0]},
          dtype_,
          device_,
          {strides_[1], strides_[0]}};
}

std::string TensorView::ToString() const {
  return "tensor(" + ToStringHelper() +
         ", dtype=" + std::string(kDataTypeToDesc.at(dtype_)) + ", device='" +
         device_.ToString() + "')";
}

bool TensorView::HasBroadcastDim() const {
  return std::any_of(shape_.begin(), shape_.end(),
                     [&, i = 0](const auto&) mutable {
                       return shape_[i] != 1 && strides_[i++] == 0;
                     });
}

bool TensorView::IsContiguous() const {
  if (ndim() == 0) {
    return true;
  }

  if (!IsMergeable(0, ndim() - 1)) {
    return false;
  }

  return stride(ndim() - 1) == 1;
}

const DataType TensorView::DefaultDataType() { return DataType::kFloat32; }

Device TensorView::DefaultDevice() { return Device{Device::Type::kCpu}; }

TensorView::Strides TensorView::DefaultStrides(const Shape& shape) {
  if (shape.empty()) {
    return {};
  }

  Strides strides(shape.size());

  strides.back() = 1;

  for (auto i{shape.size() - 2}; i != -1; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  return strides;
}

std::string TensorView::ToStringHelper() const {
  if (ndim() == 0) {
    switch (dtype_) {
      case DataType::kFloat16:
        return std::to_string(static_cast<Float16*>(data_)->ToFloat());
      case DataType::kBFloat16:
        return std::to_string(static_cast<BFloat16*>(data_)->ToFloat());
      case DataType::kFloat32:
        return std::to_string(*static_cast<float*>(data_));
      case DataType::kFloat64:
        return std::to_string(*static_cast<double*>(data_));
      case DataType::kInt8:
        return std::to_string(*static_cast<int8_t*>(data_));
      case DataType::kInt16:
        return std::to_string(*static_cast<int16_t*>(data_));
      case DataType::kInt32:
        return std::to_string(*static_cast<int32_t*>(data_));
      case DataType::kInt64:
        return std::to_string(*static_cast<int64_t*>(data_));
      case DataType::kUInt8:
        return std::to_string(*static_cast<uint8_t*>(data_));
      case DataType::kUInt16:
        return std::to_string(*static_cast<uint16_t*>(data_));
      case DataType::kUInt32:
        return std::to_string(*static_cast<uint32_t*>(data_));
      case DataType::kUInt64:
        return std::to_string(*static_cast<uint64_t*>(data_));
      default:
        return "?";
    }
  }

  std::string result{"["};

  for (auto i{Index{0}}; i < shape_[0]; ++i) {
    result += operator[](i).ToStringHelper() + ", ";
  }

  result.pop_back();
  result.back() = ']';

  return result;
}

bool TensorView::IsMergeable(TensorView::Size dim_start,
                             TensorView::Size dim_end) const {
  if (dim_start == dim_end) {
    return true;
  }

  for (TensorView::Size i = dim_start; i < dim_end; ++i) {
    if (size(i) == 1 && stride(i) == 0) {
      return false;
    }
    if (stride(i) != size(i + 1) * stride(i + 1)) {
      return false;
    }
  }

  return true;
}

}  // namespace infini::rt
