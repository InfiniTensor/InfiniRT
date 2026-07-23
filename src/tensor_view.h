#ifndef INFINI_RT_TENSOR_VIEW_H_
#define INFINI_RT_TENSOR_VIEW_H_

#include <cstdint>
#include <iterator>
#include <string>
#include <type_traits>
#include <utility>

#include "common/small_vector.h"
#include "data_type.h"
#include "device.h"
#include "hash.h"

namespace infini::rt {

namespace tensor_view_detail {

inline constexpr std::size_t kInlineMetadataCapacity = 4;

template <typename Metadata, typename Range>
Metadata CopyMetadata(const Range& range) {
  return Metadata(std::begin(range), std::end(range));
}

template <typename T, typename = void>
struct IsTensorLike : std::false_type {};

template <typename T>
struct IsTensorLike<T,
                    std::void_t<decltype(std::declval<const T&>().data()),
                                decltype(std::declval<const T&>().shape()),
                                decltype(std::declval<const T&>().dtype()),
                                decltype(std::declval<const T&>().device()),
                                decltype(std::declval<const T&>().strides())>>
    : std::true_type {};

}  // namespace tensor_view_detail

class TensorView {
 public:
  using Size = std::size_t;

  using Stride = std::ptrdiff_t;

  using Index = Stride;

  using Shape =
      detail::SmallVector<Size, tensor_view_detail::kInlineMetadataCapacity>;

  using Strides =
      detail::SmallVector<Stride, tensor_view_detail::kInlineMetadataCapacity>;

  template <typename TensorLike,
            typename = std::enable_if_t<
                tensor_view_detail::IsTensorLike<TensorLike>::value>>
  TensorView(const TensorLike& tensor)
      : data_{const_cast<void*>(static_cast<const void*>(tensor.data()))},
        shape_{tensor_view_detail::CopyMetadata<Shape>(tensor.shape())},
        dtype_{tensor.dtype()},
        device_{tensor.device()},
        strides_{
            tensor_view_detail::CopyMetadata<Strides>(tensor.strides())} {}

  TensorView(void* data, Shape shape)
      : data_{data},
        shape_{std::move(shape)},
        dtype_{DefaultDataType()},
        device_{DefaultDevice()},
        strides_{DefaultStrides(shape_)} {}

  template <typename ShapeLike>
  TensorView(void* data, const ShapeLike& shape)
      : data_{data},
        shape_{std::begin(shape), std::end(shape)},
        dtype_{DefaultDataType()},
        device_{DefaultDevice()},
        strides_{DefaultStrides(shape_)} {}

  TensorView(void* data, Shape shape, const DataType& dtype)
      : data_{data},
        shape_{std::move(shape)},
        dtype_{dtype},
        device_{DefaultDevice()},
        strides_{DefaultStrides(shape_)} {}

  template <typename ShapeLike>
  TensorView(void* data, const ShapeLike& shape, const DataType& dtype)
      : data_{data},
        shape_{std::begin(shape), std::end(shape)},
        dtype_{dtype},
        device_{DefaultDevice()},
        strides_{DefaultStrides(shape_)} {}

  TensorView(void* data, Shape shape, const Device& device)
      : data_{data},
        shape_{std::move(shape)},
        dtype_{DefaultDataType()},
        device_{device},
        strides_{DefaultStrides(shape_)} {}

  template <typename ShapeLike>
  TensorView(void* data, const ShapeLike& shape, const Device& device)
      : data_{data},
        shape_{std::begin(shape), std::end(shape)},
        dtype_{DefaultDataType()},
        device_{device},
        strides_{DefaultStrides(shape_)} {}

  TensorView(void* data, Shape shape, const DataType& dtype,
             const Device& device)
      : data_{data},
        shape_{std::move(shape)},
        dtype_{dtype},
        device_{device},
        strides_{DefaultStrides(shape_)} {}

  template <typename ShapeLike>
  TensorView(void* data, const ShapeLike& shape, const DataType& dtype,
             const Device& device)
      : data_{data},
        shape_{std::begin(shape), std::end(shape)},
        dtype_{dtype},
        device_{device},
        strides_{DefaultStrides(shape_)} {}

  TensorView(void* data, Shape shape, const DataType& dtype,
             const Device& device, Strides strides)
      : data_{data},
        shape_{std::move(shape)},
        dtype_{dtype},
        device_{device},
        strides_{std::move(strides)} {}

  template <typename ShapeLike, typename StridesLike>
  TensorView(void* data, const ShapeLike& shape, const DataType& dtype,
             const Device& device, const StridesLike& strides)
      : data_{data},
        shape_{std::begin(shape), std::end(shape)},
        dtype_{dtype},
        device_{device},
        strides_{std::begin(strides), std::end(strides)} {}

  TensorView(void* data, std::initializer_list<Size> shape,
             const DataType& dtype, const Device& device,
             std::initializer_list<Stride> strides);

  TensorView operator[](const Index& index) const;

  void*& data();

  const void* data() const;

  const DataType& dtype() const;

  const Device& device() const;

  const Shape& shape() const;

  const Strides& strides() const;

  Size size(const Index& index) const;

  Stride stride(const Index& index) const;

  Size ndim() const;

  Size element_size() const;

  Size numel() const;

  TensorView T() const;

  std::string ToString() const;

  bool HasBroadcastDim() const;

  bool IsContiguous() const;

 private:
  static const DataType DefaultDataType();

  static Device DefaultDevice();

  static Strides DefaultStrides(const Shape& shape);

  std::string ToStringHelper() const;

  bool IsMergeable(Size dim_start, Size dim_end) const;

  void* data_{nullptr};

  Shape shape_;

  const DataType dtype_;

  Device device_;

  Strides strides_;
};

}  // namespace infini::rt

template <>
struct std::hash<infini::rt::TensorView> {
  std::size_t operator()(const infini::rt::TensorView& tensor) const {
    std::size_t seed{0};

    for (const auto& size : tensor.shape()) {
      HashCombine(seed, size);
    }

    HashCombine(seed, tensor.dtype());

    HashCombine(seed, tensor.device());

    for (const auto& stride : tensor.strides()) {
      HashCombine(seed, stride);
    }

    return seed;
  }
};

template <>
struct std::equal_to<infini::rt::TensorView> {
  bool operator()(const infini::rt::TensorView& a,
                  const infini::rt::TensorView& b) const {
    return a.dtype() == b.dtype() && a.device() == b.device() &&
           a.shape() == b.shape() && a.strides() == b.strides();
  }
};

#endif
