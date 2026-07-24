#ifndef INFINI_RT_COMMON_TENSOR_METADATA_H_
#define INFINI_RT_COMMON_TENSOR_METADATA_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

#include "common/metadata_view.h"
#include "common/small_vector.h"

namespace infini::rt::detail {

struct DefaultStridesTag {};

template <typename Range>
using RangeIterator = decltype(std::begin(std::declval<const Range&>()));

template <typename Range, typename = void>
struct IsForwardRange : std::false_type {};

template <typename Range>
struct IsForwardRange<
    Range,
    std::void_t<typename std::iterator_traits<RangeIterator<Range>>::
                    iterator_category>>
    : std::is_base_of<
          std::forward_iterator_tag,
          typename std::iterator_traits<RangeIterator<Range>>::
              iterator_category> {};

template <typename Size, typename Stride, std::size_t InlineCapacity>
class TensorMetadata {
  static_assert(InlineCapacity > 0,
                "Tensor metadata requires a positive inline capacity.");

  static_assert(std::is_trivially_copyable_v<Size> &&
                    std::is_trivially_destructible_v<Size>,
                "Tensor metadata requires a trivial size type.");

  static_assert(std::is_trivially_copyable_v<Stride> &&
                    std::is_trivially_destructible_v<Stride>,
                "Tensor metadata requires a trivial stride type.");

  static_assert(alignof(Size) <= alignof(std::max_align_t) &&
                    alignof(Stride) <= alignof(std::max_align_t),
                "Tensor metadata does not support over-aligned types.");

 public:
  using Shape = SmallVector<Size, InlineCapacity>;

  using Strides = SmallVector<Stride, InlineCapacity>;

  using ShapeView = MetadataView<Size>;

  using StridesView = MetadataView<Stride>;

  TensorMetadata() = default;

  TensorMetadata(const Shape& shape, const Strides& strides) {
    InitializeRanges(shape, strides);
  }

  TensorMetadata(Shape&& shape, Strides&& strides) {
    InitializeOwned(std::move(shape), std::move(strides));
  }

  TensorMetadata(Shape&& shape, const Strides& strides) {
    InitializeMixed(std::move(shape), strides);
  }

  TensorMetadata(const Shape& shape, Strides&& strides) {
    InitializeMixed(shape, std::move(strides));
  }

  template <typename ShapeRange, typename StridesRange>
  TensorMetadata(const ShapeRange& shape, const StridesRange& strides) {
    InitializeRanges(shape, strides);
  }

  TensorMetadata(const Shape& shape, DefaultStridesTag) {
    InitializeDefaultStrides(shape);
  }

  TensorMetadata(Shape&& shape, DefaultStridesTag) {
    InitializeDefaultStrides(std::move(shape));
  }

  template <typename ShapeRange>
  TensorMetadata(const ShapeRange& shape, DefaultStridesTag) {
    InitializeDefaultStrides(shape);
  }

  TensorMetadata(const TensorMetadata& other) {
    InitializeRanges(other.shape(), other.strides());
  }

  TensorMetadata(TensorMetadata&& other) noexcept {
    MoveConstructFrom(other);
  }

  TensorMetadata& operator=(const TensorMetadata&) = delete;

  TensorMetadata& operator=(TensorMetadata&&) = delete;

  ~TensorMetadata() { ReleaseStorage(); }

  ShapeView shape() const noexcept {
    return ShapeView{ShapeData(), shape_size_};
  }

  StridesView strides() const noexcept {
    return StridesView{StridesData(), strides_size_};
  }

 private:
  using ShapeAllocation = typename Shape::HeapAllocation;

  using StridesAllocation = typename Strides::HeapAllocation;

  struct InlineStorage {
    Size shape[InlineCapacity];

    Stride strides[InlineCapacity];

    InlineStorage() noexcept {}
  };

  struct HeapStorage {
    void* allocation;

    Size* shape;

    Stride* strides;

    std::size_t shape_capacity;

    std::size_t strides_capacity;
  };

  union Storage {
    InlineStorage inline_storage;

    HeapStorage heap_storage;

    Storage() noexcept : inline_storage{} {}

    ~Storage() {}
  };

  class CombinedAllocation {
   public:
    CombinedAllocation(std::size_t shape_size, std::size_t strides_size) {
      const std::size_t shape_bytes = CheckedMultiply(shape_size, sizeof(Size));
      const std::size_t strides_bytes =
          CheckedMultiply(strides_size, sizeof(Stride));
      const std::size_t padding = alignof(Stride) - 1;
      const std::size_t bytes =
          CheckedAdd(CheckedAdd(shape_bytes, padding), strides_bytes);

      RawAllocation allocation{::operator new(bytes)};
      void* stride_storage =
          static_cast<void*>(static_cast<unsigned char*>(allocation.get()) +
                             shape_bytes);
      std::size_t stride_space = bytes - shape_bytes;

      const void* const aligned_stride_storage =
          std::align(alignof(Stride), strides_bytes, stride_storage,
                     stride_space);
      if (aligned_stride_storage == nullptr) {
        std::abort();
      }

      shape_ = shape_size == 0
                   ? static_cast<Size*>(allocation.get())
                   : ::new (allocation.get()) Size[shape_size];
      strides_ = strides_size == 0
                     ? static_cast<Stride*>(stride_storage)
                     : ::new (stride_storage) Stride[strides_size];
      allocation_ = allocation.release();
    }

    CombinedAllocation(const CombinedAllocation&) = delete;

    CombinedAllocation& operator=(const CombinedAllocation&) = delete;

    ~CombinedAllocation() { ::operator delete(allocation_); }

    void* allocation() const noexcept { return allocation_; }

    Size* shape() const noexcept { return shape_; }

    Stride* strides() const noexcept { return strides_; }

    void release() noexcept { allocation_ = nullptr; }

   private:
    struct RawDeleter {
      void operator()(void* allocation) const noexcept {
        ::operator delete(allocation);
      }
    };

    using RawAllocation = std::unique_ptr<void, RawDeleter>;

    static std::size_t CheckedMultiply(std::size_t left,
                                       std::size_t right) {
      if (right != 0 &&
          left > std::numeric_limits<std::size_t>::max() / right) {
        std::abort();
      }

      return left * right;
    }

    static std::size_t CheckedAdd(std::size_t left, std::size_t right) {
      if (left > std::numeric_limits<std::size_t>::max() - right) {
        std::abort();
      }

      return left + right;
    }

    void* allocation_{nullptr};

    Size* shape_{nullptr};

    Stride* strides_{nullptr};
  };

  bool IsInline() const noexcept {
    return shape_size_ <= InlineCapacity && strides_size_ <= InlineCapacity;
  }

  bool IsCombined() const noexcept {
    return !IsInline() && storage_.heap_storage.allocation != nullptr;
  }

  static std::uint32_t NarrowSize(std::size_t size) {
    if (size > std::numeric_limits<std::uint32_t>::max()) {
      std::abort();
    }

    return static_cast<std::uint32_t>(size);
  }

  template <typename Range>
  static std::size_t RangeSize(const Range& range) {
    const auto first = std::begin(range);
    const auto last = std::end(range);
    if (first == last) return 0;

    const auto distance = std::distance(first, last);
    assert(distance >= 0);

    return static_cast<std::size_t>(distance);
  }

  template <typename Range, typename T>
  static void CopyRange(const Range& range, T* destination) {
    for (const auto& value : range) {
      *destination++ = static_cast<T>(value);
    }
  }

  template <typename ShapeRange, typename StridesRange>
  void InitializeRanges(const ShapeRange& shape,
                        const StridesRange& strides) {
    if constexpr (IsForwardRange<ShapeRange>::value &&
                  IsForwardRange<StridesRange>::value) {
      const std::size_t shape_size = RangeSize(shape);
      const std::size_t strides_size = RangeSize(strides);
      Initialize(shape_size, strides_size, [&](Size* shape_destination,
                                                Stride* strides_destination) {
        CopyRange(shape, shape_destination);
        CopyRange(strides, strides_destination);
      });
    } else {
      Shape owned_shape{std::begin(shape), std::end(shape)};
      Strides owned_strides{std::begin(strides), std::end(strides)};
      InitializeOwned(std::move(owned_shape), std::move(owned_strides));
    }
  }

  template <typename Writer>
  void Initialize(std::size_t shape_size, std::size_t strides_size,
                  Writer&& writer) {
    const std::uint32_t narrowed_shape_size = NarrowSize(shape_size);
    const std::uint32_t narrowed_strides_size = NarrowSize(strides_size);

    if (shape_size <= InlineCapacity && strides_size <= InlineCapacity) {
      std::forward<Writer>(writer)(storage_.inline_storage.shape,
                                   storage_.inline_storage.strides);
      shape_size_ = narrowed_shape_size;
      strides_size_ = narrowed_strides_size;

      return;
    }

    CombinedAllocation allocation{shape_size, strides_size};
    std::forward<Writer>(writer)(allocation.shape(), allocation.strides());
    ActivateCombined(allocation, narrowed_shape_size, narrowed_strides_size);
  }

  void InitializeOwned(Shape&& shape, Strides&& strides) {
    const std::size_t shape_size = shape.size();
    const std::size_t strides_size = strides.size();

    if (shape_size <= InlineCapacity && strides_size <= InlineCapacity) {
      InitializeRanges(shape, strides);

      return;
    }

    if (shape.capacity() > InlineCapacity &&
        strides.capacity() > InlineCapacity) {
      const std::uint32_t narrowed_shape_size = NarrowSize(shape_size);
      const std::uint32_t narrowed_strides_size = NarrowSize(strides_size);
      ShapeAllocation shape_allocation = shape.ReleaseHeap();
      StridesAllocation strides_allocation = strides.ReleaseHeap();
      ActivateSplit(std::move(shape_allocation),
                    std::move(strides_allocation), narrowed_shape_size,
                    narrowed_strides_size);

      return;
    }

    InitializeRanges(shape, strides);
  }

  void InitializeMixed(Shape&& shape, const Strides& strides) {
    if (shape.size() > InlineCapacity && strides.size() > InlineCapacity &&
        shape.capacity() > InlineCapacity) {
      Strides owned_strides{strides};
      InitializeOwned(std::move(shape), std::move(owned_strides));

      return;
    }

    InitializeRanges(shape, strides);
  }

  void InitializeMixed(const Shape& shape, Strides&& strides) {
    if (shape.size() > InlineCapacity && strides.size() > InlineCapacity &&
        strides.capacity() > InlineCapacity) {
      Shape owned_shape{shape};
      InitializeOwned(std::move(owned_shape), std::move(strides));

      return;
    }

    InitializeRanges(shape, strides);
  }

  template <typename ShapeRange>
  void InitializeDefaultStrides(const ShapeRange& shape) {
    if constexpr (IsForwardRange<ShapeRange>::value) {
      const std::size_t shape_size = RangeSize(shape);
      Initialize(shape_size, shape_size, [&](Size* shape_destination,
                                             Stride* strides_destination) {
        CopyRange(shape, shape_destination);
        FillDefaultStrides(shape_destination, shape_size,
                           strides_destination);
      });
    } else {
      Shape owned_shape{std::begin(shape), std::end(shape)};
      InitializeDefaultStrides(std::move(owned_shape));
    }
  }

  void InitializeDefaultStrides(Shape&& shape) {
    if (shape.size() <= InlineCapacity ||
        shape.capacity() <= InlineCapacity) {
      InitializeDefaultStrides(static_cast<const Shape&>(shape));

      return;
    }

    Strides strides(shape.size());
    FillDefaultStrides(shape.data(), shape.size(), strides.data());
    InitializeOwned(std::move(shape), std::move(strides));
  }

  static void FillDefaultStrides(const Size* shape, std::size_t shape_size,
                                 Stride* strides) {
    if (shape_size == 0) return;

    strides[shape_size - 1] = 1;

    for (std::size_t index = shape_size - 1; index > 0; --index) {
      strides[index - 1] =
          strides[index] * shape[index];
    }
  }

  void ActivateCombined(CombinedAllocation& allocation,
                        std::uint32_t shape_size,
                        std::uint32_t strides_size) noexcept {
    storage_.inline_storage.~InlineStorage();
    ::new (static_cast<void*>(&storage_.heap_storage)) HeapStorage{
        allocation.allocation(), allocation.shape(), allocation.strides(), 0,
        0};
    shape_size_ = shape_size;
    strides_size_ = strides_size;
    allocation.release();
  }

  void ActivateSplit(ShapeAllocation&& shape,
                     StridesAllocation&& strides, std::uint32_t shape_size,
                     std::uint32_t strides_size) noexcept {
    const std::size_t shape_capacity = shape.capacity();
    const std::size_t strides_capacity = strides.capacity();
    Size* shape_data = shape.release();
    Stride* strides_data = strides.release();

    storage_.inline_storage.~InlineStorage();
    ::new (static_cast<void*>(&storage_.heap_storage)) HeapStorage{
        nullptr, shape_data, strides_data, shape_capacity, strides_capacity};
    shape_size_ = shape_size;
    strides_size_ = strides_size;
  }

  void MoveConstructFrom(TensorMetadata& other) noexcept {
    if (other.IsInline()) {
      Initialize(other.shape_size_, other.strides_size_,
                 [&](Size* shape_destination, Stride* strides_destination) {
                   for (std::size_t index = 0; index < other.shape_size_;
                        ++index) {
                     shape_destination[index] = other.ShapeData()[index];
                   }
                   for (std::size_t index = 0; index < other.strides_size_;
                        ++index) {
                     strides_destination[index] = other.StridesData()[index];
                   }
                 });
      other.shape_size_ = 0;
      other.strides_size_ = 0;

      return;
    }

    storage_.inline_storage.~InlineStorage();
    ::new (static_cast<void*>(&storage_.heap_storage))
        HeapStorage{other.storage_.heap_storage};
    shape_size_ = other.shape_size_;
    strides_size_ = other.strides_size_;
    other.storage_.heap_storage.~HeapStorage();
    ::new (static_cast<void*>(&other.storage_.inline_storage)) InlineStorage{};
    other.shape_size_ = 0;
    other.strides_size_ = 0;
  }

  const Size* ShapeData() const noexcept {
    return IsInline() ? storage_.inline_storage.shape
                      : storage_.heap_storage.shape;
  }

  const Stride* StridesData() const noexcept {
    return IsInline() ? storage_.inline_storage.strides
                      : storage_.heap_storage.strides;
  }

  void ReleaseStorage() noexcept {
    if (IsInline()) return;

    if (IsCombined()) {
      ::operator delete(storage_.heap_storage.allocation);

      return;
    }

    std::allocator<Size> shape_allocator;
    std::allocator_traits<std::allocator<Size>>::deallocate(
        shape_allocator, storage_.heap_storage.shape,
        storage_.heap_storage.shape_capacity);
    std::allocator<Stride> strides_allocator;
    std::allocator_traits<std::allocator<Stride>>::deallocate(
        strides_allocator, storage_.heap_storage.strides,
        storage_.heap_storage.strides_capacity);
  }

  Storage storage_;

  std::uint32_t shape_size_{0};

  std::uint32_t strides_size_{0};
};

}  // namespace infini::rt::detail

#endif
