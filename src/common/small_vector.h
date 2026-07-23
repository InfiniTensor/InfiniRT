#ifndef INFINI_RT_COMMON_SMALL_VECTOR_H_
#define INFINI_RT_COMMON_SMALL_VECTOR_H_

#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace infini::rt::detail {

template <typename T, std::size_t InlineCapacity>
class SmallVector;

template <typename T>
struct IsSmallVector : std::false_type {};

template <typename T, std::size_t InlineCapacity>
struct IsSmallVector<SmallVector<T, InlineCapacity>> : std::true_type {};

template <typename Range, typename T, typename = void>
struct IsCompatibleContainer : std::false_type {};

template <typename Range, typename T>
struct IsCompatibleContainer<
    Range, T,
    std::void_t<decltype(std::begin(std::declval<const Range&>())),
                decltype(std::end(std::declval<const Range&>())),
                decltype(static_cast<T>(
                    *std::begin(std::declval<const Range&>())))>>
    : std::true_type {};

template <typename Range, typename T, typename = void>
struct IsEqualityComparableRange : std::false_type {};

template <typename Range, typename T>
struct IsEqualityComparableRange<
    Range, T,
    std::void_t<decltype(std::begin(std::declval<const Range&>())),
                decltype(std::end(std::declval<const Range&>())),
                decltype(std::size(std::declval<const Range&>())),
                decltype(static_cast<bool>(
                    std::declval<const T&>() ==
                    *std::begin(std::declval<const Range&>())))>>
    : std::true_type {};

template <typename T, std::size_t InlineCapacity>
class SmallVector {
  static_assert(InlineCapacity > 0,
                "SmallVector requires a positive inline capacity.");

  static_assert(std::is_trivially_copyable_v<T>,
                "SmallVector requires T to be trivially copyable.");

  static_assert(std::is_trivially_destructible_v<T>,
                "SmallVector requires T to be trivially destructible.");

  static_assert(std::is_nothrow_default_constructible_v<T>,
                "SmallVector requires T to be nothrow default constructible.");

  static_assert(std::is_nothrow_copy_constructible_v<T>,
                "SmallVector requires T to be nothrow copy constructible.");

 public:
  using value_type = T;

  using size_type = std::size_t;

  using iterator = T*;

  using const_iterator = const T*;

  SmallVector() = default;

  explicit SmallVector(size_type count) { InitializeCount(count); }

  SmallVector(std::initializer_list<T> values)
      : SmallVector(values.begin(), values.end()) {}

  template <typename InputIt,
            std::enable_if_t<!std::is_integral_v<InputIt>, int> = 0>
  SmallVector(InputIt first, InputIt last) {
    SmallVector replacement;
    replacement.InitializeRange(first, last);
    MoveConstructFrom(replacement);
  }

  template <
      typename Container,
      std::enable_if_t<
          !std::is_same_v<std::decay_t<Container>, SmallVector> &&
              IsCompatibleContainer<Container, T>::value,
          int> = 0>
  explicit SmallVector(const Container& container)
      : SmallVector(std::begin(container), std::end(container)) {}

  SmallVector(const SmallVector& other) { CopyConstructFrom(other); }

  SmallVector(SmallVector&& other) noexcept { MoveConstructFrom(other); }

  SmallVector& operator=(const SmallVector& other) {
    if (this == &other) return *this;

    if (other.IsHeap()) {
      CopyAssignHeap(other);
    } else {
      AssignInline(other.data(), other.size_);
    }

    return *this;
  }

  SmallVector& operator=(SmallVector&& other) noexcept {
    if (this == &other) return *this;

    if (other.IsHeap()) {
      MoveAssignHeap(other);
    } else {
      AssignInline(other.data(), other.size_);
      other.clear();
    }

    return *this;
  }

  ~SmallVector() {
    if (IsHeap()) Deallocate(storage_.heap_data, capacity_);
  }

  size_type size() const noexcept { return size_; }

  size_type capacity() const noexcept { return capacity_; }

  bool empty() const noexcept { return size_ == 0; }

  T* data() noexcept {
    return IsHeap() ? storage_.heap_data : storage_.inline_data;
  }

  const T* data() const noexcept {
    return IsHeap() ? storage_.heap_data : storage_.inline_data;
  }

  T& front() noexcept { return data()[0]; }

  const T& front() const noexcept { return data()[0]; }

  T& back() noexcept { return data()[size_ - 1]; }

  const T& back() const noexcept { return data()[size_ - 1]; }

  T& operator[](size_type index) noexcept { return data()[index]; }

  const T& operator[](size_type index) const noexcept { return data()[index]; }

  iterator begin() noexcept { return data(); }

  const_iterator begin() const noexcept { return data(); }

  const_iterator cbegin() const noexcept { return data(); }

  iterator end() noexcept { return data() + size_; }

  const_iterator end() const noexcept { return data() + size_; }

  const_iterator cend() const noexcept { return data() + size_; }

  void clear() noexcept { size_ = 0; }

  void reserve(size_type requested_capacity) {
    if (requested_capacity <= capacity_) return;

    Reallocate(requested_capacity);
  }

  void resize(size_type count) {
    if (count <= size_) {
      size_ = count;
      return;
    }

    if (count > capacity_) Reallocate(count);

    while (size_ < count) {
      ConstructValue(data() + size_);
      ++size_;
    }
  }

  void push_back(const T& value) {
    if (size_ == capacity_) {
      T saved_value(value);
      GrowForAppend();
      Construct(data() + size_, saved_value);
    } else {
      Construct(data() + size_, value);
    }

    ++size_;
  }

  template <typename InputIt,
            std::enable_if_t<!std::is_integral_v<InputIt>, int> = 0>
  void assign(InputIt first, InputIt last) {
    SmallVector replacement(first, last);
    *this = std::move(replacement);
  }

  void assign(std::initializer_list<T> values) {
    assign(values.begin(), values.end());
  }

 private:
  using Allocator = std::allocator<T>;

  using AllocatorTraits = std::allocator_traits<Allocator>;

  union Storage {
    T inline_data[InlineCapacity];

    T* heap_data;

    constexpr Storage() : inline_data{} {}
  };

  bool IsHeap() const noexcept { return capacity_ > InlineCapacity; }

  static void Construct(T* destination, const T& value) {
    ::new (static_cast<void*>(destination)) T(value);
  }

  static void ConstructValue(T* destination) {
    ::new (static_cast<void*>(destination)) T{};
  }

  static void Deallocate(T* pointer, size_type capacity) noexcept {
    Allocator allocator;
    AllocatorTraits::deallocate(allocator, pointer, capacity);
  }

  void InitializeCount(size_type count) {
    if (count > capacity_) Reallocate(count);

    while (size_ < count) {
      ConstructValue(data() + size_);
      ++size_;
    }
  }

  template <typename InputIt>
  void InitializeRange(InputIt first, InputIt last) {
    using IteratorCategory =
        typename std::iterator_traits<InputIt>::iterator_category;

    if constexpr (
        std::is_base_of_v<std::forward_iterator_tag, IteratorCategory>) {
      const auto distance = std::distance(first, last);
      const size_type count = static_cast<size_type>(distance);
      if (count > capacity_) Reallocate(count);

      for (; first != last; ++first) {
        Construct(data() + size_, static_cast<T>(*first));
        ++size_;
      }
    } else {
      for (; first != last; ++first) push_back(static_cast<T>(*first));
    }
  }

  void CopyConstructFrom(const SmallVector& other) {
    if (other.IsHeap()) Reallocate(other.capacity_);

    for (; size_ < other.size_; ++size_) {
      Construct(data() + size_, other.data()[size_]);
    }
  }

  void MoveConstructFrom(SmallVector& other) noexcept {
    if (other.IsHeap()) {
      T* heap_data = other.storage_.heap_data;
      ::new (static_cast<void*>(&storage_.heap_data)) T*(heap_data);
      size_ = other.size_;
      capacity_ = other.capacity_;
      other.ReconstructInline();
      return;
    }

    for (; size_ < other.size_; ++size_) {
      Construct(data() + size_, other.data()[size_]);
    }
    other.clear();
  }

  void CopyAssignHeap(const SmallVector& other) {
    Allocator allocator;
    T* new_data = AllocatorTraits::allocate(allocator, other.capacity_);
    for (size_type index = 0; index < other.size_; ++index) {
      Construct(new_data + index, other.data()[index]);
    }

    ReplaceWithHeap(new_data, other.size_, other.capacity_);
  }

  void MoveAssignHeap(SmallVector& other) {
    T* heap_data = other.storage_.heap_data;
    ReplaceWithHeap(heap_data, other.size_, other.capacity_);
    other.ReconstructInline();
  }

  void AssignInline(const T* values, size_type count) {
    if (IsHeap()) SwitchToInline();

    for (size_type index = 0; index < count; ++index) {
      Construct(storage_.inline_data + index, values[index]);
    }
    size_ = count;
  }

  void ReplaceWithHeap(T* new_data, size_type new_size,
                       size_type new_capacity) {
    T* old_data = nullptr;
    size_type old_capacity = 0;

    if (IsHeap()) {
      old_data = storage_.heap_data;
      old_capacity = capacity_;
      storage_.heap_data = new_data;
    } else {
      ::new (static_cast<void*>(&storage_.heap_data)) T*(new_data);
    }

    size_ = new_size;
    capacity_ = new_capacity;
    if (old_data != nullptr) Deallocate(old_data, old_capacity);
  }

  void SwitchToInline() {
    T* old_data = storage_.heap_data;
    const size_type old_capacity = capacity_;
    ReconstructInline();
    Deallocate(old_data, old_capacity);
  }

  void ReconstructInline() {
    storage_.~Storage();
    ::new (static_cast<void*>(&storage_)) Storage();
    size_ = 0;
    capacity_ = InlineCapacity;
  }

  void Reallocate(size_type new_capacity) {
    Allocator allocator;
    T* new_data = AllocatorTraits::allocate(allocator, new_capacity);
    const T* old_data = data();
    for (size_type index = 0; index < size_; ++index) {
      Construct(new_data + index, old_data[index]);
    }

    ReplaceWithHeap(new_data, size_, new_capacity);
  }

  void GrowForAppend() {
    Allocator allocator;
    const size_type max_size = AllocatorTraits::max_size(allocator);
    size_type increment = capacity_ / 2;
    if (increment == 0) increment = 1;

    const size_type new_capacity =
        increment > max_size - capacity_ ? max_size : capacity_ + increment;
    Reallocate(new_capacity);
  }

  Storage storage_;

  size_type size_{0};

  size_type capacity_{InlineCapacity};
};

template <
    typename T, std::size_t LeftCapacity, std::size_t RightCapacity,
    std::enable_if_t<IsEqualityComparableRange<
                         SmallVector<T, RightCapacity>, T>::value,
                     int> = 0>
bool operator==(const SmallVector<T, LeftCapacity>& left,
                const SmallVector<T, RightCapacity>& right) {
  if (left.size() != right.size()) return false;

  for (std::size_t index = 0; index < left.size(); ++index) {
    if (!(left[index] == right[index])) return false;
  }

  return true;
}

template <
    typename T, std::size_t LeftCapacity, std::size_t RightCapacity,
    std::enable_if_t<IsEqualityComparableRange<
                         SmallVector<T, RightCapacity>, T>::value,
                     int> = 0>
bool operator!=(const SmallVector<T, LeftCapacity>& left,
                const SmallVector<T, RightCapacity>& right) {
  return !(left == right);
}

template <
    typename T, std::size_t InlineCapacity, typename Range,
    std::enable_if_t<
        !IsSmallVector<std::decay_t<Range>>::value &&
            IsEqualityComparableRange<Range, T>::value,
        int> = 0>
bool operator==(const SmallVector<T, InlineCapacity>& left,
                const Range& right) {
  if (left.size() != static_cast<std::size_t>(std::size(right))) return false;

  auto right_iterator = std::begin(right);
  for (std::size_t index = 0; index < left.size();
       ++index, ++right_iterator) {
    if (!(left[index] == *right_iterator)) return false;
  }

  return true;
}

template <
    typename Range, typename T, std::size_t InlineCapacity,
    std::enable_if_t<
        !IsSmallVector<std::decay_t<Range>>::value &&
            IsEqualityComparableRange<Range, T>::value,
        int> = 0>
bool operator==(const Range& left,
                const SmallVector<T, InlineCapacity>& right) {
  return right == left;
}

template <
    typename T, std::size_t InlineCapacity, typename Range,
    std::enable_if_t<
        !IsSmallVector<std::decay_t<Range>>::value &&
            IsEqualityComparableRange<Range, T>::value,
        int> = 0>
bool operator!=(const SmallVector<T, InlineCapacity>& left,
                const Range& right) {
  return !(left == right);
}

template <
    typename Range, typename T, std::size_t InlineCapacity,
    std::enable_if_t<
        !IsSmallVector<std::decay_t<Range>>::value &&
            IsEqualityComparableRange<Range, T>::value,
        int> = 0>
bool operator!=(const Range& left,
                const SmallVector<T, InlineCapacity>& right) {
  return !(right == left);
}

}  // namespace infini::rt::detail

#endif
