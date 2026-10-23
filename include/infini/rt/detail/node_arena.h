#ifndef INFINI_RT_DETAIL_NODE_ARENA_H_
#define INFINI_RT_DETAIL_NODE_ARENA_H_

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <new>
#include <vector>

namespace infini::rt::detail {

class NodeArena {
 public:
  NodeArena() = default;

  NodeArena(const NodeArena&) = delete;
  NodeArena& operator=(const NodeArena&) = delete;

  ~NodeArena() {
    for (void* block : blocks_) {
      ::operator delete(block, std::align_val_t{node_align_});
    }
  }

  void* Allocate(std::size_t bytes, std::size_t alignment) {
    if (node_size_ == 0) {
      node_size_ = std::max(bytes, sizeof(void*));
      node_align_ = std::max(alignment, alignof(void*));
    }

    if (bytes > node_size_ || alignment > node_align_) {
      return ::operator new(bytes, std::align_val_t{alignment});
    }

    if (free_ == nullptr) {
      Grow();
    }
    void* node = free_;

    std::memcpy(&free_, node, sizeof(void*));
    return node;
  }

  void Deallocate(void* node, std::size_t bytes, std::size_t alignment) {
    if (node == nullptr) {
      return;
    }
    if (bytes > node_size_ || alignment > node_align_) {
      ::operator delete(node, std::align_val_t{alignment});
      return;
    }
    std::memcpy(node, &free_, sizeof(void*));
    free_ = node;
  }

 private:
  static constexpr std::size_t kInitialNodes = 32;
  static constexpr std::size_t kMaxNodesPerBlock = 4096;

  void Grow() {
    const std::size_t count = next_count_;
    next_count_ = std::min(next_count_ * 2, kMaxNodesPerBlock);

    const std::size_t stride =
        (node_size_ + node_align_ - 1) / node_align_ * node_align_;
    void* block =
        ::operator new(stride * count, std::align_val_t{node_align_});
    blocks_.push_back(block);

    char* cursor = static_cast<char*>(block);
    for (std::size_t i = 0; i < count; ++i) {
      Deallocate(cursor + i * stride, node_size_, node_align_);
    }
  }

  std::vector<void*> blocks_;
  void* free_ = nullptr;
  std::size_t node_size_ = 0;
  std::size_t node_align_ = alignof(std::max_align_t);
  std::size_t next_count_ = kInitialNodes;
};

template <typename T>
class ArenaAllocator {
 public:
  using value_type = T;

  explicit ArenaAllocator(NodeArena* arena) : arena_(arena) {}

  template <typename U>
  ArenaAllocator(const ArenaAllocator<U>& other)
      : arena_(other.arena()) {}

  T* allocate(std::size_t count) {
    if (count != 1) {
      return static_cast<T*>(
          ::operator new(count * sizeof(T), std::align_val_t{alignof(T)}));
    }
    return static_cast<T*>(arena_->Allocate(sizeof(T), alignof(T)));
  }

  void deallocate(T* ptr, std::size_t count) {
    if (count != 1) {
      ::operator delete(ptr, std::align_val_t{alignof(T)});
      return;
    }
    arena_->Deallocate(ptr, sizeof(T), alignof(T));
  }

  NodeArena* arena() const { return arena_; }

  template <typename U>
  bool operator==(const ArenaAllocator<U>& other) const {
    return arena_ == other.arena();
  }

  template <typename U>
  bool operator!=(const ArenaAllocator<U>& other) const {
    return arena_ != other.arena();
  }

 private:
  NodeArena* arena_;
};

}

#endif
