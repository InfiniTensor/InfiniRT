#ifndef INFINI_RT_MEMORY_POOL_H_
#define INFINI_RT_MEMORY_POOL_H_

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace infini::rt {

/// ## Backend-agnostic caching allocator.
///
/// `cudaMalloc`/`cudaFree` (and other device allocators) are synchronous and
/// expensive, so runtimes typically layer a caching allocator on top. A
/// `MemoryPool` keeps freed blocks in per-size-class free lists and hands them
/// back on the next matching request, so hot allocation loops pay the upstream
/// allocator only on a cache miss.
///
/// The pool is a pure composition over an `Upstream` allocator: any type that
/// provides `Malloc(void**, size_t)`, `Free(void*)`, an `Error` type alias, and
/// a `static constexpr Error kSuccess` satisfies the contract. Every
/// `runtime::Runtime<...>` device specialization (CPU, NVIDIA, ...) qualifies,
/// and the same interface serves CPU aligned allocations. Tests can inject a
/// mock upstream to exercise the pool without any device.
///
/// Blocks are reused only when both the rounded size and the requested
/// alignment match, so a reused block is always geometrically identical to the
/// request; the pool never splits or coalesces, which keeps reuse free of
/// fragmentation hazards at the cost of some retained-but-unused memory (call
/// `ReleaseCached` to hand that back to the upstream allocator).
///
/// The pool is thread-safe: every public method takes an internal mutex. It is
/// neither copyable nor movable.
template <typename Upstream>
class MemoryPool {
 public:
  using Error = typename Upstream::Error;

  /// Runtime statistics. Byte counters are cumulative live totals; `peak_*`
  /// track high-water marks. The remaining counters are monotonic tallies.
  struct Stats {
    /// Bytes currently handed out to callers (sum of rounded block sizes).
    std::size_t bytes_in_use = 0;

    /// Bytes currently held from the upstream allocator (in use + cached).
    std::size_t bytes_reserved = 0;

    /// High-water mark of `bytes_in_use`.
    std::size_t peak_bytes_in_use = 0;

    /// High-water mark of `bytes_reserved`.
    std::size_t peak_bytes_reserved = 0;

    /// Number of `Allocate` calls that returned a non-null pointer.
    std::size_t alloc_count = 0;

    /// Number of `Deallocate` calls that released a live block.
    std::size_t free_count = 0;

    /// Allocations served from a cached free block.
    std::size_t cache_hit_count = 0;

    /// Allocations that required a fresh upstream allocation.
    std::size_t cache_miss_count = 0;

    /// Calls into `Upstream::Malloc`.
    std::size_t upstream_alloc_count = 0;

    /// Calls into `Upstream::Free`.
    std::size_t upstream_free_count = 0;
  };

  MemoryPool() = default;

  MemoryPool(const MemoryPool&) = delete;
  MemoryPool& operator=(const MemoryPool&) = delete;

  /// Frees every block still held from the upstream allocator, including
  /// blocks that were never handed back via `Deallocate`. Any outstanding
  /// pointer from `Allocate` dangles after destruction.
  ~MemoryPool() {
    for (auto& [key, blocks] : free_lists_) {
      for (const Block& block : blocks) {
        Upstream::Free(block.base);
      }
    }
    for (auto& [ptr, block] : allocated_) {
      Upstream::Free(block.base);
    }
  }

  /// Allocates at least `size` bytes, reusing a cached block when one with a
  /// matching size class and alignment is available. `alignment` of `0` uses
  /// the upstream allocator's natural alignment; otherwise the returned pointer
  /// is aligned up to `alignment` (which must be a power of two).
  ///
  /// On success writes the pointer to `*ptr` and returns `kSuccess`. A `size`
  /// of `0` succeeds with `*ptr == nullptr`. On upstream failure the upstream
  /// error is returned and `*ptr` is set to `nullptr`.
  Error Allocate(void** ptr, std::size_t size, std::size_t alignment = 0) {
    if (ptr == nullptr) {
      return InvalidValue();
    }

    *ptr = nullptr;
    if (size == 0) {
      return Upstream::kSuccess;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    const std::size_t rounded = RoundSize(size);
    const BucketKey key{rounded, alignment};

    Block block{};
    auto list_it = free_lists_.find(key);
    if (list_it != free_lists_.end() && !list_it->second.empty()) {
      block = list_it->second.back();
      list_it->second.pop_back();
      ++stats_.cache_hit_count;
    } else {
      const std::size_t upstream_size =
          alignment == 0 ? rounded : rounded + alignment;
      void* base = nullptr;
      const Error status = Upstream::Malloc(&base, upstream_size);
      ++stats_.upstream_alloc_count;
      if (status != Upstream::kSuccess) {
        return status;
      }

      block.base = base;
      block.aligned = alignment == 0 ? base : AlignUp(base, alignment);
      block.rounded_size = rounded;
      block.upstream_size = upstream_size;
      block.alignment = alignment;

      stats_.bytes_reserved += upstream_size;
      if (stats_.bytes_reserved > stats_.peak_bytes_reserved) {
        stats_.peak_bytes_reserved = stats_.bytes_reserved;
      }
      ++stats_.cache_miss_count;
    }

    allocated_.emplace(block.aligned, block);
    stats_.bytes_in_use += block.rounded_size;
    if (stats_.bytes_in_use > stats_.peak_bytes_in_use) {
      stats_.peak_bytes_in_use = stats_.bytes_in_use;
    }
    ++stats_.alloc_count;

    *ptr = block.aligned;
    return Upstream::kSuccess;
  }

  /// Returns a block from `Allocate` to the pool's free list for reuse. The
  /// block is not handed back to the upstream allocator until `ReleaseCached`
  /// or destruction. `nullptr` is a no-op. Returns an invalid-value error if
  /// `ptr` was not produced by this pool (or was already freed).
  Error Deallocate(void* ptr) {
    if (ptr == nullptr) {
      return Upstream::kSuccess;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = allocated_.find(ptr);
    if (it == allocated_.end()) {
      return InvalidValue();
    }

    const Block block = it->second;
    allocated_.erase(it);

    stats_.bytes_in_use -= block.rounded_size;
    ++stats_.free_count;

    free_lists_[BucketKey{block.rounded_size, block.alignment}].push_back(
        block);
    return Upstream::kSuccess;
  }

  /// Hands every cached (freed but not-yet-returned) block back to the upstream
  /// allocator. Blocks currently in use are untouched. This is the pool's
  /// defragmentation / trim knob: call it to release retained memory back to
  /// the device.
  void ReleaseCached() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& [key, blocks] : free_lists_) {
      for (const Block& block : blocks) {
        Upstream::Free(block.base);
        ++stats_.upstream_free_count;
        stats_.bytes_reserved -= block.upstream_size;
      }
    }
    free_lists_.clear();
  }

  /// Returns a snapshot of the pool's statistics.
  Stats GetStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
  }

 private:
  static_assert(
      std::is_invocable_v<decltype(Upstream::Malloc), void**, std::size_t>,
      "`Upstream::Malloc` must be callable with `(void**, size_t)`.");
  static_assert(std::is_invocable_v<decltype(Upstream::Free), void*>,
                "`Upstream::Free` must be callable with `(void*)`.");
  static_assert(
      std::is_same_v<std::remove_cv_t<decltype(Upstream::kSuccess)>, Error>,
      "`Upstream` must define `static constexpr Error kSuccess`.");

  // A single upstream allocation tracked by the pool. `base` is the pointer
  // owned by the upstream allocator; `aligned` is what the caller sees.
  struct Block {
    void* base = nullptr;
    void* aligned = nullptr;
    std::size_t rounded_size = 0;
    std::size_t upstream_size = 0;
    std::size_t alignment = 0;
  };

  // Free lists are keyed by rounded size and alignment so a reused block is
  // always geometrically identical to the request.
  struct BucketKey {
    std::size_t size = 0;
    std::size_t alignment = 0;

    bool operator==(const BucketKey& other) const {
      return size == other.size && alignment == other.alignment;
    }
  };

  struct BucketKeyHash {
    std::size_t operator()(const BucketKey& key) const {
      // Mix the two fields; alignment is small so a shift keeps it out of the
      // low bits that size dominates.
      return key.size ^ (key.alignment << 1);
    }
  };

  // Small allocations round to 512 B; large ones to 2 MB. This keeps the number
  // of distinct size classes bounded so freed blocks are likely to be reused.
  static constexpr std::size_t kSmallThreshold = 1u << 20;  // 1 MB
  static constexpr std::size_t kSmallGranularity = 512;
  static constexpr std::size_t kLargeGranularity = 1u << 21;  // 2 MB

  static std::size_t RoundUp(std::size_t size, std::size_t granularity) {
    return (size + granularity - 1) / granularity * granularity;
  }

  static std::size_t RoundSize(std::size_t size) {
    return size <= kSmallThreshold ? RoundUp(size, kSmallGranularity)
                                   : RoundUp(size, kLargeGranularity);
  }

  static void* AlignUp(void* ptr, std::size_t alignment) {
    const auto address = reinterpret_cast<std::uintptr_t>(ptr);
    const auto aligned = (address + alignment - 1) & ~(alignment - 1);
    return reinterpret_cast<void*>(aligned);
  }

  static Error InvalidValue() { return static_cast<Error>(1); }

  mutable std::mutex mutex_;
  std::unordered_map<void*, Block> allocated_;
  std::unordered_map<BucketKey, std::vector<Block>, BucketKeyHash> free_lists_;
  Stats stats_;
};

}  // namespace infini::rt

#endif
