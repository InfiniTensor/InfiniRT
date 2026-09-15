#ifndef INFINI_RT_ARENA_MEMORY_POOL_H_
#define INFINI_RT_ARENA_MEMORY_POOL_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <type_traits>
#include <utility>
#include <vector>

#include "infini/rt/detail/node_arena.h"
#include "infini/rt/detail/pointer_table.h"

namespace infini::rt {

struct DefaultArenaConfig {
  static constexpr std::size_t kInitialCapacity = 64ull << 20;

  static constexpr std::size_t kMaxCapacity = 512ull << 20;

  static constexpr std::size_t kSmallThreshold = 1ull << 20;

  static constexpr std::size_t kMinSliceAlignment = 512;

  static constexpr std::size_t kMinSplitRemainder = 512;

  static constexpr std::size_t kShrinkThreshold = 16;

  static constexpr std::uint32_t kEmptyScansToDestroy = 2;

  static constexpr std::size_t kThreadCacheDepth = 1;

  static constexpr std::size_t kThreadCacheBytes = 8ull << 20;
};

namespace detail {

template <typename Config, typename = void>
struct ArenaCacheDepth
    : std::integral_constant<std::size_t,
                             DefaultArenaConfig::kThreadCacheDepth> {};

template <typename Config>
struct ArenaCacheDepth<Config, std::void_t<decltype(Config::kThreadCacheDepth)>>
    : std::integral_constant<std::size_t, Config::kThreadCacheDepth> {};

template <typename Config, typename = void>
struct ArenaCacheBytes
    : std::integral_constant<std::size_t, Config::kInitialCapacity / 8> {};

template <typename Config>
struct ArenaCacheBytes<Config, std::void_t<decltype(Config::kThreadCacheBytes)>>
    : std::integral_constant<std::size_t, Config::kThreadCacheBytes> {};

}

template <typename Upstream, typename Config = DefaultArenaConfig>
class ArenaMemoryPool {
 public:
  using Error = typename Upstream::Error;

  struct Stats {
    std::size_t bytes_in_use = 0;

    std::size_t bytes_reserved = 0;

    std::size_t peak_bytes_in_use = 0;

    std::size_t peak_bytes_reserved = 0;

    std::size_t alloc_count = 0;

    std::size_t free_count = 0;

    std::size_t cache_hit_count = 0;

    std::size_t cache_miss_count = 0;

    std::size_t upstream_alloc_count = 0;

    std::size_t upstream_free_count = 0;

    std::size_t backing_count = 0;

    std::size_t bytes_free_in_backings = 0;

    std::size_t largest_free_chunk = 0;

    std::size_t bytes_internal_waste = 0;

    std::size_t bytes_unusable = 0;

    std::size_t shrink_count = 0;
  };

  ArenaMemoryPool()
      : registry_(std::make_shared<Registry>()), mutex_(registry_->mutex) {
    registry_->pool = this;
  }

  ArenaMemoryPool(const ArenaMemoryPool&) = delete;
  ArenaMemoryPool& operator=(const ArenaMemoryPool&) = delete;

  ~ArenaMemoryPool() {
    {
      std::lock_guard<std::mutex> lock(registry_->mutex);
      for (ThreadCache* cache = registry_->head; cache != nullptr;
           cache = cache->next) {
        ReclaimCacheLocked(cache);
      }
      registry_->pool = nullptr;
    }

    for (const auto& backing : backings_) {
      DestroyChunks(backing.get());
      Upstream::Free(backing->base);
    }
  }

  Error Allocate(void** ptr, std::size_t size, std::size_t alignment = 0) {
    if (ptr == nullptr) {
      return InvalidValue();
    }

    *ptr = nullptr;
    if (size == 0) {
      return Upstream::kSuccess;
    }

    const std::size_t align = SliceAlignment(alignment);
    const std::size_t rounded = RoundSize(size);

    if (align == Config::kMinSliceAlignment) {
      if (void* cached = TryCacheAllocate(rounded); cached != nullptr) {
        *ptr = cached;
        return Upstream::kSuccess;
      }
    }

    const std::size_t needed = rounded + align - Config::kMinSliceAlignment;

    std::unique_lock<std::mutex> lock(mutex_);

    {
      std::vector<std::unique_ptr<BackingStore>> doomed;
      UpdateShrinkState(size, &doomed);
      if (!doomed.empty()) {
        lock.unlock();
        FreeBackings(doomed);
        doomed.clear();
        lock.lock();
      }
    }

    if (Chunk* chunk = FindFit(needed); chunk != nullptr) {
      ++stats_.cache_hit_count;
      *ptr = Serve(chunk, rounded, align);
      return Upstream::kSuccess;
    }

    std::size_t capacity = NextCapacity(needed);
    lock.unlock();

    void* base = nullptr;
    Error status = Upstream::Malloc(&base, capacity);
    if (status != Upstream::kSuccess) {
      status = AllocateFallback(&base, &capacity, needed, status);
    }

    lock.lock();

    if (status != Upstream::kSuccess) {
      if (Chunk* chunk = FindFit(needed); chunk != nullptr) {
        ++stats_.cache_hit_count;
        *ptr = Serve(chunk, rounded, align);
        return Upstream::kSuccess;
      }
      return status;
    }

    ++stats_.upstream_alloc_count;
    Chunk* chunk = AdoptBacking(base, capacity);
    ++stats_.cache_miss_count;
    *ptr = Serve(chunk, rounded, align);
    return Upstream::kSuccess;
  }

  Error Deallocate(void* ptr) {
    if (ptr == nullptr) {
      return Upstream::kSuccess;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    Chunk* chunk = nullptr;
    if (!allocated_.Find(ptr, &chunk) || chunk->cached) {
      return InvalidValue();
    }

    ++stats_.free_count;

    const std::size_t bin = BinIndex(chunk->size);
    if (bin != kNoBin) {
      if (ThreadCache* cache = LocalCacheIfPresent();
          cache != nullptr && TryParkCached(cache, bin, chunk)) {
        return Upstream::kSuccess;
      }
    }

    allocated_.Take(ptr, &chunk);
    ReleaseChunk(chunk);
    return Upstream::kSuccess;
  }

  void ReleaseCached() {
    std::vector<std::unique_ptr<BackingStore>> doomed;
    {
      std::lock_guard<std::mutex> lock(mutex_);

      ReclaimAllCaches();
      for (std::size_t i = backings_.size(); i-- > 0;) {
        if (IsDrained(backings_[i].get())) {
          doomed.push_back(Detach(i));
        }
      }
      if (backings_.empty()) {
        next_capacity_ = Config::kInitialCapacity;
      }
    }

    FreeBackings(doomed);
  }

  Stats GetStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Stats stats = stats_;
    stats.backing_count = backings_.size();
    stats.largest_free_chunk = LargestFreeChunk();

    std::size_t cached_bytes = 0;
    SumCaches(&cached_bytes, &stats.alloc_count, &stats.cache_hit_count);

    stats.bytes_in_use -= cached_bytes;
    stats.bytes_free_in_backings += cached_bytes;
    return stats;
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
  static_assert(
      Config::kMinSliceAlignment != 0 &&
          (Config::kMinSliceAlignment & (Config::kMinSliceAlignment - 1)) == 0,
      "`Config::kMinSliceAlignment` must be a power of two.");
  static_assert(Config::kInitialCapacity <= Config::kMaxCapacity,
                "`Config::kInitialCapacity` must not exceed "
                "`Config::kMaxCapacity`.");

  struct BackingStore;

  enum class Location : std::uint8_t { kNone, kFastBin, kTree };

  struct Chunk {
    BackingStore* owner = nullptr;
    Chunk* prev = nullptr;
    Chunk* next = nullptr;
    void* ptr = nullptr;
    std::size_t size = 0;

    std::size_t requested = 0;
    bool is_free = true;

    bool cached = false;

    Location location = Location::kNone;

    Chunk* fast_prev = nullptr;
    Chunk* fast_next = nullptr;
  };

  struct BackingStore {
    void* base = nullptr;
    std::size_t capacity = 0;
    Chunk* head = nullptr;

    std::size_t live_chunks = 0;

    bool resident = false;

    bool oversize = false;
    std::uint32_t empty_scans = 0;
  };

  struct BySizeThenAddress {
    bool operator()(const Chunk* lhs, const Chunk* rhs) const {
      if (lhs->size != rhs->size) {
        return lhs->size < rhs->size;
      }
      return reinterpret_cast<std::uintptr_t>(lhs->ptr) <
             reinterpret_cast<std::uintptr_t>(rhs->ptr);
    }
  };

  using FreeSet =
      std::set<Chunk*, BySizeThenAddress, detail::ArenaAllocator<Chunk*>>;

  static constexpr std::size_t kFastBinCount = 128;
  static constexpr std::size_t kFastBinWords = (kFastBinCount + 63) / 64;

  static constexpr std::size_t kNoBin = static_cast<std::size_t>(-1);

  static Error InvalidValue() { return static_cast<Error>(1); }

  static std::size_t CountTrailingZeros(std::uint64_t word) {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<std::size_t>(__builtin_ctzll(word));
#else
    std::size_t count = 0;
    while ((word & 1ull) == 0) {
      word >>= 1;
      ++count;
    }
    return count;
#endif
  }

  static std::size_t BinIndex(std::size_t size) {
    if (size == 0 ||
        (size & (Config::kMinSliceAlignment - 1)) != 0) {
      return kNoBin;
    }
    const std::size_t multiples = size / Config::kMinSliceAlignment;
    return multiples <= kFastBinCount ? multiples - 1 : kNoBin;
  }

  static std::size_t FirstEligibleBin(std::size_t needed) {
    const std::size_t multiples =
        (needed + Config::kMinSliceAlignment - 1) / Config::kMinSliceAlignment;
    if (multiples == 0) {
      return 0;
    }
    return multiples <= kFastBinCount ? multiples - 1 : kNoBin;
  }

  static constexpr std::size_t kCacheDepth =
      detail::ArenaCacheDepth<Config>::value;
  static constexpr std::size_t kCacheBytes =
      detail::ArenaCacheBytes<Config>::value;

  struct ThreadCache {
    Chunk* lists[kFastBinCount] = {};
    std::size_t depths[kFastBinCount] = {};
    std::size_t bytes = 0;

    std::size_t alloc_count = 0;
    std::size_t cache_hit_count = 0;

    std::atomic<bool> busy{false};

    ThreadCache* next = nullptr;
  };

  struct Registry {
    std::mutex mutex;
    ArenaMemoryPool* pool = nullptr;
    ThreadCache* head = nullptr;
  };

  class ThreadCacheHandle {
   public:
    explicit ThreadCacheHandle(std::shared_ptr<Registry> registry)
        : registry_(std::move(registry)), cache_(new ThreadCache) {
      std::lock_guard<std::mutex> lock(registry_->mutex);
      cache_->next = registry_->head;
      registry_->head = cache_;
    }

    ~ThreadCacheHandle() {
      {
        std::lock_guard<std::mutex> lock(registry_->mutex);
        if (registry_->pool != nullptr) {
          registry_->pool->RetireCacheLocked(cache_);
        }
        Unlink(cache_);
      }
      delete cache_;
    }

    ThreadCacheHandle(const ThreadCacheHandle&) = delete;
    ThreadCacheHandle& operator=(const ThreadCacheHandle&) = delete;

    ThreadCache* get() const { return cache_; }

   private:
    void Unlink(ThreadCache* cache) {
      ThreadCache** link = &registry_->head;
      while (*link != nullptr && *link != cache) {
        link = &(*link)->next;
      }
      if (*link != nullptr) {
        *link = cache->next;
      }
    }

    std::shared_ptr<Registry> registry_;
    ThreadCache* cache_;
  };

  class CacheGuard {
   public:
    explicit CacheGuard(ThreadCache* cache) : cache_(cache) {
      while (cache_->busy.exchange(true, std::memory_order_acquire)) {
      }
    }
    ~CacheGuard() { cache_->busy.store(false, std::memory_order_release); }

    CacheGuard(const CacheGuard&) = delete;
    CacheGuard& operator=(const CacheGuard&) = delete;

   private:
    ThreadCache* cache_;
  };

  struct CacheEntry {
    const Registry* key;
    std::unique_ptr<ThreadCacheHandle> handle;
  };

  static std::vector<CacheEntry>& CacheMap() {
    static thread_local std::vector<CacheEntry> caches;
    return caches;
  }

  ThreadCache* LocalCache() {
    if (ThreadCache* cache = LocalCacheIfPresent(); cache != nullptr) {
      return cache;
    }

    CacheMap().push_back(
        CacheEntry{registry_.get(),
                   std::make_unique<ThreadCacheHandle>(registry_)});
    return CacheMap().back().handle->get();
  }

  ThreadCache* LocalCacheIfPresent() {
    for (const CacheEntry& entry : CacheMap()) {
      if (entry.key == registry_.get()) {
        return entry.handle->get();
      }
    }
    return nullptr;
  }

  static std::size_t RoundUp(std::size_t size, std::size_t granularity) {
    return (size + granularity - 1) / granularity * granularity;
  }

  static std::size_t RoundSize(std::size_t size) {
    return RoundUp(size, Config::kMinSliceAlignment);
  }

  static std::size_t SliceAlignment(std::size_t alignment) {
    return alignment > Config::kMinSliceAlignment ? alignment
                                                  : Config::kMinSliceAlignment;
  }

  static void* Offset(void* ptr, std::size_t bytes) {
    return static_cast<char*>(ptr) + bytes;
  }

  static std::size_t AlignPadding(void* ptr, std::size_t alignment) {
    const auto address = reinterpret_cast<std::uintptr_t>(ptr);
    const auto aligned = (address + alignment - 1) &
                         ~static_cast<std::uintptr_t>(alignment - 1);
    return static_cast<std::size_t>(aligned - address);
  }

  static bool IsDrained(const BackingStore* backing) {
    return backing->live_chunks == 0;
  }

  Chunk* NewChunk() {
    void* storage = chunk_arena_.Allocate(sizeof(Chunk), alignof(Chunk));
    return new (storage) Chunk();
  }

  void DeleteChunk(Chunk* chunk) {
    chunk->~Chunk();
    chunk_arena_.Deallocate(chunk, sizeof(Chunk), alignof(Chunk));
  }

  void DestroyChunks(BackingStore* backing) {
    Chunk* chunk = backing->head;
    while (chunk != nullptr) {
      Chunk* next = chunk->next;
      DeleteChunk(chunk);
      chunk = next;
    }
    backing->head = nullptr;
  }

  void InsertFree(Chunk* chunk) {
    const std::size_t bin = BinIndex(chunk->size);
    if (bin == kNoBin) {
      chunk->location = Location::kTree;
      free_chunks_.insert(chunk);
      return;
    }

    chunk->location = Location::kFastBin;
    chunk->fast_prev = nullptr;
    chunk->fast_next = fast_bins_[bin];
    if (chunk->fast_next != nullptr) {
      chunk->fast_next->fast_prev = chunk;
    }
    fast_bins_[bin] = chunk;
    fast_bitmap_[bin / 64] |= 1ull << (bin % 64);
  }

  void EraseFree(Chunk* chunk) {
    if (chunk->location == Location::kNone) {
      return;
    }
    if (chunk->location == Location::kTree) {
      chunk->location = Location::kNone;
      free_chunks_.erase(chunk);
      return;
    }

    const std::size_t bin = BinIndex(chunk->size);
    if (chunk->fast_prev != nullptr) {
      chunk->fast_prev->fast_next = chunk->fast_next;
    } else {
      fast_bins_[bin] = chunk->fast_next;
    }
    if (chunk->fast_next != nullptr) {
      chunk->fast_next->fast_prev = chunk->fast_prev;
    }
    chunk->fast_prev = nullptr;
    chunk->fast_next = nullptr;
    chunk->location = Location::kNone;

    if (fast_bins_[bin] == nullptr) {
      fast_bitmap_[bin / 64] &= ~(1ull << (bin % 64));
    }
  }

  void* TryCacheAllocate(std::size_t rounded) {
    const std::size_t bin = BinIndex(rounded);
    if (bin == kNoBin) {
      return nullptr;
    }

    ThreadCache* cache = LocalCache();
    CacheGuard guard(cache);
    Chunk* chunk = cache->lists[bin];
    if (chunk == nullptr) {
      return nullptr;
    }

    cache->lists[bin] = chunk->fast_next;
    --cache->depths[bin];
    cache->bytes -= chunk->size;
    ++cache->alloc_count;
    ++cache->cache_hit_count;

    chunk->fast_next = nullptr;
    chunk->cached = false;

    chunk->requested = rounded;
    return chunk->ptr;
  }

  bool TryParkCached(ThreadCache* cache, std::size_t bin, Chunk* chunk) {
    CacheGuard guard(cache);
    if (cache->depths[bin] >= kCacheDepth ||
        cache->bytes + chunk->size > kCacheBytes) {
      return false;
    }

    stats_.bytes_internal_waste -= chunk->size - chunk->requested;
    chunk->requested = chunk->size;
    chunk->cached = true;

    chunk->fast_next = cache->lists[bin];
    cache->lists[bin] = chunk;
    ++cache->depths[bin];
    cache->bytes += chunk->size;
    return true;
  }

  void ReclaimCacheLocked(ThreadCache* cache) {
    CacheGuard guard(cache);
    if (cache->bytes == 0) {
      return;
    }

    for (std::size_t bin = 0; bin < kFastBinCount; ++bin) {
      Chunk* chunk = cache->lists[bin];
      cache->lists[bin] = nullptr;
      cache->depths[bin] = 0;
      while (chunk != nullptr) {
        Chunk* next = chunk->fast_next;
        chunk->fast_next = nullptr;
        chunk->cached = false;
        ReleaseChunk(chunk);
        chunk = next;
      }
    }
    cache->bytes = 0;
  }

  void RetireCacheLocked(ThreadCache* cache) {
    ReclaimCacheLocked(cache);
    stats_.alloc_count += cache->alloc_count;
    stats_.cache_hit_count += cache->cache_hit_count;
    cache->alloc_count = 0;
    cache->cache_hit_count = 0;
  }

  void ReclaimAllCaches() {
    for (ThreadCache* cache = registry_->head; cache != nullptr;
         cache = cache->next) {
      ReclaimCacheLocked(cache);
    }
  }

  void SumCaches(std::size_t* bytes, std::size_t* allocs,
                 std::size_t* hits) const {
    for (ThreadCache* cache = registry_->head; cache != nullptr;
         cache = cache->next) {
      CacheGuard guard(cache);
      *bytes += cache->bytes;
      *allocs += cache->alloc_count;
      *hits += cache->cache_hit_count;
    }
  }

  void ReleaseChunk(Chunk* chunk) {
    stats_.bytes_in_use -= chunk->size;
    stats_.bytes_internal_waste -= chunk->size - chunk->requested;

    chunk->is_free = true;
    chunk->requested = 0;
    stats_.bytes_free_in_backings += chunk->size;

    BackingStore* owner = chunk->owner;

    coalesce_dirty_ = true;
    InsertFree(chunk);

    if (--owner->live_chunks == 0 && !owner->resident) {
      CoalesceBacking(owner);
      ++drained_candidates_;
    }
  }

  Chunk* ScanBins(std::size_t from) const {
    if (from >= kFastBinCount) {
      return nullptr;
    }
    std::size_t word = from / 64;
    std::uint64_t bits = fast_bitmap_[word] & (~0ull << (from % 64));
    while (bits == 0) {
      if (++word >= kFastBinWords) {
        return nullptr;
      }
      bits = fast_bitmap_[word];
    }
    return fast_bins_[word * 64 + CountTrailingZeros(bits)];
  }

  Chunk* FindFit(std::size_t needed) {
    if (Chunk* chunk = FindFitIndexed(needed); chunk != nullptr) {
      return chunk;
    }
    if (coalesce_dirty_) {
      CoalesceAll();
      if (Chunk* chunk = FindFitIndexed(needed); chunk != nullptr) {
        return chunk;
      }
    }

    if (!AnyCached()) {
      return nullptr;
    }
    ReclaimAllCaches();
    if (Chunk* chunk = FindFitIndexed(needed); chunk != nullptr) {
      return chunk;
    }

    if (!coalesce_dirty_) {
      return nullptr;
    }
    CoalesceAll();
    return FindFitIndexed(needed);
  }

  bool AnyCached() const {
    for (ThreadCache* cache = registry_->head; cache != nullptr;
         cache = cache->next) {
      CacheGuard guard(cache);
      if (cache->bytes != 0) {
        return true;
      }
    }
    return false;
  }

  Chunk* FindFitIndexed(std::size_t needed) {
    const std::size_t first = FirstEligibleBin(needed);

    if (first != kNoBin && fast_bins_[first] != nullptr) {
      return fast_bins_[first];
    }

    Chunk* binned = first == kNoBin ? nullptr : ScanBins(first + 1);

    Chunk probe{};
    probe.size = needed;
    probe.ptr = nullptr;
    auto it = free_chunks_.lower_bound(&probe);
    Chunk* treed = it == free_chunks_.end() ? nullptr : *it;

    if (binned == nullptr) {
      return treed;
    }
    if (treed == nullptr) {
      return binned;
    }
    return treed->size < binned->size ? treed : binned;
  }

  std::size_t LargestFreeChunk() const {
    std::size_t largest = 0;
    for (const auto& backing : backings_) {
      std::size_t run = 0;
      for (const Chunk* chunk = backing->head; chunk != nullptr;
           chunk = chunk->next) {
        if (chunk->is_free || chunk->cached) {
          run += chunk->size;
          if (run > largest) {
            largest = run;
          }
        } else {
          run = 0;
        }
      }
    }
    return largest;
  }

  void CoalesceBacking(BackingStore* backing) {
    for (Chunk* chunk = backing->head; chunk != nullptr;) {
      if (!chunk->is_free) {
        chunk = chunk->next;
        continue;
      }

      Chunk* next = chunk->next;
      if (next == nullptr || !next->is_free) {
        chunk = next;
        continue;
      }
      EraseFree(chunk);
      while (next != nullptr && next->is_free) {
        EraseFree(next);
        chunk->size += next->size;
        Chunk* after = next->next;
        DeleteChunk(next);
        next = after;
      }
      chunk->next = next;
      if (next != nullptr) {
        next->prev = chunk;
      }
      InsertFree(chunk);
      chunk = next;
    }
  }

  void CoalesceAll() {
    for (const auto& backing : backings_) {
      CoalesceBacking(backing.get());
    }
    coalesce_dirty_ = false;
  }

  Chunk* SplitAt(Chunk* chunk, std::size_t offset) {
    Chunk* tail = NewChunk();
    tail->owner = chunk->owner;
    tail->ptr = Offset(chunk->ptr, offset);
    tail->size = chunk->size - offset;
    tail->is_free = chunk->is_free;
    tail->prev = chunk;
    tail->next = chunk->next;
    if (chunk->next != nullptr) {
      chunk->next->prev = tail;
    }
    chunk->next = tail;
    chunk->size = offset;
    return tail;
  }

  void* Serve(Chunk* chunk, std::size_t rounded, std::size_t alignment) {
    EraseFree(chunk);
    BackingStore* owner = chunk->owner;
    if (owner->live_chunks++ == 0) {
      DropCandidate(owner);
    }
    stats_.bytes_free_in_backings -= chunk->size;

    chunk->is_free = false;

    const std::size_t padding = AlignPadding(chunk->ptr, alignment);
    if (padding != 0) {
      Chunk* body = SplitAt(chunk, padding);
      chunk->is_free = true;
      stats_.bytes_free_in_backings += chunk->size;
      InsertFree(chunk);
      chunk = body;
    }

    if (chunk->size - rounded >= Config::kMinSplitRemainder) {
      Chunk* tail = SplitAt(chunk, rounded);
      tail->is_free = true;
      stats_.bytes_free_in_backings += tail->size;
      InsertFree(tail);
    }

    chunk->requested = rounded;
    allocated_.Insert(chunk->ptr, chunk);

    stats_.bytes_in_use += chunk->size;
    stats_.bytes_internal_waste += chunk->size - chunk->requested;
    if (stats_.bytes_in_use > stats_.peak_bytes_in_use) {
      stats_.peak_bytes_in_use = stats_.bytes_in_use;
    }
    ++stats_.alloc_count;
    return chunk->ptr;
  }

  std::size_t NextCapacity(std::size_t needed) const {
    const std::size_t floor = needed + Config::kMinSliceAlignment;
    return next_capacity_ > floor ? next_capacity_ : floor;
  }

  Chunk* AdoptBacking(void* base, std::size_t capacity) {
    auto backing = std::make_unique<BackingStore>();
    backing->base = base;
    backing->capacity = capacity;
    backing->oversize = capacity > Config::kMaxCapacity;

    backing->resident = !backing->oversize && !HasResident();

    const std::size_t lead = AlignPadding(base, Config::kMinSliceAlignment);

    Chunk* chunk = NewChunk();
    chunk->owner = backing.get();
    chunk->ptr = Offset(base, lead);
    chunk->size = capacity - lead;
    chunk->is_free = true;
    backing->head = chunk;

    stats_.bytes_reserved += capacity;
    if (stats_.bytes_reserved > stats_.peak_bytes_reserved) {
      stats_.peak_bytes_reserved = stats_.bytes_reserved;
    }
    stats_.bytes_free_in_backings += chunk->size;
    stats_.bytes_unusable += lead;
    InsertFree(chunk);

    if (!backing->resident) {
      ++drained_candidates_;
    }

    if (!backing->oversize) {
      next_capacity_ = capacity >= Config::kMaxCapacity / 2
                           ? Config::kMaxCapacity
                           : capacity * 2;
    }

    backings_.push_back(std::move(backing));
    return chunk;
  }

  void DropCandidate(const BackingStore* backing) {
    if (!backing->resident) {
      --drained_candidates_;
    }
  }

  bool HasResident() const {
    for (const auto& backing : backings_) {
      if (backing->resident) {
        return true;
      }
    }
    return false;
  }

  std::unique_ptr<BackingStore> Detach(std::size_t index) {
    std::unique_ptr<BackingStore> backing = std::move(backings_[index]);
    backings_.erase(backings_.begin() + static_cast<std::ptrdiff_t>(index));

    CoalesceBacking(backing.get());
    DropCandidate(backing.get());

    EraseFree(backing->head);
    stats_.bytes_free_in_backings -= backing->head->size;
    stats_.bytes_unusable -= backing->capacity - backing->head->size;
    stats_.bytes_reserved -= backing->capacity;
    ++stats_.upstream_free_count;
    DestroyChunks(backing.get());
    return backing;
  }

  static void FreeBackings(
      const std::vector<std::unique_ptr<BackingStore>>& doomed) {
    for (const auto& backing : doomed) {
      Upstream::Free(backing->base);
    }
  }

  void UpdateShrinkState(std::size_t size,
                         std::vector<std::unique_ptr<BackingStore>>* doomed) {
    if (drained_candidates_ == 0) {
      return;
    }

    if (size > Config::kSmallThreshold) {
      small_alloc_since_last_trim_ = 0;
      return;
    }

    if (++small_alloc_since_last_trim_ < Config::kShrinkThreshold) {
      return;
    }

    small_alloc_since_last_trim_ = 0;

    ReclaimAllCaches();

    for (std::size_t i = backings_.size(); i-- > 0;) {
      BackingStore* backing = backings_[i].get();
      if (backing->resident) {
        continue;
      }
      if (!IsDrained(backing)) {
        backing->empty_scans = 0;
        continue;
      }
      ++backing->empty_scans;

      if (!backing->oversize &&
          backing->empty_scans < Config::kEmptyScansToDestroy) {
        continue;
      }
      doomed->push_back(Detach(i));
      ++stats_.shrink_count;
    }
  }

  Error AllocateFallback(void** base, std::size_t* capacity,
                         std::size_t needed, Error failure) {
    std::vector<std::unique_ptr<BackingStore>> doomed;
    {
      std::lock_guard<std::mutex> lock(mutex_);

      ReclaimAllCaches();
      for (std::size_t i = backings_.size(); i-- > 0;) {
        if (IsDrained(backings_[i].get())) {
          doomed.push_back(Detach(i));
        }
      }
    }

    if (!doomed.empty()) {
      FreeBackings(doomed);
      doomed.clear();
      if (Upstream::Malloc(base, *capacity) == Upstream::kSuccess) {
        return Upstream::kSuccess;
      }
    }

    const std::size_t minimum = needed + Config::kMinSliceAlignment;
    if (*capacity > minimum &&
        Upstream::Malloc(base, minimum) == Upstream::kSuccess) {
      *capacity = minimum;
      return Upstream::kSuccess;
    }
    return failure;
  }

  std::shared_ptr<Registry> registry_;

  std::mutex& mutex_;

  detail::NodeArena chunk_arena_;
  detail::NodeArena free_set_arena_;

  detail::PointerTable<Chunk*> allocated_;
  FreeSet free_chunks_{BySizeThenAddress{},
                       detail::ArenaAllocator<Chunk*>{&free_set_arena_}};

  Chunk* fast_bins_[kFastBinCount] = {};
  std::uint64_t fast_bitmap_[kFastBinWords] = {};

  std::vector<std::unique_ptr<BackingStore>> backings_;
  std::size_t next_capacity_ = Config::kInitialCapacity;
  std::size_t small_alloc_since_last_trim_ = 0;

  std::size_t drained_candidates_ = 0;

  bool coalesce_dirty_ = false;
  Stats stats_;
};

}

#endif
