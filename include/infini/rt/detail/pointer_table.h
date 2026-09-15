#ifndef INFINI_RT_DETAIL_POINTER_TABLE_H_
#define INFINI_RT_DETAIL_POINTER_TABLE_H_

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace infini::rt::detail {

template <typename Value>
class PointerTable {
 public:
  void Insert(void* key, const Value& value) {
    if ((occupied_ + 1) * 2 > slots_.size()) {
      Rehash();
    }

    const std::size_t mask = slots_.size() - 1;
    std::size_t index = Hash(key) & mask;
    std::size_t tombstone = kNoSlot;

    for (;; index = (index + 1) & mask) {
      Slot& slot = slots_[index];
      if (slot.state == State::kOccupied) {
        if (slot.key == key) {
          slot.value = value;
          return;
        }
        continue;
      }
      if (slot.state == State::kTombstone) {
        if (tombstone == kNoSlot) {
          tombstone = index;
        }
        continue;
      }
      break;
    }

    if (tombstone != kNoSlot) {
      index = tombstone;
      --tombstones_;
    } else {
      ++occupied_;
    }

    slots_[index] = Slot{key, value, State::kOccupied};
    ++live_;
  }

  bool Find(void* key, Value* out) const {
    if (live_ == 0) {
      return false;
    }

    const std::size_t mask = slots_.size() - 1;
    for (std::size_t index = Hash(key) & mask;; index = (index + 1) & mask) {
      const Slot& slot = slots_[index];
      if (slot.state == State::kEmpty) {
        return false;
      }
      if (slot.state == State::kOccupied && slot.key == key) {
        *out = slot.value;
        return true;
      }
    }
  }

  bool Take(void* key, Value* out) {
    if (live_ == 0) {
      return false;
    }

    const std::size_t mask = slots_.size() - 1;
    for (std::size_t index = Hash(key) & mask;; index = (index + 1) & mask) {
      Slot& slot = slots_[index];
      if (slot.state == State::kEmpty) {
        return false;
      }
      if (slot.state == State::kOccupied && slot.key == key) {
        *out = slot.value;
        slot.state = State::kTombstone;
        slot.key = nullptr;
        ++tombstones_;
        --live_;
        return true;
      }
    }
  }

  template <typename Visitor>
  void ForEach(Visitor&& visitor) const {
    for (const Slot& slot : slots_) {
      if (slot.state == State::kOccupied) {
        visitor(slot.value);
      }
    }
  }

  std::size_t Size() const { return live_; }

 private:
  enum class State : std::uint8_t { kEmpty, kOccupied, kTombstone };

  struct Slot {
    void* key = nullptr;
    Value value{};
    State state = State::kEmpty;
  };

  static constexpr std::size_t kInitialSlots = 16;
  static constexpr std::size_t kNoSlot = static_cast<std::size_t>(-1);

  static std::size_t Hash(void* key) {
    auto value =
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(key));
    value *= 0x9e3779b97f4a7c15ULL;
    return static_cast<std::size_t>(value >> 29);
  }

  void Rehash() {
    std::size_t capacity = kInitialSlots;
    while (capacity <= (live_ + 1) * 2) {
      capacity *= 2;
    }

    std::vector<Slot> old_slots(capacity);
    old_slots.swap(slots_);
    occupied_ = 0;
    tombstones_ = 0;
    live_ = 0;

    for (const Slot& slot : old_slots) {
      if (slot.state == State::kOccupied) {
        Insert(slot.key, slot.value);
      }
    }
  }

  std::vector<Slot> slots_;
  std::size_t occupied_ = 0;
  std::size_t tombstones_ = 0;
  std::size_t live_ = 0;
};

}

#endif
