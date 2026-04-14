#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "common/cuda_compat.hpp"
#include "genetic_algorithm/genotype_arena/layout.hpp"

namespace neuroevolution::genetic_algorithm::genotype_arena {

struct ArenaSlotState {
    bool occupied = false;
    std::uint32_t reference_count = 0;
};

struct AlignedArenaStorageDeleter {
    void operator()(std::uint8_t *pointer) const noexcept;
};

using AlignedArenaStorage = std::unique_ptr<std::uint8_t[], AlignedArenaStorageDeleter>;

struct HostGenotypeArena {
    GenotypeArenaLayout layout{};
    AlignedArenaStorage storage{};
    std::unique_ptr<ArenaSlotState[]> slot_states{};
    std::unique_ptr<std::uint32_t[]> free_slot_stack{};
    std::size_t free_slot_count = 0;
};

struct GenotypeArenaView {
    GenotypeArenaLayout layout{};
    std::uint8_t *storage = nullptr;
    ArenaSlotState *slot_states = nullptr;
    std::uint32_t *free_slot_stack = nullptr;
    std::size_t *free_slot_count = nullptr;
};

struct ConstGenotypeArenaView {
    GenotypeArenaLayout layout{};
    const std::uint8_t *storage = nullptr;
    const ArenaSlotState *slot_states = nullptr;
    const std::uint32_t *free_slot_stack = nullptr;
    const std::size_t *free_slot_count = nullptr;
};

constexpr std::uint32_t kMaxArenaSlotReferenceCount = static_cast<std::uint32_t>(-1);

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidArenaSlotState(const ArenaSlotState &slot_state) noexcept {
    return slot_state.occupied ? (slot_state.reference_count > 0) : (slot_state.reference_count == 0);
}

inline GenotypeArenaView MakeGenotypeArenaView(HostGenotypeArena &arena) noexcept {
    return {
        .layout = arena.layout,
        .storage = arena.storage.get(),
        .slot_states = arena.slot_states.get(),
        .free_slot_stack = arena.free_slot_stack.get(),
        .free_slot_count = &arena.free_slot_count,
    };
}

inline ConstGenotypeArenaView MakeConstGenotypeArenaView(const HostGenotypeArena &arena) noexcept {
    return {
        .layout = arena.layout,
        .storage = arena.storage.get(),
        .slot_states = arena.slot_states.get(),
        .free_slot_stack = arena.free_slot_stack.get(),
        .free_slot_count = &arena.free_slot_count,
    };
}

inline NEUROEVOLUTION_HOST_DEVICE std::size_t ArenaFreeSlotCount(const GenotypeArenaView &arena) noexcept {
    return (arena.free_slot_count == nullptr) ? 0 : *arena.free_slot_count;
}

inline NEUROEVOLUTION_HOST_DEVICE std::size_t ArenaFreeSlotCount(const ConstGenotypeArenaView &arena) noexcept {
    return (arena.free_slot_count == nullptr) ? 0 : *arena.free_slot_count;
}

inline NEUROEVOLUTION_HOST_DEVICE bool IsValidGenotypeArenaView(const GenotypeArenaView &arena) noexcept {
    if (!IsValidGenotypeArenaLayout(arena.layout) || (arena.storage == nullptr) || (arena.slot_states == nullptr) ||
        (arena.free_slot_stack == nullptr) || (arena.free_slot_count == nullptr) ||
        (ArenaFreeSlotCount(arena) > arena.layout.slot_count)) {
        return false;
    }

    for (std::size_t slot_index = 0; slot_index < arena.layout.slot_count; ++slot_index) {
        if (!IsValidArenaSlotState(arena.slot_states[slot_index])) {
            return false;
        }
    }

    return true;
}

inline NEUROEVOLUTION_HOST_DEVICE bool IsValidGenotypeArenaView(const ConstGenotypeArenaView &arena) noexcept {
    if (!IsValidGenotypeArenaLayout(arena.layout) || (arena.storage == nullptr) || (arena.slot_states == nullptr) ||
        (arena.free_slot_stack == nullptr) || (arena.free_slot_count == nullptr) ||
        (ArenaFreeSlotCount(arena) > arena.layout.slot_count)) {
        return false;
    }

    for (std::size_t slot_index = 0; slot_index < arena.layout.slot_count; ++slot_index) {
        if (!IsValidArenaSlotState(arena.slot_states[slot_index])) {
            return false;
        }
    }

    return true;
}

inline NEUROEVOLUTION_HOST_DEVICE bool IsValidArenaSlotIndex(const GenotypeArenaView &arena,
                                                             const std::uint32_t slot_index) noexcept {
    return IsValidGenotypeArenaView(arena) && (slot_index < arena.layout.slot_count);
}

inline NEUROEVOLUTION_HOST_DEVICE bool IsValidArenaSlotIndex(const ConstGenotypeArenaView &arena,
                                                             const std::uint32_t slot_index) noexcept {
    return IsValidGenotypeArenaView(arena) && (slot_index < arena.layout.slot_count);
}

inline bool IsValidHostGenotypeArena(const HostGenotypeArena &arena) noexcept {
    return IsValidGenotypeArenaView(MakeConstGenotypeArenaView(arena));
}

inline std::uint8_t *HostArenaSlotBytesAt(HostGenotypeArena &arena, const std::size_t slot_index) noexcept {
    return ArenaSlotBytesAt(arena.storage.get(), arena.layout, slot_index);
}

inline const std::uint8_t *HostArenaSlotBytesAt(const HostGenotypeArena &arena, const std::size_t slot_index) noexcept {
    return ArenaSlotBytesAt(arena.storage.get(), arena.layout, slot_index);
}

bool TryAllocateHostArenaStorage(HostGenotypeArena &arena);

bool TryCreateHostGenotypeArena(HostGenotypeArena &arena, std::size_t slot_count, std::size_t action_count);

inline NEUROEVOLUTION_HOST_DEVICE void ClearArenaSlotBytes(const GenotypeArenaView arena,
                                                           const std::uint32_t slot_index) noexcept {
    if (!IsValidArenaSlotIndex(arena, slot_index)) {
        return;
    }

    std::uint8_t *slot_bytes = ArenaSlotBytesAt(arena.storage, arena.layout, slot_index);
    for (std::size_t byte_index = 0; byte_index < arena.layout.slot_stride_bytes; ++byte_index) {
        slot_bytes[byte_index] = 0;
    }
}

inline NEUROEVOLUTION_HOST_DEVICE void CopyArenaSlotBytes(const std::uint8_t *source_bytes, std::uint8_t *target_bytes,
                                                          const std::size_t byte_count) noexcept {
    if ((source_bytes == nullptr) || (target_bytes == nullptr)) {
        return;
    }

    for (std::size_t byte_index = 0; byte_index < byte_count; ++byte_index) {
        target_bytes[byte_index] = source_bytes[byte_index];
    }
}

inline NEUROEVOLUTION_HOST_DEVICE bool TryAllocateArenaSlot(const GenotypeArenaView arena,
                                                            std::uint32_t &slot_index) noexcept {
    if (!IsValidGenotypeArenaView(arena) || (ArenaFreeSlotCount(arena) == 0)) {
        return false;
    }

    slot_index = arena.free_slot_stack[--(*arena.free_slot_count)];
    ArenaSlotState &slot_state = arena.slot_states[slot_index];
    if (slot_state.occupied || (slot_state.reference_count != 0)) {
        return false;
    }

    slot_state.occupied = true;
    slot_state.reference_count = 1;
    ClearArenaSlotBytes(arena, slot_index);
    return true;
}

bool TryAllocateArenaSlot(HostGenotypeArena &arena, std::uint32_t &slot_index);

inline NEUROEVOLUTION_HOST_DEVICE bool TryRetainArenaSlot(const GenotypeArenaView arena,
                                                          const std::uint32_t slot_index) noexcept {
    if (!IsValidArenaSlotIndex(arena, slot_index)) {
        return false;
    }

    ArenaSlotState &slot_state = arena.slot_states[slot_index];
    if (!slot_state.occupied || (slot_state.reference_count == 0) ||
        (slot_state.reference_count == kMaxArenaSlotReferenceCount)) {
        return false;
    }

    ++slot_state.reference_count;
    return true;
}

bool TryRetainArenaSlot(HostGenotypeArena &arena, std::uint32_t slot_index);

inline NEUROEVOLUTION_HOST_DEVICE bool TryReleaseArenaSlot(const GenotypeArenaView arena,
                                                           const std::uint32_t slot_index) noexcept {
    if (!IsValidArenaSlotIndex(arena, slot_index) || (ArenaFreeSlotCount(arena) > arena.layout.slot_count)) {
        return false;
    }

    ArenaSlotState &slot_state = arena.slot_states[slot_index];
    if (!slot_state.occupied || (slot_state.reference_count == 0)) {
        return false;
    }

    --slot_state.reference_count;
    if (slot_state.reference_count == 0) {
        slot_state.occupied = false;
        ClearArenaSlotBytes(arena, slot_index);
        if (ArenaFreeSlotCount(arena) >= arena.layout.slot_count) {
            return false;
        }

        arena.free_slot_stack[(*arena.free_slot_count)++] = slot_index;
    }

    return true;
}

bool TryReleaseArenaSlot(HostGenotypeArena &arena, std::uint32_t slot_index);

inline NEUROEVOLUTION_HOST_DEVICE bool TryCopyGenomeBytesIntoArenaSlot(const GenotypeArenaView arena,
                                                                       const std::uint32_t slot_index,
                                                                       const std::uint8_t *source_genome_bytes,
                                                                       const std::size_t source_bytes) noexcept {
    if (!IsValidArenaSlotIndex(arena, slot_index) || (source_genome_bytes == nullptr) ||
        (source_bytes != arena.layout.slot_stride_bytes) || !arena.slot_states[slot_index].occupied) {
        return false;
    }

    CopyArenaSlotBytes(source_genome_bytes, ArenaSlotBytesAt(arena.storage, arena.layout, slot_index), source_bytes);
    return true;
}

bool TryCopyGenomeBytesIntoArenaSlot(HostGenotypeArena &arena, std::uint32_t slot_index,
                                     const std::uint8_t *source_genome_bytes, std::size_t source_bytes);

inline NEUROEVOLUTION_HOST_DEVICE bool TryCloneArenaSlot(const GenotypeArenaView arena,
                                                         const std::uint32_t source_slot_index,
                                                         std::uint32_t &cloned_slot_index) noexcept {
    if (!IsValidArenaSlotIndex(arena, source_slot_index) || !arena.slot_states[source_slot_index].occupied) {
        return false;
    }

    if (!TryAllocateArenaSlot(arena, cloned_slot_index)) {
        return false;
    }

    if (!TryCopyGenomeBytesIntoArenaSlot(arena, cloned_slot_index,
                                         ArenaSlotBytesAt(arena.storage, arena.layout, source_slot_index),
                                         arena.layout.slot_stride_bytes)) {
        (void)TryReleaseArenaSlot(arena, cloned_slot_index);
        return false;
    }

    return true;
}

bool TryCloneArenaSlot(HostGenotypeArena &arena, std::uint32_t source_slot_index, std::uint32_t &cloned_slot_index);

} // namespace neuroevolution::genetic_algorithm::genotype_arena
