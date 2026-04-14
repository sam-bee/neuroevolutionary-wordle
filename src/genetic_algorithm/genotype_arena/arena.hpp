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

constexpr bool IsValidArenaSlotState(const ArenaSlotState &slot_state) noexcept {
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

bool TryAllocateArenaSlot(GenotypeArenaView arena, std::uint32_t &slot_index);

bool TryAllocateArenaSlot(HostGenotypeArena &arena, std::uint32_t &slot_index);

bool TryRetainArenaSlot(GenotypeArenaView arena, std::uint32_t slot_index);

bool TryRetainArenaSlot(HostGenotypeArena &arena, std::uint32_t slot_index);

bool TryReleaseArenaSlot(GenotypeArenaView arena, std::uint32_t slot_index);

bool TryReleaseArenaSlot(HostGenotypeArena &arena, std::uint32_t slot_index);

bool TryCopyGenomeBytesIntoArenaSlot(GenotypeArenaView arena, std::uint32_t slot_index,
                                     const std::uint8_t *source_genome_bytes, std::size_t source_bytes);

bool TryCopyGenomeBytesIntoArenaSlot(HostGenotypeArena &arena, std::uint32_t slot_index,
                                     const std::uint8_t *source_genome_bytes, std::size_t source_bytes);

bool TryCloneArenaSlot(GenotypeArenaView arena, std::uint32_t source_slot_index, std::uint32_t &cloned_slot_index);

bool TryCloneArenaSlot(HostGenotypeArena &arena, std::uint32_t source_slot_index, std::uint32_t &cloned_slot_index);

} // namespace neuroevolution::genetic_algorithm::genotype_arena
