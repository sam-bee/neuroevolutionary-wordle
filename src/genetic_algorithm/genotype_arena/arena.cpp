#include "genetic_algorithm/genotype_arena/arena.hpp"

#include <cstring>
#include <limits>
#include <new>

namespace neuroevolution::genetic_algorithm::genotype_arena {

namespace {

void ClearArenaSlotBytes(const GenotypeArenaView arena, const std::uint32_t slot_index) {
    std::memset(ArenaSlotBytesAt(arena.storage, arena.layout, slot_index), 0, arena.layout.slot_stride_bytes);
}

} // namespace

void AlignedArenaStorageDeleter::operator()(std::uint8_t *pointer) const noexcept {
    if (pointer != nullptr) {
        ::operator delete[](pointer, std::align_val_t(DynamicGenomeAlignment()));
    }
}

bool TryAllocateHostArenaStorage(HostGenotypeArena &arena) {
    arena.storage.reset();
    arena.slot_states.reset();
    arena.free_slot_stack.reset();
    arena.free_slot_count = 0;

    if (!IsValidGenotypeArenaLayout(arena.layout)) {
        return false;
    }

    try {
        arena.storage.reset(static_cast<std::uint8_t *>(
            ::operator new[](arena.layout.arena_bytes, std::align_val_t(DynamicGenomeAlignment()))));
    } catch (...) {
        return false;
    }

    arena.slot_states.reset(new (std::nothrow) ArenaSlotState[arena.layout.slot_count]());
    arena.free_slot_stack.reset(new (std::nothrow) std::uint32_t[arena.layout.slot_count]());
    if ((arena.slot_states == nullptr) || (arena.free_slot_stack == nullptr)) {
        arena = {};
        return false;
    }

    std::memset(arena.storage.get(), 0, arena.layout.arena_bytes);
    for (std::size_t slot_index = 0; slot_index < arena.layout.slot_count; ++slot_index) {
        arena.free_slot_stack[slot_index] = static_cast<std::uint32_t>((arena.layout.slot_count - 1) - slot_index);
    }
    arena.free_slot_count = arena.layout.slot_count;
    return true;
}

bool TryCreateHostGenotypeArena(HostGenotypeArena &arena, const std::size_t slot_count,
                                const std::size_t action_count) {
    arena = {};

    const std::size_t slot_stride_bytes = ComputeArenaSlotStrideBytes(action_count);
    if ((slot_count == 0) || (slot_stride_bytes == 0) ||
        (slot_count > (std::numeric_limits<std::size_t>::max() / slot_stride_bytes))) {
        return false;
    }

    arena.layout.action_count = action_count;
    arena.layout.slot_stride_bytes = slot_stride_bytes;
    arena.layout.slot_count = slot_count;
    arena.layout.arena_bytes = slot_count * slot_stride_bytes;
    if (!IsValidGenotypeArenaLayout(arena.layout)) {
        arena = {};
        return false;
    }

    return TryAllocateHostArenaStorage(arena);
}

bool TryAllocateArenaSlot(const GenotypeArenaView arena, std::uint32_t &slot_index) {
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

bool TryAllocateArenaSlot(HostGenotypeArena &arena, std::uint32_t &slot_index) {
    return TryAllocateArenaSlot(MakeGenotypeArenaView(arena), slot_index);
}

bool TryRetainArenaSlot(const GenotypeArenaView arena, const std::uint32_t slot_index) {
    if (!IsValidGenotypeArenaView(arena) || (slot_index >= arena.layout.slot_count)) {
        return false;
    }

    ArenaSlotState &slot_state = arena.slot_states[slot_index];
    if (!slot_state.occupied || (slot_state.reference_count == 0) ||
        (slot_state.reference_count == std::numeric_limits<std::uint32_t>::max())) {
        return false;
    }

    ++slot_state.reference_count;
    return true;
}

bool TryRetainArenaSlot(HostGenotypeArena &arena, const std::uint32_t slot_index) {
    return TryRetainArenaSlot(MakeGenotypeArenaView(arena), slot_index);
}

bool TryReleaseArenaSlot(const GenotypeArenaView arena, const std::uint32_t slot_index) {
    if (!IsValidGenotypeArenaView(arena) || (slot_index >= arena.layout.slot_count)) {
        return false;
    }

    ArenaSlotState &slot_state = arena.slot_states[slot_index];
    if (!slot_state.occupied || (slot_state.reference_count == 0) ||
        (ArenaFreeSlotCount(arena) > arena.layout.slot_count)) {
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

bool TryReleaseArenaSlot(HostGenotypeArena &arena, const std::uint32_t slot_index) {
    return TryReleaseArenaSlot(MakeGenotypeArenaView(arena), slot_index);
}

bool TryCopyGenomeBytesIntoArenaSlot(const GenotypeArenaView arena, const std::uint32_t slot_index,
                                     const std::uint8_t *source_genome_bytes, const std::size_t source_bytes) {
    if (!IsValidGenotypeArenaView(arena) || (slot_index >= arena.layout.slot_count) ||
        (source_genome_bytes == nullptr) || (source_bytes != arena.layout.slot_stride_bytes) ||
        !arena.slot_states[slot_index].occupied) {
        return false;
    }

    std::memcpy(ArenaSlotBytesAt(arena.storage, arena.layout, slot_index), source_genome_bytes, source_bytes);
    return true;
}

bool TryCopyGenomeBytesIntoArenaSlot(HostGenotypeArena &arena, const std::uint32_t slot_index,
                                     const std::uint8_t *source_genome_bytes, const std::size_t source_bytes) {
    return TryCopyGenomeBytesIntoArenaSlot(MakeGenotypeArenaView(arena), slot_index, source_genome_bytes, source_bytes);
}

bool TryCloneArenaSlot(const GenotypeArenaView arena, const std::uint32_t source_slot_index,
                       std::uint32_t &cloned_slot_index) {
    if (!IsValidGenotypeArenaView(arena) || (source_slot_index >= arena.layout.slot_count) ||
        !arena.slot_states[source_slot_index].occupied) {
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

bool TryCloneArenaSlot(HostGenotypeArena &arena, const std::uint32_t source_slot_index,
                       std::uint32_t &cloned_slot_index) {
    return TryCloneArenaSlot(MakeGenotypeArenaView(arena), source_slot_index, cloned_slot_index);
}

} // namespace neuroevolution::genetic_algorithm::genotype_arena
