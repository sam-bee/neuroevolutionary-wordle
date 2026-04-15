#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "common/cuda_compat.hpp"
#include "genetic_algorithm/genotype_pool/layout.hpp"

namespace neuroevolution::genetic_algorithm::genotype_pool {

struct PoolSlotState {
    bool occupied = false;
    std::uint32_t reference_count = 0;
};

struct AlignedPoolStorageDeleter {
    void operator()(std::uint8_t *pointer) const noexcept;
};

using AlignedPoolStorage = std::unique_ptr<std::uint8_t[], AlignedPoolStorageDeleter>;

struct HostGenotypePool {
    GenotypePoolLayout layout{};
    AlignedPoolStorage storage{};
    std::unique_ptr<PoolSlotState[]> slot_states{};
    std::unique_ptr<std::uint32_t[]> free_slot_stack{};
    std::size_t free_slot_count = 0;
};

struct GenotypePoolView {
    GenotypePoolLayout layout{};
    std::uint8_t *storage = nullptr;
    PoolSlotState *slot_states = nullptr;
    std::uint32_t *free_slot_stack = nullptr;
    std::size_t *free_slot_count = nullptr;
};

struct ConstGenotypePoolView {
    GenotypePoolLayout layout{};
    const std::uint8_t *storage = nullptr;
    const PoolSlotState *slot_states = nullptr;
    const std::uint32_t *free_slot_stack = nullptr;
    const std::size_t *free_slot_count = nullptr;
};

constexpr std::uint32_t kMaxPoolSlotReferenceCount = static_cast<std::uint32_t>(-1);

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidPoolSlotState(const PoolSlotState &slot_state) noexcept {
    return slot_state.occupied ? (slot_state.reference_count > 0) : (slot_state.reference_count == 0);
}

inline GenotypePoolView MakeGenotypePoolView(HostGenotypePool &pool) noexcept {
    return {
        .layout = pool.layout,
        .storage = pool.storage.get(),
        .slot_states = pool.slot_states.get(),
        .free_slot_stack = pool.free_slot_stack.get(),
        .free_slot_count = &pool.free_slot_count,
    };
}

inline ConstGenotypePoolView MakeConstGenotypePoolView(const HostGenotypePool &pool) noexcept {
    return {
        .layout = pool.layout,
        .storage = pool.storage.get(),
        .slot_states = pool.slot_states.get(),
        .free_slot_stack = pool.free_slot_stack.get(),
        .free_slot_count = &pool.free_slot_count,
    };
}

inline NEUROEVOLUTION_HOST_DEVICE std::size_t PoolFreeSlotCount(const GenotypePoolView &pool) noexcept {
    return (pool.free_slot_count == nullptr) ? 0 : *pool.free_slot_count;
}

inline NEUROEVOLUTION_HOST_DEVICE std::size_t PoolFreeSlotCount(const ConstGenotypePoolView &pool) noexcept {
    return (pool.free_slot_count == nullptr) ? 0 : *pool.free_slot_count;
}

inline NEUROEVOLUTION_HOST_DEVICE bool IsValidGenotypePoolView(const GenotypePoolView &pool) noexcept {
    if (!IsValidGenotypePoolLayout(pool.layout) || (pool.storage == nullptr) || (pool.slot_states == nullptr) ||
        (pool.free_slot_stack == nullptr) || (pool.free_slot_count == nullptr) ||
        (PoolFreeSlotCount(pool) > pool.layout.slot_count)) {
        return false;
    }

    for (std::size_t slot_index = 0; slot_index < pool.layout.slot_count; ++slot_index) {
        if (!IsValidPoolSlotState(pool.slot_states[slot_index])) {
            return false;
        }
    }

    return true;
}

inline NEUROEVOLUTION_HOST_DEVICE bool IsValidGenotypePoolView(const ConstGenotypePoolView &pool) noexcept {
    if (!IsValidGenotypePoolLayout(pool.layout) || (pool.storage == nullptr) || (pool.slot_states == nullptr) ||
        (pool.free_slot_stack == nullptr) || (pool.free_slot_count == nullptr) ||
        (PoolFreeSlotCount(pool) > pool.layout.slot_count)) {
        return false;
    }

    for (std::size_t slot_index = 0; slot_index < pool.layout.slot_count; ++slot_index) {
        if (!IsValidPoolSlotState(pool.slot_states[slot_index])) {
            return false;
        }
    }

    return true;
}

inline NEUROEVOLUTION_HOST_DEVICE bool IsValidPoolSlotIndex(const GenotypePoolView &pool,
                                                            const std::uint32_t slot_index) noexcept {
    return IsValidGenotypePoolView(pool) && (slot_index < pool.layout.slot_count);
}

inline NEUROEVOLUTION_HOST_DEVICE bool IsValidPoolSlotIndex(const ConstGenotypePoolView &pool,
                                                            const std::uint32_t slot_index) noexcept {
    return IsValidGenotypePoolView(pool) && (slot_index < pool.layout.slot_count);
}

inline bool IsValidHostGenotypePool(const HostGenotypePool &pool) noexcept {
    return IsValidGenotypePoolView(MakeConstGenotypePoolView(pool));
}

inline std::uint8_t *HostPoolSlotBytesAt(HostGenotypePool &pool, const std::size_t slot_index) noexcept {
    return PoolSlotBytesAt(pool.storage.get(), pool.layout, slot_index);
}

inline const std::uint8_t *HostPoolSlotBytesAt(const HostGenotypePool &pool, const std::size_t slot_index) noexcept {
    return PoolSlotBytesAt(pool.storage.get(), pool.layout, slot_index);
}

bool TryAllocateHostPoolStorage(HostGenotypePool &pool);

bool TryCreateHostGenotypePool(HostGenotypePool &pool, const GenotypePoolLayout &layout);

bool TryCreateHostGenotypePoolForByteBudget(HostGenotypePool &pool, std::size_t pool_byte_budget_bytes,
                                            std::size_t action_count);

bool TryCreateHostGenotypePool(HostGenotypePool &pool, std::size_t slot_count, std::size_t action_count);

inline NEUROEVOLUTION_HOST_DEVICE void ClearPoolSlotBytes(const GenotypePoolView pool,
                                                          const std::uint32_t slot_index) noexcept {
    if (!IsValidPoolSlotIndex(pool, slot_index)) {
        return;
    }

    std::uint8_t *slot_bytes = PoolSlotBytesAt(pool.storage, pool.layout, slot_index);
    for (std::size_t byte_index = 0; byte_index < pool.layout.slot_stride_bytes; ++byte_index) {
        slot_bytes[byte_index] = 0;
    }
}

inline NEUROEVOLUTION_HOST_DEVICE void CopyPoolSlotBytes(const std::uint8_t *source_bytes, std::uint8_t *target_bytes,
                                                         const std::size_t byte_count) noexcept {
    if ((source_bytes == nullptr) || (target_bytes == nullptr)) {
        return;
    }

    for (std::size_t byte_index = 0; byte_index < byte_count; ++byte_index) {
        target_bytes[byte_index] = source_bytes[byte_index];
    }
}

inline NEUROEVOLUTION_HOST_DEVICE bool TryAllocatePoolSlot(const GenotypePoolView pool,
                                                           std::uint32_t &slot_index) noexcept {
    if (!IsValidGenotypePoolView(pool) || (PoolFreeSlotCount(pool) == 0)) {
        return false;
    }

    slot_index = pool.free_slot_stack[--(*pool.free_slot_count)];
    PoolSlotState &slot_state = pool.slot_states[slot_index];
    if (slot_state.occupied || (slot_state.reference_count != 0)) {
        return false;
    }

    slot_state.occupied = true;
    slot_state.reference_count = 1;
    ClearPoolSlotBytes(pool, slot_index);
    return true;
}

bool TryAllocatePoolSlot(HostGenotypePool &pool, std::uint32_t &slot_index);

inline NEUROEVOLUTION_HOST_DEVICE bool TryRetainPoolSlot(const GenotypePoolView pool,
                                                         const std::uint32_t slot_index) noexcept {
    if (!IsValidPoolSlotIndex(pool, slot_index)) {
        return false;
    }

    PoolSlotState &slot_state = pool.slot_states[slot_index];
    if (!slot_state.occupied || (slot_state.reference_count == 0) ||
        (slot_state.reference_count == kMaxPoolSlotReferenceCount)) {
        return false;
    }

    ++slot_state.reference_count;
    return true;
}

bool TryRetainPoolSlot(HostGenotypePool &pool, std::uint32_t slot_index);

inline NEUROEVOLUTION_HOST_DEVICE bool TryReleasePoolSlot(const GenotypePoolView pool,
                                                          const std::uint32_t slot_index) noexcept {
    if (!IsValidPoolSlotIndex(pool, slot_index) || (PoolFreeSlotCount(pool) > pool.layout.slot_count)) {
        return false;
    }

    PoolSlotState &slot_state = pool.slot_states[slot_index];
    if (!slot_state.occupied || (slot_state.reference_count == 0)) {
        return false;
    }

    --slot_state.reference_count;
    if (slot_state.reference_count == 0) {
        slot_state.occupied = false;
        ClearPoolSlotBytes(pool, slot_index);
        if (PoolFreeSlotCount(pool) >= pool.layout.slot_count) {
            return false;
        }

        pool.free_slot_stack[(*pool.free_slot_count)++] = slot_index;
    }

    return true;
}

bool TryReleasePoolSlot(HostGenotypePool &pool, std::uint32_t slot_index);

inline NEUROEVOLUTION_HOST_DEVICE bool TryCopyGenomeBytesIntoPoolSlot(const GenotypePoolView pool,
                                                                      const std::uint32_t slot_index,
                                                                      const std::uint8_t *source_genome_bytes,
                                                                      const std::size_t source_bytes) noexcept {
    if (!IsValidPoolSlotIndex(pool, slot_index) || (source_genome_bytes == nullptr) ||
        (source_bytes != pool.layout.slot_stride_bytes) || !pool.slot_states[slot_index].occupied) {
        return false;
    }

    CopyPoolSlotBytes(source_genome_bytes, PoolSlotBytesAt(pool.storage, pool.layout, slot_index), source_bytes);
    return true;
}

bool TryCopyGenomeBytesIntoPoolSlot(HostGenotypePool &pool, std::uint32_t slot_index,
                                    const std::uint8_t *source_genome_bytes, std::size_t source_bytes);

inline NEUROEVOLUTION_HOST_DEVICE bool TryClonePoolSlot(const GenotypePoolView pool,
                                                        const std::uint32_t source_slot_index,
                                                        std::uint32_t &cloned_slot_index) noexcept {
    if (!IsValidPoolSlotIndex(pool, source_slot_index) || !pool.slot_states[source_slot_index].occupied) {
        return false;
    }

    if (!TryAllocatePoolSlot(pool, cloned_slot_index)) {
        return false;
    }

    if (!TryCopyGenomeBytesIntoPoolSlot(pool, cloned_slot_index,
                                        PoolSlotBytesAt(pool.storage, pool.layout, source_slot_index),
                                        pool.layout.slot_stride_bytes)) {
        (void)TryReleasePoolSlot(pool, cloned_slot_index);
        return false;
    }

    return true;
}

bool TryClonePoolSlot(HostGenotypePool &pool, std::uint32_t source_slot_index, std::uint32_t &cloned_slot_index);

} // namespace neuroevolution::genetic_algorithm::genotype_pool
