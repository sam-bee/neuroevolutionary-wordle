#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "common/cuda_compat.hpp"
#include "genetic_algorithm/genotype_slab/layout.hpp"

namespace neuroevolution::genetic_algorithm::genotype_slab {

struct SlabSlotState {
    bool occupied = false;
    std::uint32_t liveness_count = 0;
};

struct AlignedBufferStorageDeleter {
    void operator()(std::uint8_t *pointer) const noexcept;
};

using AlignedBufferStorage = std::unique_ptr<std::uint8_t[], AlignedBufferStorageDeleter>;

struct HostGenotypeSlab {
    GenotypeSlabLayout layout{};
    AlignedBufferStorage storage{};
    std::unique_ptr<SlabSlotState[]> slot_states{};
    std::unique_ptr<std::uint32_t[]> free_slot_stack{};
    std::uint32_t free_slot_count = 0;
    std::uint32_t free_slot_lock = 0;
};

struct GenotypeSlabView {
    GenotypeSlabLayout layout{};
    std::uint8_t *storage = nullptr;
    SlabSlotState *slot_states = nullptr;
    std::uint32_t *free_slot_stack = nullptr;
    std::uint32_t *free_slot_count = nullptr;
    std::uint32_t *free_slot_lock = nullptr;
};

struct ConstGenotypeSlabView {
    GenotypeSlabLayout layout{};
    const std::uint8_t *storage = nullptr;
    const SlabSlotState *slot_states = nullptr;
    const std::uint32_t *free_slot_stack = nullptr;
    const std::uint32_t *free_slot_count = nullptr;
    const std::uint32_t *free_slot_lock = nullptr;
};

constexpr std::uint32_t kMaxReferenceCount = static_cast<std::uint32_t>(-1);
constexpr std::uint32_t kMaxSlabSlotLivenessCount = kMaxReferenceCount;

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidSlabSlotState(const SlabSlotState &slot_state) noexcept {
    return slot_state.occupied ? (slot_state.liveness_count > 0) : (slot_state.liveness_count == 0);
}

inline GenotypeSlabView MakeGenotypeSlabView(HostGenotypeSlab &buffer) noexcept {
    return {
        .layout = buffer.layout,
        .storage = buffer.storage.get(),
        .slot_states = buffer.slot_states.get(),
        .free_slot_stack = buffer.free_slot_stack.get(),
        .free_slot_count = &buffer.free_slot_count,
        .free_slot_lock = &buffer.free_slot_lock,
    };
}

inline ConstGenotypeSlabView MakeConstGenotypeSlabView(const HostGenotypeSlab &buffer) noexcept {
    return {
        .layout = buffer.layout,
        .storage = buffer.storage.get(),
        .slot_states = buffer.slot_states.get(),
        .free_slot_stack = buffer.free_slot_stack.get(),
        .free_slot_count = &buffer.free_slot_count,
        .free_slot_lock = &buffer.free_slot_lock,
    };
}

inline NEUROEVOLUTION_HOST_DEVICE std::uint32_t SlabFreeSlotCount(const GenotypeSlabView &buffer) noexcept {
    return (buffer.free_slot_count == nullptr) ? 0 : *buffer.free_slot_count;
}

inline NEUROEVOLUTION_HOST_DEVICE std::uint32_t SlabFreeSlotCount(const ConstGenotypeSlabView &buffer) noexcept {
    return (buffer.free_slot_count == nullptr) ? 0 : *buffer.free_slot_count;
}

namespace detail {

inline NEUROEVOLUTION_HOST_DEVICE bool IsUsableGenotypeSlabView(const GenotypeSlabView &buffer) noexcept {
    return IsValidGenotypeSlabLayout(buffer.layout) && (buffer.storage != nullptr) && (buffer.slot_states != nullptr) &&
           (buffer.free_slot_stack != nullptr) && (buffer.free_slot_count != nullptr) &&
           (buffer.free_slot_lock != nullptr);
}

inline NEUROEVOLUTION_HOST_DEVICE bool LockSlabFreeList(std::uint32_t *lock) noexcept {
    if (lock == nullptr) {
        return false;
    }

#if defined(__CUDA_ARCH__)
    while (atomicCAS(lock, 0U, 1U) != 0U) {
    }
    __threadfence();
#else
    std::uint32_t expected = 0;
    while (!__atomic_compare_exchange_n(lock, &expected, 1U, false, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
        expected = 0;
    }
#endif

    return true;
}

inline NEUROEVOLUTION_HOST_DEVICE void UnlockSlabFreeList(std::uint32_t *lock) noexcept {
    if (lock == nullptr) {
        return;
    }

#if defined(__CUDA_ARCH__)
    __threadfence();
    atomicExch(lock, 0U);
#else
    __atomic_store_n(lock, 0U, __ATOMIC_RELEASE);
#endif
}

inline NEUROEVOLUTION_HOST_DEVICE bool AtomicTryIncrementSlabSlotLivenessCount(std::uint32_t *counter) noexcept {
    if (counter == nullptr) {
        return false;
    }

#if defined(__CUDA_ARCH__)
    std::uint32_t observed = *counter;
    while (observed != 0U && observed != kMaxSlabSlotLivenessCount) {
        const std::uint32_t previous = atomicCAS(counter, observed, observed + 1U);
        if (previous == observed) {
            return true;
        }

        observed = previous;
    }
#else
    std::uint32_t observed = __atomic_load_n(counter, __ATOMIC_RELAXED);
    while (observed != 0U && observed != kMaxSlabSlotLivenessCount) {
        const std::uint32_t next = observed + 1U;
        if (__atomic_compare_exchange_n(counter, &observed, next, false, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) {
            return true;
        }
    }
#endif

    return false;
}

inline NEUROEVOLUTION_HOST_DEVICE bool
AtomicTryDecrementSlabSlotLivenessCount(std::uint32_t *counter, std::uint32_t &previous_count) noexcept {
    previous_count = 0;
    if (counter == nullptr) {
        return false;
    }

#if defined(__CUDA_ARCH__)
    std::uint32_t observed = *counter;
    while (observed != 0U) {
        const std::uint32_t previous = atomicCAS(counter, observed, observed - 1U);
        if (previous == observed) {
            previous_count = observed;
            return true;
        }

        observed = previous;
    }
#else
    std::uint32_t observed = __atomic_load_n(counter, __ATOMIC_RELAXED);
    while (observed != 0U) {
        const std::uint32_t next = observed - 1U;
        if (__atomic_compare_exchange_n(counter, &observed, next, false, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) {
            previous_count = observed;
            return true;
        }
    }
#endif

    return false;
}

inline NEUROEVOLUTION_HOST_DEVICE void ClearSlabSlotBytesUnchecked(const GenotypeSlabView buffer,
                                                                   const std::uint32_t slot_index) noexcept {
    std::uint8_t *slot_bytes = SlabSlotBytesAt(buffer.storage, buffer.layout, slot_index);
    for (std::size_t byte_index = 0; byte_index < buffer.layout.slot_stride_bytes; ++byte_index) {
        slot_bytes[byte_index] = 0;
    }
}

} // namespace detail

inline NEUROEVOLUTION_HOST_DEVICE bool IsValidGenotypeSlabView(const GenotypeSlabView &buffer) noexcept {
    if (!IsValidGenotypeSlabLayout(buffer.layout) || (buffer.storage == nullptr) || (buffer.slot_states == nullptr) ||
        (buffer.free_slot_stack == nullptr) || (buffer.free_slot_count == nullptr) ||
        (buffer.free_slot_lock == nullptr) || (SlabFreeSlotCount(buffer) > buffer.layout.slot_count)) {
        return false;
    }

    for (std::size_t slot_index = 0; slot_index < buffer.layout.slot_count; ++slot_index) {
        if (!IsValidSlabSlotState(buffer.slot_states[slot_index])) {
            return false;
        }
    }

    return true;
}

inline NEUROEVOLUTION_HOST_DEVICE bool IsValidGenotypeSlabView(const ConstGenotypeSlabView &buffer) noexcept {
    if (!IsValidGenotypeSlabLayout(buffer.layout) || (buffer.storage == nullptr) || (buffer.slot_states == nullptr) ||
        (buffer.free_slot_stack == nullptr) || (buffer.free_slot_count == nullptr) ||
        (buffer.free_slot_lock == nullptr) || (SlabFreeSlotCount(buffer) > buffer.layout.slot_count)) {
        return false;
    }

    for (std::size_t slot_index = 0; slot_index < buffer.layout.slot_count; ++slot_index) {
        if (!IsValidSlabSlotState(buffer.slot_states[slot_index])) {
            return false;
        }
    }

    return true;
}

inline NEUROEVOLUTION_HOST_DEVICE bool IsValidSlabSlotIndex(const GenotypeSlabView &buffer,
                                                            const std::uint32_t slot_index) noexcept {
    return IsValidGenotypeSlabView(buffer) && (slot_index < buffer.layout.slot_count);
}

inline NEUROEVOLUTION_HOST_DEVICE bool IsValidSlabSlotIndex(const ConstGenotypeSlabView &buffer,
                                                            const std::uint32_t slot_index) noexcept {
    return IsValidGenotypeSlabView(buffer) && (slot_index < buffer.layout.slot_count);
}

inline bool IsValidHostGenotypeSlab(const HostGenotypeSlab &buffer) noexcept {
    return IsValidGenotypeSlabView(MakeConstGenotypeSlabView(buffer));
}

inline std::uint8_t *HostSlabSlotBytesAt(HostGenotypeSlab &buffer, const std::size_t slot_index) noexcept {
    return SlabSlotBytesAt(buffer.storage.get(), buffer.layout, slot_index);
}

inline const std::uint8_t *HostSlabSlotBytesAt(const HostGenotypeSlab &buffer, const std::size_t slot_index) noexcept {
    return SlabSlotBytesAt(buffer.storage.get(), buffer.layout, slot_index);
}

bool TryAllocateHostSlabStorage(HostGenotypeSlab &buffer);

bool TryCreateHostGenotypeSlab(HostGenotypeSlab &buffer, const GenotypeSlabLayout &layout);

bool TryCreateHostGenotypeSlabForByteBudget(HostGenotypeSlab &buffer, std::size_t slab_byte_budget_bytes,
                                            std::size_t action_count);

bool TryCreateHostGenotypeSlab(HostGenotypeSlab &buffer, std::size_t slot_count, std::size_t action_count);

inline NEUROEVOLUTION_HOST_DEVICE void ClearSlabSlotBytes(const GenotypeSlabView buffer,
                                                          const std::uint32_t slot_index) noexcept {
    if (!IsValidSlabSlotIndex(buffer, slot_index)) {
        return;
    }

    std::uint8_t *slot_bytes = SlabSlotBytesAt(buffer.storage, buffer.layout, slot_index);
    for (std::size_t byte_index = 0; byte_index < buffer.layout.slot_stride_bytes; ++byte_index) {
        slot_bytes[byte_index] = 0;
    }
}

inline NEUROEVOLUTION_HOST_DEVICE void CopySlabSlotBytes(const std::uint8_t *source_bytes, std::uint8_t *target_bytes,
                                                         const std::size_t byte_count) noexcept {
    if ((source_bytes == nullptr) || (target_bytes == nullptr)) {
        return;
    }

    for (std::size_t byte_index = 0; byte_index < byte_count; ++byte_index) {
        target_bytes[byte_index] = source_bytes[byte_index];
    }
}

inline NEUROEVOLUTION_HOST_DEVICE bool TryAllocateSlabSlot(const GenotypeSlabView buffer,
                                                           std::uint32_t &slot_index) noexcept {
    if (!detail::IsUsableGenotypeSlabView(buffer)) {
        return false;
    }

    if (!detail::LockSlabFreeList(buffer.free_slot_lock)) {
        return false;
    }

    if (*buffer.free_slot_count == 0) {
        detail::UnlockSlabFreeList(buffer.free_slot_lock);
        return false;
    }

    --(*buffer.free_slot_count);
    slot_index = buffer.free_slot_stack[*buffer.free_slot_count];
    if (slot_index >= buffer.layout.slot_count) {
        ++(*buffer.free_slot_count);
        detail::UnlockSlabFreeList(buffer.free_slot_lock);
        return false;
    }

    SlabSlotState &slot_state = buffer.slot_states[slot_index];
    if (slot_state.occupied || (slot_state.liveness_count != 0)) {
        buffer.free_slot_stack[*buffer.free_slot_count] = slot_index;
        ++(*buffer.free_slot_count);
        detail::UnlockSlabFreeList(buffer.free_slot_lock);
        return false;
    }

    slot_state.occupied = true;
    slot_state.liveness_count = 1;
    detail::UnlockSlabFreeList(buffer.free_slot_lock);
    detail::ClearSlabSlotBytesUnchecked(buffer, slot_index);
    return true;
}

bool TryAllocateSlabSlot(HostGenotypeSlab &buffer, std::uint32_t &slot_index);

inline NEUROEVOLUTION_HOST_DEVICE bool TryRetainSlabSlot(const GenotypeSlabView buffer,
                                                         const std::uint32_t slot_index) noexcept {
    if (!detail::IsUsableGenotypeSlabView(buffer) || (slot_index >= buffer.layout.slot_count)) {
        return false;
    }

    SlabSlotState &slot_state = buffer.slot_states[slot_index];
    return detail::AtomicTryIncrementSlabSlotLivenessCount(&slot_state.liveness_count);
}

bool TryRetainSlabSlot(HostGenotypeSlab &buffer, std::uint32_t slot_index);

inline NEUROEVOLUTION_HOST_DEVICE bool TryReleaseSlabSlot(const GenotypeSlabView buffer,
                                                          const std::uint32_t slot_index) noexcept {
    if (!detail::IsUsableGenotypeSlabView(buffer) || (slot_index >= buffer.layout.slot_count)) {
        return false;
    }

    SlabSlotState &slot_state = buffer.slot_states[slot_index];
    std::uint32_t previous_liveness_count = 0;
    if (!detail::AtomicTryDecrementSlabSlotLivenessCount(&slot_state.liveness_count, previous_liveness_count)) {
        return false;
    }

    if (previous_liveness_count != 1U) {
        return true;
    }

    detail::ClearSlabSlotBytesUnchecked(buffer, slot_index);
    slot_state.occupied = false;
    if (!detail::LockSlabFreeList(buffer.free_slot_lock)) {
        return false;
    }

    if (*buffer.free_slot_count >= buffer.layout.slot_count) {
        detail::UnlockSlabFreeList(buffer.free_slot_lock);
        return false;
    }

    buffer.free_slot_stack[*buffer.free_slot_count] = slot_index;
    ++(*buffer.free_slot_count);
    detail::UnlockSlabFreeList(buffer.free_slot_lock);
    return true;
}

bool TryReleaseSlabSlot(HostGenotypeSlab &buffer, std::uint32_t slot_index);

inline NEUROEVOLUTION_HOST_DEVICE bool TryCopyGenomeBytesIntoSlabSlot(const GenotypeSlabView buffer,
                                                                      const std::uint32_t slot_index,
                                                                      const std::uint8_t *source_genome_bytes,
                                                                      const std::size_t source_bytes) noexcept {
    if (!IsValidSlabSlotIndex(buffer, slot_index) || (source_genome_bytes == nullptr) ||
        (source_bytes != buffer.layout.slot_stride_bytes) || !buffer.slot_states[slot_index].occupied) {
        return false;
    }

    CopySlabSlotBytes(source_genome_bytes, SlabSlotBytesAt(buffer.storage, buffer.layout, slot_index), source_bytes);
    return true;
}

bool TryCopyGenomeBytesIntoSlabSlot(HostGenotypeSlab &buffer, std::uint32_t slot_index,
                                    const std::uint8_t *source_genome_bytes, std::size_t source_bytes);

inline NEUROEVOLUTION_HOST_DEVICE bool TryCloneSlabSlot(const GenotypeSlabView buffer,
                                                        const std::uint32_t source_slot_index,
                                                        std::uint32_t &cloned_slot_index) noexcept {
    if (!IsValidSlabSlotIndex(buffer, source_slot_index) || !buffer.slot_states[source_slot_index].occupied) {
        return false;
    }

    if (!TryAllocateSlabSlot(buffer, cloned_slot_index)) {
        return false;
    }

    if (!TryCopyGenomeBytesIntoSlabSlot(buffer, cloned_slot_index,
                                        SlabSlotBytesAt(buffer.storage, buffer.layout, source_slot_index),
                                        buffer.layout.slot_stride_bytes)) {
        (void)TryReleaseSlabSlot(buffer, cloned_slot_index);
        return false;
    }

    return true;
}

bool TryCloneSlabSlot(HostGenotypeSlab &buffer, std::uint32_t source_slot_index, std::uint32_t &cloned_slot_index);

} // namespace neuroevolution::genetic_algorithm::genotype_slab
