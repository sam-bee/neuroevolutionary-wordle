#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "common/cuda_compat.hpp"
#include "genetic_algorithm/genotype_buffer/layout.hpp"

namespace neuroevolution::genetic_algorithm::genotype_buffer {

struct BufferSlotState {
    bool occupied = false;
    std::uint32_t reference_count = 0;
};

struct AlignedBufferStorageDeleter {
    void operator()(std::uint8_t *pointer) const noexcept;
};

using AlignedBufferStorage = std::unique_ptr<std::uint8_t[], AlignedBufferStorageDeleter>;

struct HostGenotypeBuffer {
    GenotypeBufferLayout layout{};
    AlignedBufferStorage storage{};
    std::unique_ptr<BufferSlotState[]> slot_states{};
    std::unique_ptr<std::uint32_t[]> free_slot_stack{};
    std::uint32_t free_slot_count = 0;
    std::uint32_t free_slot_lock = 0;
};

struct GenotypeBufferView {
    GenotypeBufferLayout layout{};
    std::uint8_t *storage = nullptr;
    BufferSlotState *slot_states = nullptr;
    std::uint32_t *free_slot_stack = nullptr;
    std::uint32_t *free_slot_count = nullptr;
    std::uint32_t *free_slot_lock = nullptr;
};

struct ConstGenotypeBufferView {
    GenotypeBufferLayout layout{};
    const std::uint8_t *storage = nullptr;
    const BufferSlotState *slot_states = nullptr;
    const std::uint32_t *free_slot_stack = nullptr;
    const std::uint32_t *free_slot_count = nullptr;
    const std::uint32_t *free_slot_lock = nullptr;
};

constexpr std::uint32_t kMaxBufferSlotReferenceCount = static_cast<std::uint32_t>(-1);

constexpr NEUROEVOLUTION_HOST_DEVICE bool IsValidBufferSlotState(const BufferSlotState &slot_state) noexcept {
    return slot_state.occupied ? (slot_state.reference_count > 0) : (slot_state.reference_count == 0);
}

inline GenotypeBufferView MakeGenotypeBufferView(HostGenotypeBuffer &buffer) noexcept {
    return {
        .layout = buffer.layout,
        .storage = buffer.storage.get(),
        .slot_states = buffer.slot_states.get(),
        .free_slot_stack = buffer.free_slot_stack.get(),
        .free_slot_count = &buffer.free_slot_count,
        .free_slot_lock = &buffer.free_slot_lock,
    };
}

inline ConstGenotypeBufferView MakeConstGenotypeBufferView(const HostGenotypeBuffer &buffer) noexcept {
    return {
        .layout = buffer.layout,
        .storage = buffer.storage.get(),
        .slot_states = buffer.slot_states.get(),
        .free_slot_stack = buffer.free_slot_stack.get(),
        .free_slot_count = &buffer.free_slot_count,
        .free_slot_lock = &buffer.free_slot_lock,
    };
}

inline NEUROEVOLUTION_HOST_DEVICE std::uint32_t BufferFreeSlotCount(const GenotypeBufferView &buffer) noexcept {
    return (buffer.free_slot_count == nullptr) ? 0 : *buffer.free_slot_count;
}

inline NEUROEVOLUTION_HOST_DEVICE std::uint32_t BufferFreeSlotCount(const ConstGenotypeBufferView &buffer) noexcept {
    return (buffer.free_slot_count == nullptr) ? 0 : *buffer.free_slot_count;
}

namespace detail {

inline NEUROEVOLUTION_HOST_DEVICE bool IsUsableGenotypeBufferView(const GenotypeBufferView &buffer) noexcept {
    return IsValidGenotypeBufferLayout(buffer.layout) && (buffer.storage != nullptr) &&
           (buffer.slot_states != nullptr) && (buffer.free_slot_stack != nullptr) &&
           (buffer.free_slot_count != nullptr) && (buffer.free_slot_lock != nullptr);
}

inline NEUROEVOLUTION_HOST_DEVICE bool LockBufferFreeList(std::uint32_t *lock) noexcept {
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

inline NEUROEVOLUTION_HOST_DEVICE void UnlockBufferFreeList(std::uint32_t *lock) noexcept {
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

inline NEUROEVOLUTION_HOST_DEVICE bool AtomicTryIncrementSlotReferenceCount(std::uint32_t *counter) noexcept {
    if (counter == nullptr) {
        return false;
    }

#if defined(__CUDA_ARCH__)
    std::uint32_t observed = *counter;
    while (observed != 0U && observed != kMaxBufferSlotReferenceCount) {
        const std::uint32_t previous = atomicCAS(counter, observed, observed + 1U);
        if (previous == observed) {
            return true;
        }

        observed = previous;
    }
#else
    std::uint32_t observed = __atomic_load_n(counter, __ATOMIC_RELAXED);
    while (observed != 0U && observed != kMaxBufferSlotReferenceCount) {
        const std::uint32_t next = observed + 1U;
        if (__atomic_compare_exchange_n(counter, &observed, next, false, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) {
            return true;
        }
    }
#endif

    return false;
}

inline NEUROEVOLUTION_HOST_DEVICE bool AtomicTryDecrementSlotReferenceCount(std::uint32_t *counter,
                                                                            std::uint32_t &previous_count) noexcept {
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

inline NEUROEVOLUTION_HOST_DEVICE void ClearBufferSlotBytesUnchecked(const GenotypeBufferView buffer,
                                                                     const std::uint32_t slot_index) noexcept {
    std::uint8_t *slot_bytes = BufferSlotBytesAt(buffer.storage, buffer.layout, slot_index);
    for (std::size_t byte_index = 0; byte_index < buffer.layout.slot_stride_bytes; ++byte_index) {
        slot_bytes[byte_index] = 0;
    }
}

} // namespace detail

inline NEUROEVOLUTION_HOST_DEVICE bool IsValidGenotypeBufferView(const GenotypeBufferView &buffer) noexcept {
    if (!IsValidGenotypeBufferLayout(buffer.layout) || (buffer.storage == nullptr) || (buffer.slot_states == nullptr) ||
        (buffer.free_slot_stack == nullptr) || (buffer.free_slot_count == nullptr) ||
        (buffer.free_slot_lock == nullptr) || (BufferFreeSlotCount(buffer) > buffer.layout.slot_count)) {
        return false;
    }

    for (std::size_t slot_index = 0; slot_index < buffer.layout.slot_count; ++slot_index) {
        if (!IsValidBufferSlotState(buffer.slot_states[slot_index])) {
            return false;
        }
    }

    return true;
}

inline NEUROEVOLUTION_HOST_DEVICE bool IsValidGenotypeBufferView(const ConstGenotypeBufferView &buffer) noexcept {
    if (!IsValidGenotypeBufferLayout(buffer.layout) || (buffer.storage == nullptr) || (buffer.slot_states == nullptr) ||
        (buffer.free_slot_stack == nullptr) || (buffer.free_slot_count == nullptr) ||
        (buffer.free_slot_lock == nullptr) || (BufferFreeSlotCount(buffer) > buffer.layout.slot_count)) {
        return false;
    }

    for (std::size_t slot_index = 0; slot_index < buffer.layout.slot_count; ++slot_index) {
        if (!IsValidBufferSlotState(buffer.slot_states[slot_index])) {
            return false;
        }
    }

    return true;
}

inline NEUROEVOLUTION_HOST_DEVICE bool IsValidBufferSlotIndex(const GenotypeBufferView &buffer,
                                                              const std::uint32_t slot_index) noexcept {
    return IsValidGenotypeBufferView(buffer) && (slot_index < buffer.layout.slot_count);
}

inline NEUROEVOLUTION_HOST_DEVICE bool IsValidBufferSlotIndex(const ConstGenotypeBufferView &buffer,
                                                              const std::uint32_t slot_index) noexcept {
    return IsValidGenotypeBufferView(buffer) && (slot_index < buffer.layout.slot_count);
}

inline bool IsValidHostGenotypeBuffer(const HostGenotypeBuffer &buffer) noexcept {
    return IsValidGenotypeBufferView(MakeConstGenotypeBufferView(buffer));
}

inline std::uint8_t *HostBufferSlotBytesAt(HostGenotypeBuffer &buffer, const std::size_t slot_index) noexcept {
    return BufferSlotBytesAt(buffer.storage.get(), buffer.layout, slot_index);
}

inline const std::uint8_t *HostBufferSlotBytesAt(const HostGenotypeBuffer &buffer,
                                                 const std::size_t slot_index) noexcept {
    return BufferSlotBytesAt(buffer.storage.get(), buffer.layout, slot_index);
}

bool TryAllocateHostBufferStorage(HostGenotypeBuffer &buffer);

bool TryCreateHostGenotypeBuffer(HostGenotypeBuffer &buffer, const GenotypeBufferLayout &layout);

bool TryCreateHostGenotypeBufferForByteBudget(HostGenotypeBuffer &buffer, std::size_t buffer_byte_budget_bytes,
                                              std::size_t action_count);

bool TryCreateHostGenotypeBuffer(HostGenotypeBuffer &buffer, std::size_t slot_count, std::size_t action_count);

inline NEUROEVOLUTION_HOST_DEVICE void ClearBufferSlotBytes(const GenotypeBufferView buffer,
                                                            const std::uint32_t slot_index) noexcept {
    if (!IsValidBufferSlotIndex(buffer, slot_index)) {
        return;
    }

    std::uint8_t *slot_bytes = BufferSlotBytesAt(buffer.storage, buffer.layout, slot_index);
    for (std::size_t byte_index = 0; byte_index < buffer.layout.slot_stride_bytes; ++byte_index) {
        slot_bytes[byte_index] = 0;
    }
}

inline NEUROEVOLUTION_HOST_DEVICE void CopyBufferSlotBytes(const std::uint8_t *source_bytes, std::uint8_t *target_bytes,
                                                           const std::size_t byte_count) noexcept {
    if ((source_bytes == nullptr) || (target_bytes == nullptr)) {
        return;
    }

    for (std::size_t byte_index = 0; byte_index < byte_count; ++byte_index) {
        target_bytes[byte_index] = source_bytes[byte_index];
    }
}

inline NEUROEVOLUTION_HOST_DEVICE bool TryAllocateBufferSlot(const GenotypeBufferView buffer,
                                                             std::uint32_t &slot_index) noexcept {
    if (!detail::IsUsableGenotypeBufferView(buffer)) {
        return false;
    }

    if (!detail::LockBufferFreeList(buffer.free_slot_lock)) {
        return false;
    }

    if (*buffer.free_slot_count == 0) {
        detail::UnlockBufferFreeList(buffer.free_slot_lock);
        return false;
    }

    --(*buffer.free_slot_count);
    slot_index = buffer.free_slot_stack[*buffer.free_slot_count];
    if (slot_index >= buffer.layout.slot_count) {
        ++(*buffer.free_slot_count);
        detail::UnlockBufferFreeList(buffer.free_slot_lock);
        return false;
    }

    BufferSlotState &slot_state = buffer.slot_states[slot_index];
    if (slot_state.occupied || (slot_state.reference_count != 0)) {
        buffer.free_slot_stack[*buffer.free_slot_count] = slot_index;
        ++(*buffer.free_slot_count);
        detail::UnlockBufferFreeList(buffer.free_slot_lock);
        return false;
    }

    slot_state.occupied = true;
    slot_state.reference_count = 1;
    detail::UnlockBufferFreeList(buffer.free_slot_lock);
    detail::ClearBufferSlotBytesUnchecked(buffer, slot_index);
    return true;
}

bool TryAllocateBufferSlot(HostGenotypeBuffer &buffer, std::uint32_t &slot_index);

inline NEUROEVOLUTION_HOST_DEVICE bool TryRetainBufferSlot(const GenotypeBufferView buffer,
                                                           const std::uint32_t slot_index) noexcept {
    if (!detail::IsUsableGenotypeBufferView(buffer) || (slot_index >= buffer.layout.slot_count)) {
        return false;
    }

    BufferSlotState &slot_state = buffer.slot_states[slot_index];
    return detail::AtomicTryIncrementSlotReferenceCount(&slot_state.reference_count);
}

bool TryRetainBufferSlot(HostGenotypeBuffer &buffer, std::uint32_t slot_index);

inline NEUROEVOLUTION_HOST_DEVICE bool TryReleaseBufferSlot(const GenotypeBufferView buffer,
                                                            const std::uint32_t slot_index) noexcept {
    if (!detail::IsUsableGenotypeBufferView(buffer) || (slot_index >= buffer.layout.slot_count)) {
        return false;
    }

    BufferSlotState &slot_state = buffer.slot_states[slot_index];
    std::uint32_t previous_reference_count = 0;
    if (!detail::AtomicTryDecrementSlotReferenceCount(&slot_state.reference_count, previous_reference_count)) {
        return false;
    }

    if (previous_reference_count != 1U) {
        return true;
    }

    detail::ClearBufferSlotBytesUnchecked(buffer, slot_index);
    slot_state.occupied = false;
    if (!detail::LockBufferFreeList(buffer.free_slot_lock)) {
        return false;
    }

    if (*buffer.free_slot_count >= buffer.layout.slot_count) {
        detail::UnlockBufferFreeList(buffer.free_slot_lock);
        return false;
    }

    buffer.free_slot_stack[*buffer.free_slot_count] = slot_index;
    ++(*buffer.free_slot_count);
    detail::UnlockBufferFreeList(buffer.free_slot_lock);
    return true;
}

bool TryReleaseBufferSlot(HostGenotypeBuffer &buffer, std::uint32_t slot_index);

inline NEUROEVOLUTION_HOST_DEVICE bool TryCopyGenomeBytesIntoBufferSlot(const GenotypeBufferView buffer,
                                                                        const std::uint32_t slot_index,
                                                                        const std::uint8_t *source_genome_bytes,
                                                                        const std::size_t source_bytes) noexcept {
    if (!IsValidBufferSlotIndex(buffer, slot_index) || (source_genome_bytes == nullptr) ||
        (source_bytes != buffer.layout.slot_stride_bytes) || !buffer.slot_states[slot_index].occupied) {
        return false;
    }

    CopyBufferSlotBytes(source_genome_bytes, BufferSlotBytesAt(buffer.storage, buffer.layout, slot_index),
                        source_bytes);
    return true;
}

bool TryCopyGenomeBytesIntoBufferSlot(HostGenotypeBuffer &buffer, std::uint32_t slot_index,
                                      const std::uint8_t *source_genome_bytes, std::size_t source_bytes);

inline NEUROEVOLUTION_HOST_DEVICE bool TryCloneBufferSlot(const GenotypeBufferView buffer,
                                                          const std::uint32_t source_slot_index,
                                                          std::uint32_t &cloned_slot_index) noexcept {
    if (!IsValidBufferSlotIndex(buffer, source_slot_index) || !buffer.slot_states[source_slot_index].occupied) {
        return false;
    }

    if (!TryAllocateBufferSlot(buffer, cloned_slot_index)) {
        return false;
    }

    if (!TryCopyGenomeBytesIntoBufferSlot(buffer, cloned_slot_index,
                                          BufferSlotBytesAt(buffer.storage, buffer.layout, source_slot_index),
                                          buffer.layout.slot_stride_bytes)) {
        (void)TryReleaseBufferSlot(buffer, cloned_slot_index);
        return false;
    }

    return true;
}

bool TryCloneBufferSlot(HostGenotypeBuffer &buffer, std::uint32_t source_slot_index, std::uint32_t &cloned_slot_index);

} // namespace neuroevolution::genetic_algorithm::genotype_buffer
