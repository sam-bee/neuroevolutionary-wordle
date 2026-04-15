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
    std::size_t free_slot_count = 0;
};

struct GenotypeBufferView {
    GenotypeBufferLayout layout{};
    std::uint8_t *storage = nullptr;
    BufferSlotState *slot_states = nullptr;
    std::uint32_t *free_slot_stack = nullptr;
    std::size_t *free_slot_count = nullptr;
};

struct ConstGenotypeBufferView {
    GenotypeBufferLayout layout{};
    const std::uint8_t *storage = nullptr;
    const BufferSlotState *slot_states = nullptr;
    const std::uint32_t *free_slot_stack = nullptr;
    const std::size_t *free_slot_count = nullptr;
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
    };
}

inline ConstGenotypeBufferView MakeConstGenotypeBufferView(const HostGenotypeBuffer &buffer) noexcept {
    return {
        .layout = buffer.layout,
        .storage = buffer.storage.get(),
        .slot_states = buffer.slot_states.get(),
        .free_slot_stack = buffer.free_slot_stack.get(),
        .free_slot_count = &buffer.free_slot_count,
    };
}

inline NEUROEVOLUTION_HOST_DEVICE std::size_t BufferFreeSlotCount(const GenotypeBufferView &buffer) noexcept {
    return (buffer.free_slot_count == nullptr) ? 0 : *buffer.free_slot_count;
}

inline NEUROEVOLUTION_HOST_DEVICE std::size_t BufferFreeSlotCount(const ConstGenotypeBufferView &buffer) noexcept {
    return (buffer.free_slot_count == nullptr) ? 0 : *buffer.free_slot_count;
}

inline NEUROEVOLUTION_HOST_DEVICE bool IsValidGenotypeBufferView(const GenotypeBufferView &buffer) noexcept {
    if (!IsValidGenotypeBufferLayout(buffer.layout) || (buffer.storage == nullptr) || (buffer.slot_states == nullptr) ||
        (buffer.free_slot_stack == nullptr) || (buffer.free_slot_count == nullptr) ||
        (BufferFreeSlotCount(buffer) > buffer.layout.slot_count)) {
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
        (BufferFreeSlotCount(buffer) > buffer.layout.slot_count)) {
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
    if (!IsValidGenotypeBufferView(buffer) || (BufferFreeSlotCount(buffer) == 0)) {
        return false;
    }

    slot_index = buffer.free_slot_stack[--(*buffer.free_slot_count)];
    BufferSlotState &slot_state = buffer.slot_states[slot_index];
    if (slot_state.occupied || (slot_state.reference_count != 0)) {
        return false;
    }

    slot_state.occupied = true;
    slot_state.reference_count = 1;
    ClearBufferSlotBytes(buffer, slot_index);
    return true;
}

bool TryAllocateBufferSlot(HostGenotypeBuffer &buffer, std::uint32_t &slot_index);

inline NEUROEVOLUTION_HOST_DEVICE bool TryRetainBufferSlot(const GenotypeBufferView buffer,
                                                           const std::uint32_t slot_index) noexcept {
    if (!IsValidBufferSlotIndex(buffer, slot_index)) {
        return false;
    }

    BufferSlotState &slot_state = buffer.slot_states[slot_index];
    if (!slot_state.occupied || (slot_state.reference_count == 0) ||
        (slot_state.reference_count == kMaxBufferSlotReferenceCount)) {
        return false;
    }

    ++slot_state.reference_count;
    return true;
}

bool TryRetainBufferSlot(HostGenotypeBuffer &buffer, std::uint32_t slot_index);

inline NEUROEVOLUTION_HOST_DEVICE bool TryReleaseBufferSlot(const GenotypeBufferView buffer,
                                                            const std::uint32_t slot_index) noexcept {
    if (!IsValidBufferSlotIndex(buffer, slot_index) || (BufferFreeSlotCount(buffer) > buffer.layout.slot_count)) {
        return false;
    }

    BufferSlotState &slot_state = buffer.slot_states[slot_index];
    if (!slot_state.occupied || (slot_state.reference_count == 0)) {
        return false;
    }

    --slot_state.reference_count;
    if (slot_state.reference_count == 0) {
        slot_state.occupied = false;
        ClearBufferSlotBytes(buffer, slot_index);
        if (BufferFreeSlotCount(buffer) >= buffer.layout.slot_count) {
            return false;
        }

        buffer.free_slot_stack[(*buffer.free_slot_count)++] = slot_index;
    }

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
