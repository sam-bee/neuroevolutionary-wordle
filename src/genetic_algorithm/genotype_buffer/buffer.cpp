#include "genetic_algorithm/genotype_buffer/buffer.hpp"

#include <limits>
#include <new>

namespace neuroevolution::genetic_algorithm::genotype_buffer {

void AlignedBufferStorageDeleter::operator()(std::uint8_t *pointer) const noexcept {
    if (pointer != nullptr) {
        ::operator delete[](pointer, std::align_val_t(DynamicGenomeAlignment()));
    }
}

bool TryAllocateHostBufferStorage(HostGenotypeBuffer &buffer) {
    buffer.storage.reset();
    buffer.slot_states.reset();
    buffer.free_slot_stack.reset();
    buffer.free_slot_count = 0;

    if (!IsValidGenotypeBufferLayout(buffer.layout)) {
        return false;
    }

    try {
        buffer.storage.reset(static_cast<std::uint8_t *>(
            ::operator new[](buffer.layout.buffer_bytes, std::align_val_t(DynamicGenomeAlignment()))));
    } catch (...) {
        return false;
    }

    buffer.slot_states.reset(new (std::nothrow) BufferSlotState[buffer.layout.slot_count]());
    buffer.free_slot_stack.reset(new (std::nothrow) std::uint32_t[buffer.layout.slot_count]());
    if ((buffer.slot_states == nullptr) || (buffer.free_slot_stack == nullptr)) {
        buffer = {};
        return false;
    }

    std::memset(buffer.storage.get(), 0, buffer.layout.buffer_bytes);
    for (std::size_t slot_index = 0; slot_index < buffer.layout.slot_count; ++slot_index) {
        buffer.free_slot_stack[slot_index] = static_cast<std::uint32_t>((buffer.layout.slot_count - 1) - slot_index);
    }
    buffer.free_slot_count = buffer.layout.slot_count;
    return true;
}

bool TryCreateHostGenotypeBuffer(HostGenotypeBuffer &buffer, const std::size_t slot_count,
                                 const std::size_t action_count) {
    const std::size_t slot_stride_bytes = ComputeBufferSlotStrideBytes(action_count);
    GenotypeBufferLayout layout{};
    if ((slot_count == 0) || (slot_stride_bytes == 0) ||
        (slot_count > (std::numeric_limits<std::size_t>::max() / slot_stride_bytes)) ||
        !TryCreateBufferLayoutForByteBudget(layout, slot_count * slot_stride_bytes, action_count) ||
        (layout.slot_count != slot_count)) {
        return false;
    }

    return TryCreateHostGenotypeBuffer(buffer, layout);
}

bool TryCreateHostGenotypeBufferForByteBudget(HostGenotypeBuffer &buffer, const std::size_t buffer_byte_budget_bytes,
                                              const std::size_t action_count) {
    GenotypeBufferLayout layout{};
    if (!TryCreateBufferLayoutForByteBudget(layout, buffer_byte_budget_bytes, action_count)) {
        return false;
    }

    return TryCreateHostGenotypeBuffer(buffer, layout);
}

bool TryCreateHostGenotypeBuffer(HostGenotypeBuffer &buffer, const GenotypeBufferLayout &layout) {
    buffer = {};
    buffer.layout = layout;
    if (!IsValidGenotypeBufferLayout(buffer.layout)) {
        buffer = {};
        return false;
    }

    return TryAllocateHostBufferStorage(buffer);
}

bool TryAllocateBufferSlot(HostGenotypeBuffer &buffer, std::uint32_t &slot_index) {
    return TryAllocateBufferSlot(MakeGenotypeBufferView(buffer), slot_index);
}

bool TryRetainBufferSlot(HostGenotypeBuffer &buffer, const std::uint32_t slot_index) {
    return TryRetainBufferSlot(MakeGenotypeBufferView(buffer), slot_index);
}

bool TryReleaseBufferSlot(HostGenotypeBuffer &buffer, const std::uint32_t slot_index) {
    return TryReleaseBufferSlot(MakeGenotypeBufferView(buffer), slot_index);
}

bool TryCopyGenomeBytesIntoBufferSlot(HostGenotypeBuffer &buffer, const std::uint32_t slot_index,
                                      const std::uint8_t *source_genome_bytes, const std::size_t source_bytes) {
    return TryCopyGenomeBytesIntoBufferSlot(MakeGenotypeBufferView(buffer), slot_index, source_genome_bytes,
                                            source_bytes);
}

bool TryCloneBufferSlot(HostGenotypeBuffer &buffer, const std::uint32_t source_slot_index,
                        std::uint32_t &cloned_slot_index) {
    return TryCloneBufferSlot(MakeGenotypeBufferView(buffer), source_slot_index, cloned_slot_index);
}

} // namespace neuroevolution::genetic_algorithm::genotype_buffer
