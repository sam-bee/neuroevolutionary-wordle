#include "genetic_algorithm/genotype_slab/slab_allocator.hpp"

#include <limits>
#include <new>

namespace neuroevolution::genetic_algorithm::genotype_slab {

void AlignedBufferStorageDeleter::operator()(std::uint8_t *pointer) const noexcept {
    if (pointer != nullptr) {
        ::operator delete[](pointer, std::align_val_t(DynamicGenomeAlignment()));
    }
}

bool TryAllocateHostSlabStorage(HostGenotypeSlab &buffer) {
    buffer.storage.reset();
    buffer.slot_states.reset();
    buffer.free_slot_stack.reset();
    buffer.free_slot_count = 0;
    buffer.free_slot_lock = 0;

    if (!IsValidGenotypeSlabLayout(buffer.layout)) {
        return false;
    }

    try {
        buffer.storage.reset(static_cast<std::uint8_t *>(
            ::operator new[](buffer.layout.slab_bytes, std::align_val_t(DynamicGenomeAlignment()))));
    } catch (...) {
        return false;
    }

    buffer.slot_states.reset(new (std::nothrow) SlabSlotState[buffer.layout.slot_count]());
    buffer.free_slot_stack.reset(new (std::nothrow) std::uint32_t[buffer.layout.slot_count]());
    if ((buffer.slot_states == nullptr) || (buffer.free_slot_stack == nullptr)) {
        buffer = {};
        return false;
    }

    std::memset(buffer.storage.get(), 0, buffer.layout.slab_bytes);
    for (std::size_t slot_index = 0; slot_index < buffer.layout.slot_count; ++slot_index) {
        buffer.free_slot_stack[slot_index] = static_cast<std::uint32_t>((buffer.layout.slot_count - 1) - slot_index);
    }
    buffer.free_slot_count = static_cast<std::uint32_t>(buffer.layout.slot_count);
    buffer.free_slot_lock = 0;
    return true;
}

bool TryCreateHostGenotypeSlab(HostGenotypeSlab &buffer, const std::size_t slot_count, const std::size_t action_count) {
    const std::size_t slot_stride_bytes = ComputeSlabSlotStrideBytes(action_count);
    GenotypeSlabLayout layout{};
    if ((slot_count == 0) || (slot_stride_bytes == 0) ||
        (slot_count > (std::numeric_limits<std::size_t>::max() / slot_stride_bytes)) ||
        !TryCreateSlabLayoutForByteBudget(layout, slot_count * slot_stride_bytes, action_count) ||
        (layout.slot_count != slot_count)) {
        return false;
    }

    return TryCreateHostGenotypeSlab(buffer, layout);
}

bool TryCreateHostGenotypeSlabForByteBudget(HostGenotypeSlab &buffer, const std::size_t slab_byte_budget_bytes,
                                            const std::size_t action_count) {
    GenotypeSlabLayout layout{};
    if (!TryCreateSlabLayoutForByteBudget(layout, slab_byte_budget_bytes, action_count)) {
        return false;
    }

    return TryCreateHostGenotypeSlab(buffer, layout);
}

bool TryCreateHostGenotypeSlab(HostGenotypeSlab &buffer, const GenotypeSlabLayout &layout) {
    buffer = {};
    buffer.layout = layout;
    if (!IsValidGenotypeSlabLayout(buffer.layout)) {
        buffer = {};
        return false;
    }

    return TryAllocateHostSlabStorage(buffer);
}

bool TryAllocateSlabSlot(HostGenotypeSlab &buffer, std::uint32_t &slot_index) {
    return TryAllocateSlabSlot(MakeGenotypeSlabView(buffer), slot_index);
}

bool TryRetainSlabSlot(HostGenotypeSlab &buffer, const std::uint32_t slot_index) {
    return TryRetainSlabSlot(MakeGenotypeSlabView(buffer), slot_index);
}

bool TryReleaseSlabSlot(HostGenotypeSlab &buffer, const std::uint32_t slot_index) {
    return TryReleaseSlabSlot(MakeGenotypeSlabView(buffer), slot_index);
}

bool TryCopyGenomeBytesIntoSlabSlot(HostGenotypeSlab &buffer, const std::uint32_t slot_index,
                                    const std::uint8_t *source_genome_bytes, const std::size_t source_bytes) {
    return TryCopyGenomeBytesIntoSlabSlot(MakeGenotypeSlabView(buffer), slot_index, source_genome_bytes, source_bytes);
}

bool TryCloneSlabSlot(HostGenotypeSlab &buffer, const std::uint32_t source_slot_index,
                      std::uint32_t &cloned_slot_index) {
    return TryCloneSlabSlot(MakeGenotypeSlabView(buffer), source_slot_index, cloned_slot_index);
}

} // namespace neuroevolution::genetic_algorithm::genotype_slab
