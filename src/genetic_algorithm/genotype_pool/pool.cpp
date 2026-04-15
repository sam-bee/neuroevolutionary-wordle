#include "genetic_algorithm/genotype_pool/pool.hpp"

#include <limits>
#include <new>

namespace neuroevolution::genetic_algorithm::genotype_pool {

void AlignedPoolStorageDeleter::operator()(std::uint8_t *pointer) const noexcept {
    if (pointer != nullptr) {
        ::operator delete[](pointer, std::align_val_t(DynamicGenomeAlignment()));
    }
}

bool TryAllocateHostPoolStorage(HostGenotypePool &pool) {
    pool.storage.reset();
    pool.slot_states.reset();
    pool.free_slot_stack.reset();
    pool.free_slot_count = 0;

    if (!IsValidGenotypePoolLayout(pool.layout)) {
        return false;
    }

    try {
        pool.storage.reset(static_cast<std::uint8_t *>(
            ::operator new[](pool.layout.pool_bytes, std::align_val_t(DynamicGenomeAlignment()))));
    } catch (...) {
        return false;
    }

    pool.slot_states.reset(new (std::nothrow) PoolSlotState[pool.layout.slot_count]());
    pool.free_slot_stack.reset(new (std::nothrow) std::uint32_t[pool.layout.slot_count]());
    if ((pool.slot_states == nullptr) || (pool.free_slot_stack == nullptr)) {
        pool = {};
        return false;
    }

    std::memset(pool.storage.get(), 0, pool.layout.pool_bytes);
    for (std::size_t slot_index = 0; slot_index < pool.layout.slot_count; ++slot_index) {
        pool.free_slot_stack[slot_index] = static_cast<std::uint32_t>((pool.layout.slot_count - 1) - slot_index);
    }
    pool.free_slot_count = pool.layout.slot_count;
    return true;
}

bool TryCreateHostGenotypePool(HostGenotypePool &pool, const std::size_t slot_count, const std::size_t action_count) {
    const std::size_t slot_stride_bytes = ComputePoolSlotStrideBytes(action_count);
    GenotypePoolLayout layout{};
    if ((slot_count == 0) || (slot_stride_bytes == 0) ||
        (slot_count > (std::numeric_limits<std::size_t>::max() / slot_stride_bytes)) ||
        !TryCreatePoolLayoutForByteBudget(layout, slot_count * slot_stride_bytes, action_count) ||
        (layout.slot_count != slot_count)) {
        return false;
    }

    return TryCreateHostGenotypePool(pool, layout);
}

bool TryCreateHostGenotypePoolForByteBudget(HostGenotypePool &pool, const std::size_t pool_byte_budget_bytes,
                                            const std::size_t action_count) {
    GenotypePoolLayout layout{};
    if (!TryCreatePoolLayoutForByteBudget(layout, pool_byte_budget_bytes, action_count)) {
        return false;
    }

    return TryCreateHostGenotypePool(pool, layout);
}

bool TryCreateHostGenotypePool(HostGenotypePool &pool, const GenotypePoolLayout &layout) {
    pool = {};
    pool.layout = layout;
    if (!IsValidGenotypePoolLayout(pool.layout)) {
        pool = {};
        return false;
    }

    return TryAllocateHostPoolStorage(pool);
}

bool TryAllocatePoolSlot(HostGenotypePool &pool, std::uint32_t &slot_index) {
    return TryAllocatePoolSlot(MakeGenotypePoolView(pool), slot_index);
}

bool TryRetainPoolSlot(HostGenotypePool &pool, const std::uint32_t slot_index) {
    return TryRetainPoolSlot(MakeGenotypePoolView(pool), slot_index);
}

bool TryReleasePoolSlot(HostGenotypePool &pool, const std::uint32_t slot_index) {
    return TryReleasePoolSlot(MakeGenotypePoolView(pool), slot_index);
}

bool TryCopyGenomeBytesIntoPoolSlot(HostGenotypePool &pool, const std::uint32_t slot_index,
                                    const std::uint8_t *source_genome_bytes, const std::size_t source_bytes) {
    return TryCopyGenomeBytesIntoPoolSlot(MakeGenotypePoolView(pool), slot_index, source_genome_bytes, source_bytes);
}

bool TryClonePoolSlot(HostGenotypePool &pool, const std::uint32_t source_slot_index, std::uint32_t &cloned_slot_index) {
    return TryClonePoolSlot(MakeGenotypePoolView(pool), source_slot_index, cloned_slot_index);
}

} // namespace neuroevolution::genetic_algorithm::genotype_pool
