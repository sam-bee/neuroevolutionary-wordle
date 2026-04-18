# Cellular Genetic Algorithm Design

This document records a possible **experimental** direction for replacing the current globally mixed parent-selection
model with a **cellular genetic algorithm**.

It is a design note, not a statement of current implementation.

The current GA is still the non-spatial generational tournament-selected system described in
[`genetic-algorithm-design.md`](genetic-algorithm-design.md). This document describes a possible spatial alternative.

## Design Goal

The aim is to introduce **local mating structure** without turning the system into a model where organisms move around
continuously in space.

In this design:

- every organism occupies a fixed location in a 2-dimensional population lattice for the duration of a generation
- mating is local rather than panmictic
- offspring are written back into the same spatial population structure
- “spatial coordinates” are therefore best understood as **cell coordinates in a population grid**

This is the standard cellular-GA interpretation of spatial structure, rather than a mobile-particle or free-migration
model.

## Core Population Model

The population is arranged on a **square 2D grid**.

Each occupied grid cell contains exactly one genotype.

The initial proposal is to treat the grid as **toroidal**:

- the top wraps to the bottom
- the left wraps to the right

This avoids edge cells having systematically fewer neighbours than interior cells.

## Population Size and Grid Shape

To simplify the runtime and keep the spatial structure regular, the population size should be converted to a square
number at startup.

The intended rule is:

1. derive or request a nominal population size as usual from the runtime budgets
2. round that size **down** to the largest square number that does not exceed it
3. lay the population out as an `N x N` grid

So the cellular runtime would operate on:

- a square population
- one genotype per grid cell
- a stable lattice shape for the duration of a run, unless a later design change decides otherwise

## Spatial Coordinates

Under this design, genotypes do **not** need to carry free-moving world coordinates.

Instead:

- a genotype’s spatial location is its assigned grid cell
- the coordinate is implicit in the population layout
- the genotype does not “move” during a generation

After replacement, the next generation again occupies the same lattice positions.

So the meaningful spatial state is:

- row index
- column index

not an independently mutated or drifting `(x, y)` trait.

## Reproduction Model

The intended model is a **generational cellular GA**.

The reproductive loop for one generation would be:

1. evaluate the current grid population
2. visit each grid cell as a focal reproduction site
3. use the occupant of that focal cell as the first parent
4. choose a second parent from within a local breeding neighbourhood
5. produce one child
6. write that child into the same cell position in the next-generation grid
7. once all cells have been processed, replace the old grid with the new one

So each cell gets one reproduction event per generation, but no genotype is guaranteed to survive, because the child
for that site may differ substantially from the focal parent.

## First Parent

The recommended first draft is:

- **first parent = the genotype currently occupying the focal cell**

This is the most standard cellular-GA choice and matches the intended “fixed spatial coordinate” interpretation.

It also avoids adding a second layer of global selection pressure before locality has had any effect.

## Second Parent

The recommended first draft is:

- **second parent = a locally selected mate from within a breeding radius around the focal cell**

The recommended default local selector is:

- **roulette-wheel selection** within the local breeding neighbourhood

More concretely:

- collect the eligible neighbours inside the breeding radius
- normalize local fitness values over those neighbours
- sample the second parent from that local probability distribution

This intentionally differs from the project’s current global tournament-selection design.

### Why Tournament Instead of Roulette Wheel

The current experimental direction for this document is **local roulette-wheel selection** for the second parent,
rather than tournament selection.

## Breeding Radius

The breeding radius should be interpreted as a **local neighbourhood radius on the toroidal grid**.

The recommended first draft is:

- **radius = 2**
- **Chebyshev / Moore neighbourhood**

That means the focal cell can mate with the 24 surrounding cells in the `5 x 5` Moore window around it:

- all cells within Chebyshev distance `<= 2`
- excluding the focal cell itself

This is a sensible default because:

- it is standard in cellular-GA work
- it is still local enough to create meaningful spatial structure
- it gives a broader mating pool than radius 1 while remaining spatially constrained

If later experiments suggest the GA diffuses too slowly, the first thing to vary should be the radius.

## Child Placement

The recommended first draft is:

- **child location = the same cell as the first parent, but in the next-generation grid**

So if a child is produced at grid cell `(r, c)`, that child occupies `(r, c)` in the offspring population.

This keeps the spatial model simple:

- no explicit dispersal step
- no separate child-placement search
- no mobile organisms

It also keeps the design close to the canonical cellular-GA model.

## Replacement Style

The simplest first draft is:

- **synchronous generational replacement**

That means:

- all children are produced from the evaluated current grid
- they are written into a separate next-generation grid
- the swap happens only after the full generation is assembled

This is a good first step because it preserves the project’s existing generational character and avoids mixing local
replacement policy with the spatial-parent-selection change.

## Elitism

This experimental design should still use **no elitism**.

Spatial structure is meant to change mating and information diffusion, not to reintroduce elite carry-over.

## Fitness Range

This experimental design assumes a fitness function whose output is normalized to a bounded floating-point interval:

- `epsilon` to `1.0`

The intent is to make local roulette-wheel parent selection straightforward.

That means:

- `epsilon` is a very small positive floor above zero
- `1.0` represents the theoretical ceiling for the current evaluation scheme
- intermediate values represent normalized task performance

The normalization should be defined by dividing the raw fitness score by the maximum theoretically possible score for
the current evaluation setup, then clamping the result to a small positive minimum.

## Recommended First Draft Summary

If this feature is prototyped, the recommended first version is:

- square toroidal grid
- population size rounded down to the nearest feasible square number at startup
- one genotype per cell
- first parent is the focal cell occupant
- second parent chosen by **local roulette-wheel selection**
- breeding radius `2`
- Chebyshev / Moore neighbourhood with 24 candidate second parents
- child written to the same cell in the next-generation grid
- synchronous generational replacement
- no elitism

## Questions To Revisit Later

This first draft leaves several choices open for later discussion:

- whether radius should remain fixed or become configurable
- whether Moore neighbourhood is preferable to von Neumann / NEWS
- whether the second parent may be the focal parent itself
- what exact epsilon value should be used for the positive fitness floor
- whether local replacement should later become asynchronous
- whether child placement should remain fixed at the focal cell or later allow local dispersal

For now, the most conservative and academically standard option is the fixed-cell cellular-GA design described above.

## References

- Salto, C. and Alba, E. "Cellular Genetic Algorithms: Understanding the Behavior of Using Neighborhoods" (2019):
  https://ri.conicet.gov.ar/bitstream/11336/154756/2/CONICET_Digital_Nro.8b67ee4d-ae65-4371-9102-e9481c04ef8d_A.pdf
- Tomassini, M. *Spatially Structured Evolutionary Algorithms: Artificial Evolution in Space and Time* (2005):
  https://link.springer.com/book/10.1007/3-540-29938-6
