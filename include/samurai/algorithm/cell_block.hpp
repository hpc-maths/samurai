// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <vector>

#include "../stencil.hpp"

// Direct walks of a level cell array inside a block of cells, without set
// expressions. They give the same cells, in the same order, as a traversal of
// the intersection of the array with the block.
namespace samurai::detail
{
    // The cells [lo[d], hi[d]) in each direction d, x first.
    template <class value_t, std::size_t dim>
    struct CellBlock
    {
        std::array<value_t, dim> lo;
        std::array<value_t, dim> hi;
    };

    // Position of the first interval of intervals[first, last) that ends after v.
    template <class Intervals, class value_t>
    std::size_t first_ending_after(const Intervals& intervals, std::size_t first, std::size_t last, value_t v)
    {
        const auto begin = intervals.begin();
        auto it          = std::partition_point(begin + static_cast<std::ptrdiff_t>(first),
                                       begin + static_cast<std::ptrdiff_t>(last),
                                       [v](const auto& interval)
                                       {
                                           return interval.end <= v;
                                       });
        return static_cast<std::size_t>(it - begin);
    }

    template <std::size_t d, class LCA, class Block, class Coords, class Func>
    void for_each_interval_in_block_rec(const LCA& lca, const Block& block, std::size_t first, std::size_t last, Coords& coords, Func& f)
    {
        const auto& intervals = lca[d];
        for (auto k = first_ending_after(intervals, first, last, block.lo[d]); k < last && intervals[k].start < block.hi[d]; ++k)
        {
            const auto& interval = intervals[k];
            const auto start     = std::max(interval.start, block.lo[d]);
            const auto end       = std::min(interval.end, block.hi[d]);
            if constexpr (d == 0)
            {
                coords[0] = start;
                f(start, end, coords, interval.index);
            }
            else
            {
                for (auto v = start; v < end; ++v)
                {
                    coords[d]       = v;
                    const auto slot = static_cast<std::size_t>(interval.index + v);
                    for_each_interval_in_block_rec<d - 1>(lca, block, lca.offsets(d)[slot], lca.offsets(d)[slot + 1], coords, f);
                }
            }
        }
    }

    // Calls f(start, end, coords, shift) for each interval [start, end) of lca
    // inside block, in traversal order: coords holds the cell (start, y, z) and
    // the storage index of the cell x is x + shift.
    template <class LCA, class Block, class Func>
    void for_each_interval_in_block(const LCA& lca, const Block& block, Func&& f)
    {
        constexpr std::size_t dim = LCA::dim;
        std::array<typename LCA::value_t, dim> coords{};
        for_each_interval_in_block_rec<dim - 1>(lca, block, 0, lca[dim - 1].size(), coords, f);
    }

    // Storage index of the cell `coords` of lca, -1 when lca does not hold it.
    template <class LCA>
    auto cell_index(const LCA& lca, const std::array<typename LCA::value_t, LCA::dim>& coords) -> typename LCA::index_t
    {
        std::size_t first = 0;
        std::size_t last  = lca[LCA::dim - 1].size();
        for (std::size_t d = LCA::dim - 1;; --d)
        {
            const auto& intervals = lca[d];
            const auto k          = first_ending_after(intervals, first, last, coords[d]);
            if (k == last || intervals[k].start > coords[d])
            {
                return -1;
            }
            const auto slot = intervals[k].index + coords[d];
            if (d == 0)
            {
                return slot;
            }
            first = lca.offsets(d)[static_cast<std::size_t>(slot)];
            last  = lca.offsets(d)[static_cast<std::size_t>(slot) + 1];
        }
    }

    // Storage indices of the cells of one row of lca, for lookups at
    // increasing or close x: the last interval found is tried first.
    template <class LCA>
    class RowLookup
    {
      public:

        using value_t = typename LCA::value_t;
        using index_t = typename LCA::index_t;

        // The row of `coords`, coords[0] being ignored.
        RowLookup(const LCA& lca, const std::array<value_t, LCA::dim>& coords)
            : m_intervals(&lca[0])
        {
            m_last = lca[LCA::dim - 1].size();
            for (std::size_t d = LCA::dim - 1; d > 0; --d)
            {
                const auto& intervals = lca[d];
                const auto k          = first_ending_after(intervals, m_first, m_last, coords[d]);
                if (k == m_last || intervals[k].start > coords[d])
                {
                    m_first = m_last = 0;
                    break;
                }
                const auto slot = static_cast<std::size_t>(intervals[k].index + coords[d]);
                m_first         = lca.offsets(d)[slot];
                m_last          = lca.offsets(d)[slot + 1];
            }
            m_hint = m_first;
        }

        // Storage index of the cell x of the row, -1 when it is not there.
        index_t operator()(value_t x)
        {
            const auto& intervals = *m_intervals;
            if (m_hint >= m_last || !intervals[m_hint].contains(x))
            {
                m_hint = first_ending_after(intervals, m_first, m_last, x);
                if (m_hint == m_last || intervals[m_hint].start > x)
                {
                    return -1;
                }
            }
            return intervals[m_hint].index + x;
        }

      private:

        const std::vector<typename LCA::interval_t>* m_intervals;
        std::size_t m_first = 0;
        std::size_t m_last  = 0;
        std::size_t m_hint  = 0;
    };

    // The cells of a box-shaped lca as a block, nothing when lca is empty or not
    // a box.
    template <class LCA>
    auto as_block(const LCA& lca) -> std::optional<CellBlock<typename LCA::value_t, LCA::dim>>
    {
        constexpr std::size_t dim = LCA::dim;
        if (lca.empty())
        {
            return std::nullopt;
        }
        CellBlock<typename LCA::value_t, dim> block;
        std::size_t volume = 1;
        for (std::size_t d = 0; d < dim; ++d)
        {
            block.lo[d] = lca[d].front().start;
            block.hi[d] = lca[d].front().end;
            for (const auto& interval : lca[d])
            {
                block.lo[d] = std::min(block.lo[d], interval.start);
                block.hi[d] = std::max(block.hi[d], interval.end);
            }
            volume *= static_cast<std::size_t>(block.hi[d] - block.lo[d]);
        }
        if (lca.nb_cells() != volume)
        {
            return std::nullopt;
        }
        return block;
    }

    // Whether lca lies in block.
    template <class LCA, class Block>
    bool block_holds(const Block& block, const LCA& lca)
    {
        for (std::size_t d = 0; d < LCA::dim; ++d)
        {
            const bool inside = std::all_of(lca[d].begin(),
                                            lca[d].end(),
                                            [&](const auto& interval)
                                            {
                                                return interval.start >= block.lo[d] && interval.end <= block.hi[d];
                                            });
            if (!inside)
            {
                return false;
            }
        }
        return true;
    }

    // The block at a level delta_l coarser: the cells that cover it.
    template <class value_t, std::size_t dim>
    CellBlock<value_t, dim> coarsen(CellBlock<value_t, dim> block, std::size_t delta_l)
    {
        for (std::size_t d = 0; d < dim; ++d)
        {
            block.lo[d] = block.lo[d] >> delta_l;
            block.hi[d] = ((block.hi[d] - 1) >> delta_l) + 1;
        }
        return block;
    }

    // The block at a level delta_l finer: the children of its cells.
    template <class value_t, std::size_t dim>
    CellBlock<value_t, dim> refine(CellBlock<value_t, dim> block, std::size_t delta_l)
    {
        for (std::size_t d = 0; d < dim; ++d)
        {
            block.lo[d] = block.lo[d] << delta_l;
            block.hi[d] = block.hi[d] << delta_l;
        }
        return block;
    }

    template <class value_t, std::size_t dim>
    CellBlock<value_t, dim> translate(CellBlock<value_t, dim> block, const DirectionVector<dim>& t)
    {
        for (std::size_t d = 0; d < dim; ++d)
        {
            block.lo[d] += t[d];
            block.hi[d] += t[d];
        }
        return block;
    }

    // The cells of block in its layer `layer` along `direction`, a cartesian
    // direction: its cells on the face when layer is 0, outside it after.
    template <class value_t, std::size_t dim>
    CellBlock<value_t, dim> face_layer(CellBlock<value_t, dim> block, const DirectionVector<dim>& direction, int layer)
    {
        for (std::size_t d = 0; d < dim; ++d)
        {
            if (direction[d] > 0)
            {
                block.lo[d] = block.hi[d] - 1 + static_cast<value_t>(layer);
            }
            else if (direction[d] < 0)
            {
                block.lo[d] = block.lo[d] - static_cast<value_t>(layer);
            }
            if (direction[d] != 0)
            {
                block.hi[d] = block.lo[d] + 1;
            }
        }
        return block;
    }

    template <class value_t, std::size_t dim>
    CellBlock<value_t, dim> intersect(CellBlock<value_t, dim> a, const CellBlock<value_t, dim>& b)
    {
        for (std::size_t d = 0; d < dim; ++d)
        {
            a.lo[d] = std::max(a.lo[d], b.lo[d]);
            a.hi[d] = std::min(a.hi[d], b.hi[d]);
        }
        return a;
    }

    // The cells [start, end) of the row of `coords`, coords[0] being ignored.
    template <class value_t, std::size_t dim>
    CellBlock<value_t, dim> row_block(const std::array<value_t, dim>& coords, value_t start, value_t end)
    {
        CellBlock<value_t, dim> block{coords, coords};
        block.lo[0] = start;
        block.hi[0] = end;
        for (std::size_t d = 1; d < dim; ++d)
        {
            ++block.hi[d];
        }
        return block;
    }

    // The block of the single cell `coords`.
    template <class value_t, std::size_t dim>
    CellBlock<value_t, dim> cell_block(const std::array<value_t, dim>& coords)
    {
        CellBlock<value_t, dim> block{coords, coords};
        std::transform(coords.begin(),
                       coords.end(),
                       block.hi.begin(),
                       [](value_t c)
                       {
                           return c + 1;
                       });
        return block;
    }

    template <class value_t, std::size_t dim>
    std::array<value_t, dim> translate(std::array<value_t, dim> coords, const DirectionVector<dim>& t)
    {
        for (std::size_t d = 0; d < dim; ++d)
        {
            coords[d] += t[d];
        }
        return coords;
    }

    // self(mesh.corner(direction)).on(level) as a block, nothing when the
    // corner is not a box.
    template <class Mesh>
    auto corner_block(const Mesh& mesh,
                      const DirectionVector<Mesh::dim>& direction,
                      std::size_t level) -> std::optional<CellBlock<typename Mesh::interval_t::value_t, Mesh::dim>>
    {
        const auto& corner = mesh.corner(direction);
        auto block         = as_block(corner);
        if (!block || corner.level() < level)
        {
            return std::nullopt;
        }
        return coarsen(*block, corner.level() - level);
    }
}
