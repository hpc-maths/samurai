// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_base.hpp"
#include "traversers/lca_traverser.hpp"

namespace samurai
{

    template <class LCA>
    class LCAView;

    template <class LCA>
    struct SetTraits<LCAView<LCA>>
    {
        static_assert(std::same_as<LevelCellArray<LCA::dim, typename LCA::interval_t>, LCA>);

        template <std::size_t>
        using traverser_t = LCATraverser<LCA>;

        struct Workspace
        {
            // we do not need the offsets for the last dim
            // the offset at dimension d will be initialized when calling
            // get_traverser_impl<d+1>
            std::array<std::ptrdiff_t, LCA::dim - 1> start_offset;
            std::array<std::ptrdiff_t, LCA::dim - 1> end_offset;
        };

        static constexpr std::size_t dim()
        {
            return LCA::dim;
        }
    };

    template <class LCA>
    class LCAView : public SetBase<LCAView<LCA>>
    {
        using Self = LCAView<LCA>;

      public:

        SAMURAI_SET_TYPEDEFS

        explicit LCAView(const LCA& lca)
            : m_lca(lca)
        {
        }

        inline std::size_t level_impl() const
        {
            return m_lca.level();
        }

        inline bool exist_impl() const
        {
            return !empty_impl();
        }

        inline bool empty_impl() const
        {
            return m_lca.empty();
        }

        template <std::size_t d>
        inline constexpr void init_workspace_impl(const std::size_t, std::integral_constant<std::size_t, d>, Workspace&) const
        {
        }

        template <std::size_t d>
        inline traverser_t<d> get_traverser_impl(const yz_index_t& index, std::integral_constant<std::size_t, d>, Workspace& workspace) const
        {
            if constexpr (d == Base::dim - 1)
            {
                return traverser_t<d>(m_lca[d].cbegin(), m_lca[d].cend());
            }
            else
            {
                // In 3d, we would be in the y dimension
                // we need to find an interval that contains the prescibed z.
                const auto& z_intervals     = m_lca[d + 1];
                const auto begin_z_interval = (d == Base::dim - 2) ? z_intervals.cbegin()
                                                                   : z_intervals.cbegin() + workspace.start_offset[d + 1];
                const auto end_z_interval = (d == Base::dim - 2) ? z_intervals.cend() : z_intervals.cbegin() + workspace.end_offset[d + 1];

                const auto z = index[d];

                const auto z_interval_it = std::find_if(begin_z_interval,
                                                        end_z_interval,
                                                        [z](const auto& z_interval)
                                                        {
                                                            return z_interval.contains(z);
                                                        });

                const auto& y_intervals = m_lca[d];

                auto& y_start_offset = workspace.start_offset[d];
                auto& y_end_offset   = workspace.end_offset[d];

                if (z_interval_it == end_z_interval)
                {
                    y_start_offset = y_end_offset;
                    return traverser_t<d>(y_intervals.cend(), y_intervals.cend());
                }
                const auto& y_offsets   = m_lca.offsets(d + 1);
                const auto y_offset_idx = std::size_t(z_interval_it->index + z);

                y_start_offset = std::ptrdiff_t(y_offsets[y_offset_idx]);
                y_end_offset   = std::ptrdiff_t(y_offsets[y_offset_idx + 1]);
                return traverser_t<d>(y_intervals.cbegin() + y_start_offset, y_intervals.cbegin() + y_end_offset);
            }
        }

      private:

        const LCA& m_lca;
    };

    template <std::size_t Dim, class TInterval>
    LCAView<LevelCellArray<Dim, TInterval>> self(const LevelCellArray<Dim, TInterval>& lca)
    {
        return LCAView<LevelCellArray<Dim, TInterval>>(lca);
    }

} // namespace samurai
