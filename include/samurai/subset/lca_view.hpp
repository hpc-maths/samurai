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
            Workspace()
            {
                start_offset_guess.fill(0);
            }

            // we do not need the offsets for the last dim
            // the offset at dimension d will be initialized when calling
            // get_traverser_impl<d+1>
            std::array<std::ptrdiff_t, LCA::dim - 1> start_offset;
            std::array<std::ptrdiff_t, LCA::dim - 1> end_offset;

            std::array<std::ptrdiff_t, LCA::dim - 1> start_offset_guess;
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

        using const_interval_iterator = typename std::vector<typename LCA::interval_t>::const_iterator;

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
                    y_end_offset   = 0; // to avoid Conditional jump or move depends on uninitialised value(s)
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

        template <std::size_t d>
        inline traverser_t<d>
        get_traverser_unordered_impl(const yz_index_t& index, std::integral_constant<std::size_t, d>, Workspace& workspace) const
        {
            ptrdiff_t start_offset = 0;
            ptrdiff_t end_offset   = std::ssize(m_lca[Base::dim - 1]);

            for (std::size_t dCur = Base::dim - 1; dCur != d; --dCur)
            {
                const auto y             = index[dCur - 1];
                const auto& y_intervals  = m_lca[dCur];
                auto& start_offset_guess = workspace.start_offset_guess[dCur - 1];

                const auto begin_y_intervals = y_intervals.cbegin() + start_offset;
                const auto end_y_intervals   = y_intervals.cbegin() + end_offset;
                const auto y_intervals_size  = std::distance(begin_y_intervals, end_y_intervals);
                // if guess was wrong for the higer dimensions, the guess is wrong an may even be out of [begin_y_intervals,
                // end_y_intervals) we thus need to cap start_offset to ensure begin_y_intervals <= begin_y_intervals_guess <=
                // end_y_intervals
                const auto begin_y_intervals_guess = begin_y_intervals + std::min(start_offset, y_intervals_size);

                assert(begin_y_intervals <= begin_y_intervals_guess and begin_y_intervals_guess <= end_y_intervals);

                // we know the interval that contains y is likely to be in the range [begin_y_intervals_guess, end_y_intervals)
                // first we try to find it within [begin_y_intervals_guess, end_y_intervals)
                // hopefully, *begin_y_intervals_guess contains y.
                auto y_interval_it = std::find_if(begin_y_intervals_guess,
                                                  end_y_intervals,
                                                  [y](const auto& y_interval)
                                                  {
                                                      return y_interval.contains(y);
                                                  });
                if (y_interval_it == end_y_intervals)
                {
                    // we did not find an interval that contains y in [begin_y_intervals_guess, end_y_intervals)
                    // we try to find it in [begin_y_intervals, begin_y_intervals_guess)
                    y_interval_it = std::find_if(begin_y_intervals,
                                                 begin_y_intervals_guess,
                                                 [y](const auto& y_interval)
                                                 {
                                                     return y_interval.contains(y);
                                                 });
                    if (y_interval_it == begin_y_intervals_guess)
                    {
                        // there is no interval that contains y
                        return traverser_t<d>(m_lca[d].cend(), m_lca[d].cend());
                    }
                }
                start_offset_guess = std::distance(begin_y_intervals, y_interval_it);
                assert(0 <= start_offset_guess and start_offset_guess < std::distance(begin_y_intervals, end_y_intervals));

                const auto& y_offsets   = m_lca.offsets(dCur);
                const auto y_offset_idx = std::size_t(y + y_interval_it->index);

                start_offset = ptrdiff_t(y_offsets[y_offset_idx]);
                end_offset   = ptrdiff_t(y_offsets[y_offset_idx + 1]);
            }
            return traverser_t<d>(m_lca[d].cbegin() + start_offset, m_lca[d].cbegin() + end_offset);
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
