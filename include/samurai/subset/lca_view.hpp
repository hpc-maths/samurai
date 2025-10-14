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

        using const_interval_iterator = typename std::vector<interval_t>::const_iterator;

        explicit LCAView(const LCA& lca)
            : m_lca(lca)
        {
        }

        LCAView(const LCAView& other)
            : m_lca(other.m_lca)
        {
        }

        LCAView(LCAView&& other)
            : m_lca(std::move(other.m_lca))
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
        inline constexpr void init_workspace_impl(const std::size_t, std::integral_constant<std::size_t, d>, Workspace) const
        {
        }

        template <class index_t, std::size_t d>
        inline traverser_t<d> get_traverser_impl(const index_t& index, std::integral_constant<std::size_t, d> d_ic, Workspace) const
        {
            return get_traverser_impl_detail(index,
                                             m_lca[Base::dim - 1].cbegin(),
                                             m_lca[Base::dim - 1].cend(),
                                             d_ic,
                                             std::integral_constant<std::size_t, Base::dim - 1>{});
        }

      private:

        template <class index_t, std::size_t d, std::size_t dCur>
        inline traverser_t<d> get_traverser_impl_detail(const index_t& index,
                                                        const_interval_iterator begin_y_interval,
                                                        const_interval_iterator end_y_interval,
                                                        std::integral_constant<std::size_t, d> d_ic,
                                                        std::integral_constant<std::size_t, dCur>) const
        {
            if constexpr (dCur != Base::dim - 1)
            {
                const auto& y         = index[dCur];
                const auto& y_offsets = m_lca.offsets(dCur + 1);
                // we need to find an interval that contains y.
                const auto y_interval_it = std::find_if(begin_y_interval,
                                                        end_y_interval,
                                                        [y](const auto& y_interval)
                                                        {
                                                            return y_interval.contains(y);
                                                        });
                if (y_interval_it != end_y_interval)
                {
                    const std::size_t y_offset_idx = std::size_t(y + y_interval_it->index);

                    const_interval_iterator begin_x_interval = m_lca[dCur].cbegin() + ptrdiff_t(y_offsets[y_offset_idx]);
                    const_interval_iterator end_x_interval   = m_lca[dCur].cbegin() + ptrdiff_t(y_offsets[y_offset_idx + 1]);

                    if constexpr (d == dCur)
                    {
                        return traverser_t<d>(begin_x_interval, end_x_interval);
                    }
                    else
                    {
                        return get_traverser_impl_detail(index,
                                                         begin_x_interval,
                                                         end_x_interval,
                                                         d_ic,
                                                         std::integral_constant<std::size_t, dCur - 1>{});
                    }
                }
                else
                {
                    return traverser_t<d>(m_lca[d].cend(), m_lca[d].cend());
                }
            }
            else if constexpr (d != dCur)
            {
                return get_traverser_impl_detail(index,
                                                 m_lca[dCur].cbegin(),
                                                 m_lca[dCur].cend(),
                                                 d_ic,
                                                 std::integral_constant<std::size_t, dCur - 1>{});
            }
            else
            {
                return traverser_t<d>(m_lca[dCur].cbegin(), m_lca[dCur].cend());
            }
        }

        const LCA& m_lca;
    };

    template <std::size_t Dim, class TInterval>
    LCAView<LevelCellArray<Dim, TInterval>> self(const LevelCellArray<Dim, TInterval>& lca)
    {
        return LCAView<LevelCellArray<Dim, TInterval>>(lca);
    }

} // namespace samurai
