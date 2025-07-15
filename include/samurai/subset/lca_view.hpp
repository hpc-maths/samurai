// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "lca_traverser.hpp"
#include "set_base.hpp"

namespace samurai
{
    template <LCA_concept LCA>
    class LCAView;

    template <LCA_concept LCA>
    struct SetTraits<LCAView<LCA>>
    {
        using traverser_t = LCATraverser<LCA>;
    };

    template <LCA_concept LCA>
    class LCAView : public SetBase<LCAView<LCA>>
    {
        using Base = SetBase<LCAView<LCA>>;

      public:

        using traverser_t = typename Base::traverser_t;

        LCAView(const LCA& lca)
            : m_lca(lca)
        {
        }

        std::size_t level() const
        {
            return m_lca.level();
        }

        bool exist() const
        {
            return !empty();
        }

        bool empty() const
        {
            return m_lca.empty();
        }

        template <class index_t, std::size_t d>
        traverser_t get_traverser(const index_t& index, std::integral_constant<std::size_t, d>) const
        {
            if constexpr (d != Base::dim - 1)
            {
                const auto& y           = index[d];
                const auto& y_intervals = m_lca[d + 1];
                const auto& y_offsets   = m_lca.offsets(d + 1);
                // we need to find an interval that contains y.
                const auto y_interval_it = std::find_if(y_intervals.cbegin(),
                                                        y_intervals.cend(),
                                                        [y](const auto& y_interval)
                                                        {
                                                            return y_interval.contains(y);
                                                        });
                if (y_interval_it != y_intervals.cend())
                {
                    const std::size_t y_offset_idx = std::size_t(y + y_interval_it->index);
                    return traverser_t(m_lca[d].cbegin() + ptrdiff_t(y_offsets[y_offset_idx]),
                                       m_lca[d].cbegin() + ptrdiff_t(y_offsets[y_offset_idx + 1]));
                }
                else
                {
                    return traverser_t(m_lca[d].cend(), m_lca[d].cend());
                }
            }
            else
            {
                return traverser_t(m_lca[d].cbegin(), m_lca[d].cend());
            }
        }

      private:

        const LCA& m_lca;
    };

    template <LCA_concept LCA>
    LCAView<LCA> self(const LCA& lca)
    {
        return LCAView<LCA>(lca);
    }

}
