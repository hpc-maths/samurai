// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_base.hpp"

namespace samurai
{
    template <LCA_concept LCA>
    class LCAView;

    template <LCA_concept LCA>
    struct SetTraits<LCAView<LCA>>
    {
        using interval_t = typename LCA::interval_t;

        static constexpr std::size_t dim = LCA::dim;
    };

    template <LCA_concept LCA>
    class LCAView : public SetBase<LCAView<LCA>>
    {
      public:

        LCAView(const LCA& lca)
            : m_lca(lca)
        {
        }

        std::size_t min_level() const
        {
            return m_lca.level();
        }

        std::size_t level() const
        {
            return m_lca.level();
        }

        std::size_t ref_level() const
        {
            return m_lca.level();
        }

        bool exists() const
        {
            return !empty();
        }

        bool empty() const
        {
            return m_lca.empty();
        }

      private:

        LCA& m_lca;
    };

}
