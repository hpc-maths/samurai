// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_base.hpp"
#include "traversers/contraction_traverser.hpp"

namespace samurai
{

    template <class Set>
    class Contraction;

    template <class Set>
    struct SetTraits<Contraction<Set>>
    {
        static_assert(IsSet<Set>::value);

        template <std::size_t d>
        using traverser_t = ContractionTraverser<typename Set::template traverser_t<d>>;

        struct Workspace
        {
            typename Set::Workspace child_workspace;
        };

        static constexpr std::size_t dim()
        {
            return Set::dim;
        }
    };

    template <class Set>
    class Contraction : public SetBase<Contraction<Set>>
    {
        using Self = Contraction<Set>;

      public:

        SAMURAI_SET_TYPEDEFS

        using contraction_t    = std::array<value_t, Base::dim>;
        using do_contraction_t = std::array<bool, Base::dim>;

        Contraction(const Set& set, const contraction_t& contraction)
            : m_set(set)
            , m_contraction(contraction)
        {
            assert(std::all_of(m_contraction.cbegin(),
                               m_contraction.cend(),
                               [](const contraction_t& c)
                               {
                                   return c >= 0;
                               }));
        }

        Contraction(const Set& set, const value_t contraction)
            : m_set(set)
        {
            assert(contraction >= 0);
            std::fill(m_contraction.begin(), m_contraction.end(), contraction);
        }

        Contraction(const Set& set, const value_t contraction, const do_contraction_t& do_contraction)
            : m_set(set)
        {
            for (std::size_t i = 0; i != m_contraction.size(); ++i)
            {
                m_contraction[i] = contraction * do_contraction[i];
            }
        }

        inline std::size_t level_impl() const
        {
            return m_set.level();
        }

        inline bool exist_impl() const
        {
            return m_set.exist();
        }

        inline bool empty_impl() const
        {
            return Base::empty_default_impl();
        }

        template <std::size_t d>
        inline void
        init_workspace_impl(const std::size_t n_traversers, std::integral_constant<std::size_t, d> d_ic, Workspace& workspace) const
        {
            m_set.init_workspace(n_traversers, d_ic, workspace.child_workspace);
        }

        template <std::size_t d>
        inline traverser_t<d>
        get_traverser_impl(const yz_index_t& index, std::integral_constant<std::size_t, d> d_ic, Workspace& workspace) const
        {
            return traverser_t<d>(m_set.get_traverser(index, d_ic, workspace.child_workspace), m_contraction[d]);
        }

        template <std::size_t d>
        inline traverser_t<d>
        get_traverser_unordered_impl(const yz_index_t& index, std::integral_constant<std::size_t, d> d_ic, Workspace& workspace) const
        {
            return traverser_t<d>(m_set.get_traverser_unordered(index, d_ic, workspace.child_workspace), m_contraction[d]);
        }

      private:

        Set m_set;
        contraction_t m_contraction;
    };

    template <class Set>
    auto contract(const Set& set, const typename Contraction<std::decay_t<decltype(self(set))>>::contraction_t& contraction)
    {
        return Contraction(self(set), contraction);
    }

    template <class Set>
    auto contract(const Set& set, const typename Contraction<std::decay_t<decltype(self(set))>>::value_t& contraction)
    {
        return Contraction(self(set), contraction);
    }

    template <class Set>
    auto contract(const Set& set,
                  const typename Contraction<std::decay_t<decltype(self(set))>>::value_t& contraction,
                  const typename Contraction<std::decay_t<decltype(self(set))>>::do_contraction_t& do_contraction) // idk how to make this
                                                                                                                   // more readable,
                                                                                                                   // perhaps a traits...
    {
        return Contraction(self(set), contraction, do_contraction);
    }

} // namespace samurai
