// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_base.hpp"
#include "traversers/expansion_traverser.hpp"
#include "traversers/last_dim_expansion_traverser.hpp"

namespace samurai
{

    namespace detail
    {
        template <class Set, typename Seq>
        struct ExpansionWorkspace;

        template <class Set, std::size_t... ds>
        struct ExpansionWorkspace<Set, std::index_sequence<ds...>>
        {
            template <std::size_t d>
            using child_traverser_t = typename Set::template traverser_t<d>;

            template <std::size_t d>
            using array_of_child_traverser_t = std::vector<child_traverser_t<d>>;

            using Type = std::tuple<array_of_child_traverser_t<ds>...>;

            static_assert(std::tuple_size<Type>::value == Set::dim - 1);
        };

    } // namespace detail

    template <class Set>
    class Expansion;

    template <class Set>
    struct SetTraits<Expansion<Set>>
    {
        static_assert(IsSet<Set>::value);

        template <std::size_t d>
        using traverser_t = std::conditional_t<d == Set::dim - 1,
                                               LastDimExpansionTraverser<typename Set::template traverser_t<d>>,
                                               ExpansionTraverser<typename Set::template traverser_t<d>>>;

        struct Workspace
        {
            typename detail::ExpansionWorkspace<Set, std::make_index_sequence<Set::dim - 1>>::Type expansion_workspace;
            typename Set::Workspace child_workspace;
        };

        static constexpr std::size_t dim()
        {
            return Set::dim;
        }
    };

    template <class Set>
    class Expansion : public SetBase<Expansion<Set>>
    {
        using Self = Expansion<Set>;

      public:

        SAMURAI_SET_TYPEDEFS

        using expansion_t    = std::array<value_t, Base::dim>;
        using do_expansion_t = std::array<bool, Base::dim>;

        Expansion(const Set& set, const expansion_t& expansions)
            : m_set(set)
            , m_expansions(expansions)
        {
        }

        Expansion(const Set& set, const value_t expansion)
            : m_set(set)
        {
            m_expansions.fill(expansion);
        }

        Expansion(const Set& set, const value_t expansion, const do_expansion_t& do_expansion)
            : m_set(set)
        {
            for (std::size_t i = 0; i != m_expansions.size(); ++i)
            {
                m_expansions[i] = expansion * do_expansion[i];
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
            return m_set.empty();
        }

        template <std::size_t d>
        inline void
        init_workspace_impl(const std::size_t n_traversers, std::integral_constant<std::size_t, d> d_ic, Workspace& workspace) const
        {
            if constexpr (d == Base::dim - 1)
            {
                m_set.init_workspace(n_traversers, d_ic, workspace.child_workspace);
            }
            else
            {
                const std::size_t my_work_size = n_traversers * 2 * std::size_t(m_expansions[d + 1] + 1);

                auto& childTraversers = std::get<d>(workspace.expansion_workspace);
                childTraversers.clear();
                childTraversers.reserve(my_work_size);

                m_set.init_workspace(my_work_size, d_ic, workspace.child_workspace);
            }
        }

        template <std::size_t d>
        inline traverser_t<d>
        get_traverser_impl(const yz_index_t& index, std::integral_constant<std::size_t, d> d_ic, Workspace& workspace) const
        {
            if constexpr (d == Base::dim - 1)
            {
                return traverser_t<d>(m_set.get_traverser(index, d_ic, workspace.child_workspace), m_expansions[d]);
            }
            else
            {
                auto& childTraversers = std::get<d>(workspace.expansion_workspace);

                yz_index_t tmp_index(index);

                const auto childTraversers_begin = childTraversers.end();

                for (value_t width = 0; width != m_expansions[d + 1] + 1; ++width)
                {
                    // it's d and not d+1 because tmp_index represents m_expansions[1,..,dim]
                    tmp_index[d] = index[d] + width;
                    childTraversers.push_back(m_set.get_traverser_unordered(tmp_index, d_ic, workspace.child_workspace));
                    if (childTraversers.back().is_empty())
                    {
                        childTraversers.pop_back();
                    }
                    // same
                    tmp_index[d] = index[d] - width;
                    childTraversers.push_back(m_set.get_traverser_unordered(tmp_index, d_ic, workspace.child_workspace));
                    if (childTraversers.back().is_empty())
                    {
                        childTraversers.pop_back();
                    }
                }

                return traverser_t<d>(childTraversers_begin, childTraversers.end(), m_expansions[d]);
            }
        }

        template <std::size_t d>
        inline traverser_t<d>
        get_traverser_unordered_impl(const yz_index_t& index, std::integral_constant<std::size_t, d> d_ic, Workspace& workspace) const
        {
            if constexpr (d == Base::dim - 1)
            {
                return traverser_t<d>(m_set.get_traverser_unordered(index, d_ic, workspace.child_workspace), m_expansions[d]);
            }
            else
            {
                return get_traverser_impl(index, d_ic, workspace);
            }
        }

      private:

        Set m_set;
        expansion_t m_expansions;
    };

    template <class Set>
    auto expand(const Set& set, const typename Contraction<std::decay_t<decltype(self(set))>>::contraction_t& expansions)
    {
        return Expansion(self(set), expansions);
    }

    template <class Set>
    auto expand(const Set& set, const typename Contraction<std::decay_t<decltype(self(set))>>::value_t& expansion)
    {
        return Expansion(self(set), expansion);
    }

    template <class Set>
    auto expand(const Set& set,
                const typename Contraction<std::decay_t<decltype(self(set))>>::value_t& expansion,
                const typename Contraction<std::decay_t<decltype(self(set))>>::do_expansion_t& do_expansion)
    {
        return Expansion(self(set), expansion, do_expansion);
    }

} // namespace samurai
