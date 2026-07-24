// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "../../box.hpp"
#include "../../static_algorithm.hpp"
#include "../node.hpp" // the static set DSL (self, intersection, translate, ...)
#include "dynamic_set.hpp"

namespace samurai
{
    namespace detail
    {
        /**
         * Turn a runtime dimension `target` into the compile-time
         * `std::integral_constant` expected by the static algebra, and invoke
         * `f` with it. All `dim` branches return the same type, so this works
         * both for value-returning calls (get_traverser) and void ones
         * (init_workspace).
         */
        template <std::size_t dim, std::size_t d = 0, class Func>
        SAMURAI_INLINE decltype(auto) static_switch(std::size_t target, Func&& f)
        {
            if constexpr (d + 1 < dim)
            {
                if (target == d)
                {
                    return f(std::integral_constant<std::size_t, d>{});
                }
                return static_switch<dim, d + 1>(target, std::forward<Func>(f));
            }
            else
            {
                assert(target == d && "dimension out of range");
                return f(std::integral_constant<std::size_t, d>{});
            }
        }
    } // namespace detail

    ////////////////////////////////////////////////////////////////////////
    //// ExecNode: drives a static set behind the runtime ISet interface
    ////////////////////////////////////////////////////////////////////////

    /**
     * Runs a concrete static set `StaticSet` behind the virtual `ISet`
     * interface. It owns the set and its traversal scratch (`Workspace`), and
     * forwards each runtime call to the matching compile-time one via
     * `static_switch`. `clone` stays abstract: only a concrete node knows how
     * to deep-copy the dynamic children the static set was built from.
     */
    template <class StaticSet>
    class ExecNode : public ISet<StaticSet::dim, typename StaticSet::interval_t>
    {
        using Base                       = ISet<StaticSet::dim, typename StaticSet::interval_t>;
        static constexpr std::size_t dim = StaticSet::dim;

      public:

        using typename Base::traverser_t;
        using typename Base::yz_index_t;

        std::size_t level() const override
        {
            return m_exec.level();
        }

        bool exist() const override
        {
            return m_exec.exist();
        }

        bool empty() const override
        {
            return m_exec.empty();
        }

        void init_workspace(std::size_t d, std::size_t n_traversers) override
        {
            detail::static_switch<dim>(d,
                                       [&](auto d_ic)
                                       {
                                           m_exec.init_workspace(n_traversers, d_ic, m_ws);
                                       });
        }

        traverser_t get_traverser(std::size_t d, const yz_index_t& index) override
        {
            return detail::static_switch<dim>(d,
                                              [&](auto d_ic)
                                              {
                                                  return make_any_traverser(m_exec.get_traverser(index, d_ic, m_ws));
                                              });
        }

        traverser_t get_traverser_unordered(std::size_t d, const yz_index_t& index) override
        {
            return detail::static_switch<dim>(d,
                                              [&](auto d_ic)
                                              {
                                                  return make_any_traverser(m_exec.get_traverser_unordered(index, d_ic, m_ws));
                                              });
        }

      protected:

        explicit ExecNode(StaticSet exec)
            : m_exec(std::move(exec))
        {
        }

      private:

        StaticSet m_exec;
        typename StaticSet::Workspace m_ws;
    };

    ////////////////////////////////////////////////////////////////////////
    //// Node: a static set + the dynamic children it was built from
    ////////////////////////////////////////////////////////////////////////

    /**
     * The single concrete `ISet` node. It stores:
     *   - the dynamic children (`shared_ptr<ISet>`), which it owns for cloning;
     *   - a `rebuild` functor that turns those children into the static set
     *     actually executed.
     *
     * A leaf passes no children and a `rebuild` that captures the data (e.g. an
     * LCA); a modifier passes one child; a binary operator passes two. `clone`
     * is uniform: deep-copy the children, then rebuild.
     */
    template <std::size_t dim_,
              class TInterval,
              class Rebuild,
              class StaticSet = std::invoke_result_t<Rebuild&, const std::vector<std::shared_ptr<ISet<dim_, TInterval>>>&>>
    class Node final : public ExecNode<StaticSet>
    {
        using iset_t    = ISet<dim_, TInterval>;
        using child_vec = std::vector<std::shared_ptr<iset_t>>;

        static_assert(StaticSet::dim == dim_);

      public:

        Node(child_vec children, Rebuild rebuild)
            : ExecNode<StaticSet>(rebuild(children))
            , m_children(std::move(children))
            , m_rebuild(std::move(rebuild))
        {
        }

        std::shared_ptr<iset_t> clone() const override
        {
            child_vec cloned;
            cloned.reserve(m_children.size());
            for (const auto& child : m_children)
            {
                cloned.push_back(child->clone());
            }
            return std::make_shared<Node>(std::move(cloned), m_rebuild);
        }

      private:

        child_vec m_children;
        Rebuild m_rebuild;
    };

    ////////////////////////////////////////////////////////////////////////
    //// DynamicSet: the user-facing handle
    ////////////////////////////////////////////////////////////////////////

    /**
     * A cheap value handle over a runtime set expression. Mirrors the static
     * set surface (`on`, `operator()`, `to_lca`, `level`) and is the natural
     * type to expose to bindings.
     */
    template <std::size_t dim_, class TInterval>
    class DynamicSet
    {
      public:

        static constexpr std::size_t dim = dim_;

        using interval_t = TInterval;
        using iset_t     = ISet<dim_, TInterval>;

        explicit DynamicSet(std::shared_ptr<iset_t> set)
            : m_set(std::move(set))
        {
        }

        const std::shared_ptr<iset_t>& ptr() const
        {
            return m_set;
        }

        std::size_t level() const
        {
            return m_set->level();
        }

        /// Independent deep copy, safe to traverse from another thread.
        DynamicSet clone() const
        {
            return DynamicSet(m_set->clone());
        }

        DynamicSet on(std::size_t level) const; // defined below, once `dyn::on` is available

        template <class Func>
        void operator()(Func&& func) const
        {
            apply(as_static_set(m_set), std::forward<Func>(func));
        }

        auto to_lca() const
        {
            return as_static_set(m_set).to_lca();
        }

      private:

        std::shared_ptr<iset_t> m_set;
    };

    /// Build a `DynamicSet` from a child list and a rebuild functor.
    template <std::size_t dim_, class TInterval, class Rebuild>
    DynamicSet<dim_, TInterval> make_node(std::vector<std::shared_ptr<ISet<dim_, TInterval>>> children, Rebuild rebuild)
    {
        auto node = std::make_shared<Node<dim_, TInterval, Rebuild>>(std::move(children), std::move(rebuild));
        return DynamicSet<dim_, TInterval>(std::move(node));
    }

    template <class T>
    struct is_dynamic_set : std::false_type
    {
    };

    template <std::size_t dim_, class TInterval>
    struct is_dynamic_set<DynamicSet<dim_, TInterval>> : std::true_type
    {
    };

    ////////////////////////////////////////////////////////////////////////
    //// Dynamic DSL - mirrors the static free functions
    ////////////////////////////////////////////////////////////////////////

    namespace dyn
    {
        template <std::size_t Dim, class TInterval>
        DynamicSet<Dim, TInterval> self(const LevelCellArray<Dim, TInterval>& lca)
        {
            using LCA = LevelCellArray<Dim, TInterval>;
            return make_node<Dim, TInterval>({},
                                             [lca_ptr = &lca](const auto&)
                                             {
                                                 return LCAView<LCA>(*lca_ptr);
                                             });
        }

        template <class TValue, std::size_t Dim>
        DynamicSet<Dim, Interval<TValue>> box(std::size_t level, const Box<TValue, Dim>& b)
        {
            // `b` must outlive the returned set, as for the static `asBoxView`.
            return make_node<Dim, Interval<TValue>>({},
                                                    [level, box_ptr = &b](const auto&)
                                                    {
                                                        return asBoxView(level, *box_ptr);
                                                    });
        }

        namespace detail
        {
            template <SetOperator op, std::size_t dim, class TInterval>
            DynamicSet<dim, TInterval> binary(const DynamicSet<dim, TInterval>& a, const DynamicSet<dim, TInterval>& b)
            {
                return make_node<dim, TInterval>(
                    {a.ptr(), b.ptr()},
                    [](const auto& children)
                    {
                        using adaptor_t = DynamicSetAdaptor<dim, TInterval>;
                        if constexpr (op == SetOperator::UNION)
                        {
                            return union_(adaptor_t(children[0]), adaptor_t(children[1]));
                        }
                        else if constexpr (op == SetOperator::INTERSECTION)
                        {
                            return intersection(adaptor_t(children[0]), adaptor_t(children[1]));
                        }
                        else
                        {
                            return difference(adaptor_t(children[0]), adaptor_t(children[1]));
                        }
                    });
            }

            template <SetOperator op, std::size_t dim, class TInterval>
            DynamicSet<dim, TInterval> fold(const std::vector<DynamicSet<dim, TInterval>>& sets)
            {
                assert(!sets.empty());
                DynamicSet<dim, TInterval> acc = sets.front();
                for (std::size_t i = 1; i < sets.size(); ++i)
                {
                    acc = binary<op>(acc, sets[i]);
                }
                return acc;
            }
        } // namespace detail

        // Runtime overloads: the number of operands is a std::vector known at
        // runtime (the natural entry point for bindings).

        template <std::size_t dim, class TInterval>
        DynamicSet<dim, TInterval> union_(const std::vector<DynamicSet<dim, TInterval>>& sets)
        {
            return detail::fold<SetOperator::UNION>(sets);
        }

        template <std::size_t dim, class TInterval>
        DynamicSet<dim, TInterval> intersection(const std::vector<DynamicSet<dim, TInterval>>& sets)
        {
            assert(sets.size() >= 2);
            return detail::fold<SetOperator::INTERSECTION>(sets);
        }

        template <std::size_t dim, class TInterval>
        DynamicSet<dim, TInterval> difference(const std::vector<DynamicSet<dim, TInterval>>& sets)
        {
            assert(sets.size() >= 2);
            return detail::fold<SetOperator::DIFFERENCE>(sets);
        }

        // Variadic overloads: mirror the static DSL, `dim`/`TInterval` deduced
        // from the arguments. Constrained to DynamicSet so a single std::vector
        // argument routes to the runtime overloads above.

        template <class Set, class... Sets>
            requires(is_dynamic_set<Set>::value && (std::same_as<Set, Sets> && ...))
        Set union_(const Set& first, const Sets&... rest)
        {
            return union_(std::vector<Set>{first, rest...});
        }

        template <class Set, class... Sets>
            requires(is_dynamic_set<Set>::value && (std::same_as<Set, Sets> && ...) && sizeof...(Sets) >= 1)
        Set intersection(const Set& first, const Sets&... rest)
        {
            return intersection(std::vector<Set>{first, rest...});
        }

        template <class Set, class... Sets>
            requires(is_dynamic_set<Set>::value && (std::same_as<Set, Sets> && ...) && sizeof...(Sets) >= 1)
        Set difference(const Set& first, const Sets&... rest)
        {
            return difference(std::vector<Set>{first, rest...});
        }

        template <std::size_t dim, class TInterval>
        DynamicSet<dim, TInterval> on(const DynamicSet<dim, TInterval>& set, std::size_t level)
        {
            return make_node<dim, TInterval>({set.ptr()},
                                             [level](const auto& children)
                                             {
                                                 return as_static_set(children[0]).on(level);
                                             });
        }

        template <std::size_t dim, class TInterval, class Translation>
        DynamicSet<dim, TInterval> translate(const DynamicSet<dim, TInterval>& set, const Translation& t)
        {
            return make_node<dim, TInterval>({set.ptr()},
                                             [t](const auto& children)
                                             {
                                                 return samurai::translate(as_static_set(children[0]), t);
                                             });
        }

        template <std::size_t dim, class TInterval>
        DynamicSet<dim, TInterval> expand(const DynamicSet<dim, TInterval>& set, int width)
        {
            return make_node<dim, TInterval>({set.ptr()},
                                             [width](const auto& children)
                                             {
                                                 return nestedExpand(as_static_set(children[0]), width);
                                             });
        }

        template <std::size_t dim, class TInterval>
        DynamicSet<dim, TInterval> contract(const DynamicSet<dim, TInterval>& set, std::size_t width)
        {
            return make_node<dim, TInterval>({set.ptr()},
                                             [width](const auto& children)
                                             {
                                                 return samurai::contract(as_static_set(children[0]), width);
                                             });
        }

        template <std::size_t dim, class TInterval>
        DynamicSet<dim, TInterval>
        contract(const DynamicSet<dim, TInterval>& set, std::size_t width, const std::array<bool, dim>& directions)
        {
            return make_node<dim, TInterval>({set.ptr()},
                                             [width, directions](const auto& children)
                                             {
                                                 return samurai::contract(as_static_set(children[0]), width, directions);
                                             });
        }
    } // namespace dyn

    template <std::size_t dim_, class TInterval>
    DynamicSet<dim_, TInterval> DynamicSet<dim_, TInterval>::on(std::size_t level) const
    {
        return dyn::on(*this, level);
    }

} // namespace samurai
