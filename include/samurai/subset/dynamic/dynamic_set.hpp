// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cstddef>
#include <memory>
#include <utility>

#include <xtensor/containers/xfixed.hpp>

#include "../../samurai_config.hpp"
#include "../set_base.hpp"
#include "any_traverser.hpp"

namespace samurai
{
    ////////////////////////////////////////////////////////////////////////
    //// Runtime set interface
    ////////////////////////////////////////////////////////////////////////

    /**
     * Runtime (type-erased) counterpart of the static set concept described by
     * `SetBase`. It exposes the same primitives through virtual calls, so a set
     * expression whose structure is only known at runtime can be manipulated
     * uniformly (and, later, exposed to bindings such as Python).
     *
     * Traversers are handed out as `AnyTraverser`, and the dimension `d`
     * becomes a runtime argument (instead of a template parameter). The
     * traversal scratch memory - the static `Workspace` - is kept *inside* each
     * concrete `ISet`, so the virtual signatures stay free of an erased
     * workspace type. Thread-safety is then obtained the same way the static
     * side gets it: `clone()` a tree per thread.
     */
    template <std::size_t dim_, class TInterval>
    class ISet
    {
      public:

        static constexpr std::size_t dim = dim_;

        using interval_t  = TInterval;
        using value_t     = typename interval_t::value_t;
        using yz_index_t  = xt::xtensor_fixed<value_t, xt::xshape<dim - 1>>;
        using traverser_t = AnyTraverser<interval_t>;

        virtual ~ISet() = default;

        /// Deep copy carrying independent traversal scratch (for per-thread use).
        virtual std::shared_ptr<ISet> clone() const = 0;

        virtual std::size_t level() const = 0;

        virtual bool exist() const = 0;

        virtual bool empty() const = 0;

        virtual void init_workspace(std::size_t d, std::size_t n_traversers) = 0;

        virtual traverser_t get_traverser(std::size_t d, const yz_index_t& index) = 0;

        virtual traverser_t get_traverser_unordered(std::size_t d, const yz_index_t& index) = 0;

      protected:

        ISet()                       = default;
        ISet(const ISet&)            = default;
        ISet(ISet&&)                 = default;
        ISet& operator=(const ISet&) = default;
        ISet& operator=(ISet&&)      = default;
    };

    ////////////////////////////////////////////////////////////////////////
    //// DynamicSetAdaptor: the dynamic -> static bridge
    ////////////////////////////////////////////////////////////////////////

    template <std::size_t dim_, class TInterval>
    class DynamicSetAdaptor;

    template <std::size_t dim_, class TInterval>
    struct SetTraits<DynamicSetAdaptor<dim_, TInterval>>
    {
        template <std::size_t>
        using traverser_t = AnyTraverser<TInterval>;

        // The real scratch memory lives inside the ISet, so nothing is needed here.
        struct Workspace
        {
        };

        static constexpr std::size_t dim()
        {
            return dim_;
        }
    };

    /**
     * A *static* set (it models `SetBase`) that forwards to a runtime `ISet`.
     *
     * This is the single boundary where the dynamic world meets the static one
     * at the set level: because `DynamicSetAdaptor` satisfies the static set
     * concept, it can be plugged as a child of *any* existing static combinator
     * (translation, projection, expansion, contraction, n-ary operators, ...)
     * without changing those templates. This is what lets the dynamic algebra
     * reuse the static algorithms verbatim.
     */
    template <std::size_t dim_, class TInterval>
    class DynamicSetAdaptor : public SetBase<DynamicSetAdaptor<dim_, TInterval>>
    {
        using Self = DynamicSetAdaptor<dim_, TInterval>;

      public:

        SAMURAI_SET_TYPEDEFS

        using iset_t = ISet<dim_, TInterval>;

        explicit DynamicSetAdaptor(std::shared_ptr<iset_t> set)
            : m_set(std::move(set))
        {
        }

        SAMURAI_INLINE std::size_t level_impl() const
        {
            return m_set->level();
        }

        SAMURAI_INLINE bool exist_impl() const
        {
            return m_set->exist();
        }

        SAMURAI_INLINE bool empty_impl() const
        {
            return m_set->empty();
        }

        template <std::size_t d>
        SAMURAI_INLINE void init_workspace_impl(const std::size_t n_traversers, std::integral_constant<std::size_t, d>, Workspace&) const
        {
            m_set->init_workspace(d, n_traversers);
        }

        template <std::size_t d>
        SAMURAI_INLINE traverser_t<d> get_traverser_impl(const yz_index_t& index, std::integral_constant<std::size_t, d>, Workspace&) const
        {
            return m_set->get_traverser(d, index);
        }

        template <std::size_t d>
        SAMURAI_INLINE traverser_t<d>
        get_traverser_unordered_impl(const yz_index_t& index, std::integral_constant<std::size_t, d>, Workspace&) const
        {
            return m_set->get_traverser_unordered(d, index);
        }

        const std::shared_ptr<iset_t>& ptr() const
        {
            return m_set;
        }

      private:

        std::shared_ptr<iset_t> m_set;
    };

    /**
     * Wrap a runtime `ISet` into a static `DynamicSetAdaptor`.
     */
    template <std::size_t dim_, class TInterval>
    DynamicSetAdaptor<dim_, TInterval> as_static_set(std::shared_ptr<ISet<dim_, TInterval>> set)
    {
        return DynamicSetAdaptor<dim_, TInterval>(std::move(set));
    }

} // namespace samurai
