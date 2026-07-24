// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <memory>
#include <utility>

#include "../../samurai_config.hpp"
#include "../traversers/set_traverser_base.hpp"

namespace samurai
{
    ////////////////////////////////////////////////////////////////////////
    //// Runtime traverser interface
    ////////////////////////////////////////////////////////////////////////

    /**
     * Runtime (type-erased) counterpart of the static traverser concept
     * described by `SetTraverserBase`.
     *
     * It exposes exactly the same three primitives -- `is_empty`,
     * `next_interval`, `current_interval` -- but through virtual calls, so a
     * traverser whose concrete type is only known at runtime can be used
     * uniformly.
     *
     * `clone` returns an independent copy carrying the same iteration state.
     * It is needed because the static traversers store their children *by
     * value* (see e.g. `IntersectionTraverser`), so any traverser plugged into
     * them must be copyable.
     */
    template <class TInterval>
    class ISetTraverser
    {
      public:

        using interval_t = TInterval;

        virtual ~ISetTraverser() = default;

        virtual std::unique_ptr<ISetTraverser> clone() const = 0;

        virtual bool is_empty() const = 0;

        virtual void next_interval() = 0;

        virtual interval_t current_interval() const = 0;

      protected:

        // Prevent slicing: copies go through `clone`.
        ISetTraverser()                                = default;
        ISetTraverser(const ISetTraverser&)            = default;
        ISetTraverser(ISetTraverser&&)                 = default;
        ISetTraverser& operator=(const ISetTraverser&) = default;
        ISetTraverser& operator=(ISetTraverser&&)      = default;
    };

    ////////////////////////////////////////////////////////////////////////
    //// AnyTraverser: the dynamic -> static bridge
    ////////////////////////////////////////////////////////////////////////

    template <class TInterval>
    class AnyTraverser;

    template <class TInterval>
    struct SetTraverserTraits<AnyTraverser<TInterval>>
    {
        using interval_t = TInterval;
        // Type erasure loses the reference: `current_interval` returns a value.
        using current_interval_t = interval_t;
    };

    /**
     * A *static* traverser (it models `SetTraverserBase`) that forwards to a
     * runtime `ISetTraverser`.
     *
     * This is the single boundary where the dynamic world meets the static
     * one: because `AnyTraverser` satisfies the static traverser concept, it
     * can be used as the `traverser_t<d>` of any existing static traverser
     * (union, intersection, difference, ...) without changing a single line of
     * those algorithms.
     */
    template <class TInterval>
    class AnyTraverser : public SetTraverserBase<AnyTraverser<TInterval>>
    {
        using Self = AnyTraverser<TInterval>;

      public:

        SAMURAI_SET_TRAVERSER_TYPEDEFS

        explicit AnyTraverser(std::unique_ptr<ISetTraverser<TInterval>> impl)
            : m_impl(std::move(impl))
        {
        }

        AnyTraverser(const AnyTraverser& other)
            : m_impl(other.m_impl->clone())
        {
        }

        AnyTraverser(AnyTraverser&&) noexcept            = default;
        AnyTraverser& operator=(AnyTraverser&&) noexcept = default;
        ~AnyTraverser()                                  = default;

        AnyTraverser& operator=(const AnyTraverser& other)
        {
            if (this != &other)
            {
                m_impl = other.m_impl->clone();
            }
            return *this;
        }

        SAMURAI_INLINE bool is_empty_impl() const
        {
            return m_impl->is_empty();
        }

        SAMURAI_INLINE void next_interval_impl()
        {
            m_impl->next_interval();
        }

        SAMURAI_INLINE current_interval_t current_interval_impl() const
        {
            return m_impl->current_interval();
        }

      private:

        std::unique_ptr<ISetTraverser<TInterval>> m_impl;
    };

    ////////////////////////////////////////////////////////////////////////
    //// SetTraverserModel: the static -> dynamic bridge
    ////////////////////////////////////////////////////////////////////////

    /**
     * Wraps a concrete static traverser into the runtime `ISetTraverser`
     * interface. This is the other direction of the bridge: it lets a leaf's
     * traverser (or a whole static sub-expression's traverser) be handed out
     * behind the virtual interface.
     */
    template <class SetTraverser>
    class SetTraverserModel final : public ISetTraverser<typename SetTraverser::interval_t>
    {
      public:

        using interval_t = typename SetTraverser::interval_t;

        explicit SetTraverserModel(SetTraverser traverser)
            : m_traverser(std::move(traverser))
        {
        }

        std::unique_ptr<ISetTraverser<interval_t>> clone() const override
        {
            return std::make_unique<SetTraverserModel>(m_traverser);
        }

        bool is_empty() const override
        {
            return m_traverser.is_empty();
        }

        void next_interval() override
        {
            m_traverser.next_interval();
        }

        interval_t current_interval() const override
        {
            return m_traverser.current_interval();
        }

      private:

        SetTraverser m_traverser;
    };

    /**
     * Erase the type of a static traverser into an `AnyTraverser`.
     */
    template <class SetTraverser>
    AnyTraverser<typename SetTraverser::interval_t> make_any_traverser(SetTraverser traverser)
    {
        using interval_t = typename SetTraverser::interval_t;
        return AnyTraverser<interval_t>(std::make_unique<SetTraverserModel<SetTraverser>>(std::move(traverser)));
    }

} // namespace samurai
