// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <concepts>
#include <cstddef>
#include <utility>

#include "../../samurai_config.hpp"
#include "../traversers/set_traverser_base.hpp"
#include "small_object.hpp"

namespace samurai
{
    ////////////////////////////////////////////////////////////////////////
    //// Runtime traverser interface
    ////////////////////////////////////////////////////////////////////////

    /**
     * Runtime (type-erased) counterpart of the static traverser concept
     * described by `SetTraverserBase`. It exposes the same three primitives --
     * `is_empty`, `next_interval`, `current_interval` -- through virtual calls,
     * plus the `copy_to` / `move_to` that let it live in a `sbo::SmallObject`.
     * Concrete models get `copy_to` / `move_to` for free from `sbo::Cloneable`.
     */
    template <class TInterval>
    class ISetTraverser
    {
      public:

        using interval_t = TInterval;

        virtual ~ISetTraverser() = default;

        virtual bool is_empty() const = 0;

        virtual void next_interval() = 0;

        virtual interval_t current_interval() const = 0;

        virtual sbo::PlacedObject<ISetTraverser> copy_to(void* buffer, std::size_t capacity) const = 0;

        virtual sbo::PlacedObject<ISetTraverser> move_to(void* buffer, std::size_t capacity) = 0;

      protected:

        ISetTraverser()                                = default;
        ISetTraverser(const ISetTraverser&)            = default;
        ISetTraverser(ISetTraverser&&)                 = default;
        ISetTraverser& operator=(const ISetTraverser&) = default;
        ISetTraverser& operator=(ISetTraverser&&)      = default;
    };

    ////////////////////////////////////////////////////////////////////////
    //// SetTraverserModel: wraps a concrete static traverser
    ////////////////////////////////////////////////////////////////////////

    /**
     * Wraps a concrete static traverser into the runtime `ISetTraverser`
     * interface, so a leaf's traverser (or a whole static sub-expression's
     * traverser) can be handed out behind the virtual interface.
     */
    template <class SetTraverser>
    class SetTraverserModel final : public sbo::Cloneable<ISetTraverser<typename SetTraverser::interval_t>, SetTraverserModel<SetTraverser>>
    {
      public:

        using interval_t = typename SetTraverser::interval_t;

        explicit SetTraverserModel(SetTraverser traverser)
            : m_traverser(std::move(traverser))
        {
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
     *
     * The erased traverser is held in a `sbo::SmallObject`, so leaf traversers
     * (and their copies into the parent operators' tuples, the bulk of a
     * traversal's allocations) stay inline instead of hitting the heap.
     */
    template <class TInterval>
    class AnyTraverser : public SetTraverserBase<AnyTraverser<TInterval>>
    {
        using Self   = AnyTraverser<TInterval>;
        using impl_t = ISetTraverser<TInterval>;

        // Sized to hold a leaf model (a couple of iterators + vptr) with margin.
        static constexpr std::size_t buffer_size = 48;

      public:

        SAMURAI_SET_TRAVERSER_TYPEDEFS

        template <class SetTraverser>
            requires(IsSetTraverser<SetTraverser>::value && !std::same_as<SetTraverser, Self>)
        explicit AnyTraverser(SetTraverser traverser)
            : m_impl(SetTraverserModel<SetTraverser>(std::move(traverser)))
        {
        }

        SAMURAI_INLINE bool is_empty_impl() const
        {
            return m_impl.get()->is_empty();
        }

        SAMURAI_INLINE void next_interval_impl()
        {
            m_impl.get()->next_interval();
        }

        SAMURAI_INLINE current_interval_t current_interval_impl() const
        {
            return m_impl.get()->current_interval();
        }

      private:

        sbo::SmallObject<impl_t, buffer_size> m_impl;
    };

    /**
     * Erase the type of a static traverser into an `AnyTraverser`.
     */
    template <class SetTraverser>
    AnyTraverser<typename SetTraverser::interval_t> make_any_traverser(SetTraverser traverser)
    {
        return AnyTraverser<typename SetTraverser::interval_t>(std::move(traverser));
    }

} // namespace samurai
