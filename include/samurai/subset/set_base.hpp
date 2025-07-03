// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cstddef>

#include <utility>

#include "set_traverser_base.hpp"

namespace samurai
{

    ////////////////////////////////////////////////////////////////////////
    //// Forward Declarations
    ////////////////////////////////////////////////////////////////////////

    template <class Set>
    class Projection;
    template <class Set>
    struct SetTraits;
    template <class Derived>
    class SetBase;
    template <typename T>
    concept Set_concept = std::is_base_of<SetBase<T>, T>::value;

    template <Set_concept Set, class Func>
    void apply(const Set& set, Func&& func);

    ////////////////////////////////////////////////////////////////////////
    //// Class definition
    ////////////////////////////////////////////////////////////////////////

    template <class Derived>
    class SetBase
    {
        using DerivedTraits = SetTraits<Derived>;

      public:

        static constexpr std::size_t dim = DerivedTraits::traverser_t::dim;

        using traverser_t = typename DerivedTraits::traverser_t;
        using interval_t  = typename SetTraverserTraits<traverser_t>::interval_t;
        using value_t     = typename interval_t::value_t;

        const Derived& derived_cast() const
        {
            return static_cast<const Derived&>(*this);
        }

        Derived& derived_cast()
        {
            return static_cast<Derived&>(*this);
        }

        std::size_t level() const
        {
            return derived_cast().level();
        }

        bool exist() const
        {
            return derived_cast().exists();
        }

        bool empty() const
        {
            return derived_cast().empty();
        }

        template <class index_t, std::size_t d>
        traverser_t get_traverser(const index_t& index, std::integral_constant<std::size_t, d> d_ic) const
        {
            return derived_cast().get_traverser(index, d_ic);
        }

        //~ inline Projection<Derived> on(const std::size_t level);

        template <class Func>
        void operator()(Func&& func) const
        {
            apply(derived_cast(), std::forward<Func>(func));
        }

        template <class... ApplyOp>
        void apply_op(ApplyOp&&... op) const
        {
            auto func = [&](auto& interval, auto& index)
            {
                (op(level(), interval, index), ...);
            };
            apply(derived_cast(), func);
        }
    };

} // namespace samurai

//~ #include "projected_set.hpp"

//~ namespace samurai
//~ {

//~ template <class Derived>
//~ Projection<Derived> SetBase<Derived>::on(const std::size_t level)
//~ {
//~ return Projection<Derived>(derived_cast(), level);
//~ }

//~ }
