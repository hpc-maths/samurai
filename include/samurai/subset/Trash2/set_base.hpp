// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <cstddef>

#include <utility>

#pragma once

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
    void apply(Set&& global_set, Func&& func);

    ////////////////////////////////////////////////////////////////////////
    //// Class definition
    ////////////////////////////////////////////////////////////////////////

    template <class Derived>
    class SetBase
    {
        using DerivedTraits = SetTraits<Derived>;

      public:

        using interval_t = typename DerivedTraits::interval_t;
        using value_t    = typename interval_t::value_t;

        static constexpr std::size_t dim = DerivedTraits::dim;

        const Derived& derived_cast() const
        {
            return static_cast<const Derived&>(*this);
        }

        Derived& derived_cast()
        {
            return static_cast<Derived&>(*this);
        }

        Derived&& derived_forward()
        {
            return static_cast<Derived&&>(*this);
        }

        std::size_t min_level() const
        {
            return derived_cast().min_level();
        }

        std::size_t level() const
        {
            return derived_cast().level();
        }

        std::size_t ref_level() const
        {
            return derived_cast().ref_level();
        }

        bool exists() const
        {
            return derived_cast().exists();
        }

        bool empty() const
        {
            return derived_cast().empty();
        }

        inline Projection<Derived> on(const std::size_t level);

        template <class Func>
        void operator()(Func&& func)
        {
            apply(derived_forward(), std::forward<Func>(func));
        }

        template <class... ApplyOp>
        void apply_op(ApplyOp&&... op)
        {
            auto func = [&](auto& interval, auto& index)
            {
                (op(level(), interval, index), ...);
            };
            apply(derived_forward(), func);
        }
    };

} // namespace samurai

#include "projection.hpp"

namespace samurai
{

    template <class Derived>
    Projection<Derived> SetBase<Derived>::on(const std::size_t level)
    {
        return Projection<Derived>(derived_cast(), level);
    }

}
