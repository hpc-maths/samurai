// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cstddef>
#include <utility>

namespace samurai
{
    template <class UndefinedSetTraverserRange>
    struct SetTraverserRangeTraits;

    /*
     * For the sake of conscision, the ranges are only forward range
     * but they could easily be converted to a random_access range.
     */
    template <class Derived>
    class SetTraverserRangeBase
    {
      public:

        using DerivedTraits = SetTraverserRangeTraits<Derived>;

        using Iterator = typename DerivedTraits::Iterator;

        const Derived& derived_cast() const
        {
            return static_cast<const Derived&>(*this);
        }

        Derived& derived_cast()
        {
            return static_cast<Derived&>(*this);
        }

        Iterator begin()
        {
            derived_cast().begin_impl();
        }

        Iterator end()
        {
            derived_cast().end_impl();
        }
    };

    template <typename T>
    struct IsSetTraverserRange : std::bool_constant<std::is_base_of<SetTraverserRangeBase<T>, T>::value>
    {
    };

#define SAMURAI_SET_TRAVERSER_RANGE_TYPEDEFS \
    using Base     = SetTraverserBase<Self>; \
    using Iterator = typename Base::Iterator;

} // namespace samurai
