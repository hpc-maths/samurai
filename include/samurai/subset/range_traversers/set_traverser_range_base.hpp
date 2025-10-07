// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cstddef>
#include <utility>

namespace samurai
{
    template <class UndefinedSetTraverserRange>
    struct SetTraverserRangeTraits;

    template <class Derived>
    class SetTraverserRangeBase
    {
      public:

        using DerivedTraits = SetTraverserRangeTraits<Derived>;

        using Iterator       = typename DerivedTraits::Iterator;
        using const_Iterator = typename DerivedTraits::const_Iterator;

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

        const_Iterator begin() const
        {
            derived_cast().begin_impl();
        }

        const_Iterator cbegin() const
        {
            derived_cast().begin_impl();
        }
    };

    template <typename T>
    struct IsSetTraverserRange : std::bool_constant<std::is_base_of<SetTraverserRangeBase<T>, T>::value>
    {
    };

#define SAMURAI_SET_TRAVERSER_RANGE_TYPEDEFS        \
    using Base           = SetTraverserBase<Self>;  \
    using Iterator       = typename Base::Iterator; \
    using const_Iterator = typename Base::const_Iterator;

} // namespace samurai
