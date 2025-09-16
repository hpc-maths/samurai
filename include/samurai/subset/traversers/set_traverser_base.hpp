// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <cstddef>

#include <utility>

#pragma once

namespace samurai
{
    template <class SetTraverser>
    struct SetTraverserTraits;

    template <class Derived>
    class SetTraverserBase;

    template <typename T>
    concept SetTraverser_concept = std::is_base_of<SetTraverserBase<T>, T>::value;

    template <class Derived>
    class SetTraverserBase
    {
      public:

        using interval_t         = typename SetTraverserTraits<Derived>::interval_t;
        using current_interval_t = typename SetTraverserTraits<Derived>::current_interval_t;
        using value_t            = typename interval_t::value_t;

        const Derived& derived_cast() const
        {
            return static_cast<const Derived&>(*this);
        }

        Derived& derived_cast()
        {
            return static_cast<Derived&>(*this);
        }

        inline bool is_empty() const
        {
            return derived_cast().is_empty_impl();
        }

        inline void next_interval()
        {
            derived_cast().next_interval_impl();
        }

        inline current_interval_t current_interval() const
        {
            return derived_cast().current_interval_impl();
        }
    };
}
