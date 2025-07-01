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

        using interval_t = typename SetTraits<Set>::interval_t;
        using value_t    = typename interval_t::value_t;

        static constexpr static constexpr std::size_t dim = SetTraits<Set>::dim;

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
            return derived_cast().is_empty();
        }

        inline void next_interval()
        {
            derived_cast().next_interval();
        }

        inline const interval_t& current_interval() const
        {
            derived_cast().current_interval()
        }
    };
}
