// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cstddef>
#include <utility>

namespace samurai
{
    template <class SetTraverser>
    struct SetTraverserTraits;

    template <class Derived>
    class SetTraverserBase;

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
			assert(!is_empty());
            derived_cast().next_interval_impl();
        }

        inline current_interval_t current_interval() const
        {
            return derived_cast().current_interval_impl();
        }
    };
    
    template<typename T>
    struct IsSetTraverser : std::bool_constant< std::is_base_of<SetTraverserBase<T>, T>::value > {};
    
    #define SAMURAI_SET_TRAVERSER_TYPEDEFS \
		using Base               = SetTraverserBase<Self>; \
		using interval_t         = typename Base::interval_t; \
		using current_interval_t = typename Base::current_interval_t; \
		using value_t            = typename Base::value_t; \
		
} // namespace samurai
