// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_base.hpp"
#include "traversers/contraction_traverser.hpp"

namespace samurai
{

    template <Set_concept Set>
    class Contraction;

    template <Set_concept Set>
    struct SetTraits<Contraction<Set>>
    {
        template <std::size_t d>
        using traverser_t = ContractionTraverser<typename Set::template traverser_t<d>>;

        static constexpr std::size_t getDim()
        {
            return SetTraits<Set>::getDim();
        }
    };

    template <Set_concept Set>
    class Contraction : public SetBase<Contraction<Set>>
    {
        using Self = Contraction<Set>;
        using Base = SetBase<Self>;

      public:

        template <std::size_t d>
        using traverser_t = typename Base::template traverser_t<d>;

        using value_t          = typename Base::value_t;
        using contraction_t    = std::array<value_t, Base::dim>;
        using do_contraction_t = std::array<bool, Base::dim>;

        Contraction(const Set& set, const contraction_t& contraction)
            : m_set(set)
            , m_contraction(contraction)
        {
            assert(std::all_of(m_contraction.cbegin(),
                               m_contraction.cend(),
                               [](const contraction_t& c)
                               {
                                   return c >= 0;
                               }));
        }

        Contraction(const Set& set, const value_t contraction)
            : m_set(set)
        {
            assert(contraction >= 0);
            std::fill(m_contraction.begin(), m_contraction.end(), contraction);
        }
        
        Contraction(const Set& set, const value_t contraction, const do_contraction_t& do_contraction)
			: m_set(set)
        {
			for (std::size_t i=0; i!=m_contraction.size(); ++i)
			{
				m_contraction[i] = contraction*do_contraction[i];
			}
			
		}

        std::size_t level_impl() const
        {
            return m_set.level_impl();
        }

        bool exist_impl() const
        {
            return m_set.exist_impl();
        }

        bool empty_impl() const
        {
            return m_set.empty_impl();
        }

        template <class index_t, std::size_t d>
        traverser_t<d> get_traverser_impl(const index_t& index, std::integral_constant<std::size_t, d> d_ic) const
        {
            return traverser_t<d>(m_set.get_traverser_impl(index, d_ic), m_contraction[d]);
        }

      private:

        Set m_set;
        contraction_t m_contraction;
    };

    template <class Set>
    auto contract(const Set& set, const typename Contraction<typename SelfTraits<Set>::Type>::contraction_t& contraction)
    {
        return Contraction(self(set), contraction);
    }

    template <class Set>
    auto contract(const Set& set, const typename Contraction<typename SelfTraits<Set>::Type>::value_t contraction)
    {
        return Contraction(self(set), contraction);
    }

    template <class Set>
    auto contract(const Set& set, const typename Contraction<typename SelfTraits<Set>::Type>::value_t contraction, const typename Contraction<typename SelfTraits<Set>::Type>::do_contraction_t& do_contraction)
    {
        return Contraction(self(set), contraction, do_contraction);
    }
}
