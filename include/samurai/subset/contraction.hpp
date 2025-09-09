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

        using value_t       = typename Base::value_t;
        using contraction_t = xt::xtensor_fixed<value_t, xt::xshape<Base::dim>>;

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

        std::size_t level() const
        {
            return m_set.level();
        }

        bool exist() const
        {
            return m_set.exist();
        }

        bool empty() const
        {
            return m_set.empty();
        }

        template <class index_t, std::size_t d>
        traverser_t<d> get_traverser(const index_t& index, std::integral_constant<std::size_t, d> d_ic) const
        {
            return traverser_t<d>(m_set.get_traverser(index, d_ic), m_contraction[d]);
        }

      private:

        Set m_set;
        contraction_t m_contraction;
    };

    template <class Set>
    auto contraction(const Set& set,
                     const std::array<typename SetBase<decltype(self(set))>::value_t, SetBase<decltype(self(set))>::getDim()>& contraction)
    {
        return Contraction(self(set), contraction);
    }

    template <class Set>
    auto contraction(const Set& set, const typename SetBase<decltype(self(set))>::value_t contraction)
    {
        return Contraction(self(set), contraction);
    }
}
