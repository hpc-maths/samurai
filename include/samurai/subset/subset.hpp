// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "difference_traverser.hpp"
#include "intersection_traverser.hpp"
#include "union_traverser.hpp"

#include "set_base.hpp"

#include "box_view.hpp"
#include "lca_view.hpp"

namespace samurai
{
    enum class SetOperator
    {
        UNION,
        INTERSECTION,
        DIFFERENCE
    };

    template <SetOperator op, Set_concept... Sets>
    class SubSet;

    template <Set_concept... Sets>
    struct SetTraits<SubSet<SetOperator::UNION, Sets...>>
    {
        using traverser_t = UnionTraverser<typename SetTraits<Sets>::traverser_t...>;
    };

    template <Set_concept... Sets>
    struct SetTraits<SubSet<SetOperator::INTERSECTION, Sets...>>
    {
        using traverser_t = IntersectionTraverser<typename SetTraits<Sets>::traverser_t...>;
    };

    template <Set_concept... Sets>
    struct SetTraits<SubSet<SetOperator::DIFFERENCE, Sets...>>
    {
        using traverser_t = DifferenceTraverser<typename SetTraits<Sets>::traverser_t...>;
    };

    template <SetOperator op, Set_concept... Sets>
    class SubSet : public SetBase<SubSet<op, Sets...>>
    {
        static_assert(sizeof...(Sets) >= 2);

        using Base      = SetBase<SubSet<op, Sets...>>;
        using Childrens = std::tuple<Sets...>;

      public:

        using traverser_t = typename Base::traverser_t;

        static constexpr std::size_t nIntervals = std::tuple_size_v<Childrens>;

        SubSet(const Sets&... sets)
            : m_sets(sets...)
        {
            m_level = std::apply(
                [this](const auto&... set) -> std::size_t
                {
                    return vmax(set.level()...);
                },
                m_sets);

            enumerate_const_items(m_sets,
                                  [this](const auto i, const auto& set)
                                  {
                                      m_shifts[i] = std::size_t(m_level - set.level());
                                  });
        }

        std::size_t level() const
        {
            return m_level;
        }

        bool exist() const
        {
            return std::apply(
                [this](const auto first_set, const auto&... other_sets) -> std::size_t
                {
                    if constexpr (op == SetOperator::UNION)
                    {
                        return first_set.exist() || (other_sets.exist() || ...);
                    }
                    else if constexpr (op == SetOperator::INTERSECTION)
                    {
                        return first_set.exist() && (other_sets.exist() && ...);
                    }
                    else
                    {
                        return first_set.exist();
                    }
                },
                m_sets);
        }

        bool empty() const
        {
            return std::apply(
                [this](const auto first_set, const auto&... other_sets) -> std::size_t
                {
                    if constexpr (op == SetOperator::UNION)
                    {
                        return first_set.empty() && (other_sets.empty() && ...);
                    }
                    else if constexpr (op == SetOperator::INTERSECTION)
                    {
                        return first_set.empty() || (other_sets.empty() || ...);
                    }
                    else
                    {
                        return first_set.empty();
                    }
                },
                m_sets);
        }

        template <class index_t, std::size_t d>
        traverser_t get_traverser(const index_t& index, std::integral_constant<std::size_t, d> d_ic) const
        {
            return get_traverser_impl(index, d_ic, std::make_index_sequence<nIntervals>{});
        }

      private:

        template <class index_t, std::size_t d, std::size_t... Is>
        traverser_t get_traverser_impl(const index_t& index, std::integral_constant<std::size_t, d> d_ic, std::index_sequence<Is...>) const
        {
            return traverser_t(m_shifts, std::get<Is>(m_sets).get_traverser(index >> m_shifts[Is], d_ic)...);
        }

        Childrens m_sets;
        std::size_t m_level;
        std::array<std::size_t, nIntervals> m_shifts;
    };

    template <Set_concept... Sets>
    using Union = SubSet<SetOperator::UNION, Sets...>;

    template <Set_concept... Sets>
    using Intersection = SubSet<SetOperator::INTERSECTION, Sets...>;

    template <Set_concept... Sets>
    using Difference = SubSet<SetOperator::DIFFERENCE, Sets...>;

    ////////////////////////////////////////////////////////////////////////
    //// functions
    ////////////////////////////////////////////////////////////////////////
    template <class FirstSet, class SecondSet, class... OtherSets>
    auto union_(const FirstSet& firstSet, const SecondSet& secondSet, const OtherSets&... otherSets)
    {
        return Union(self(firstSet), self(secondSet), self(otherSets)...);
    }

    template <class FirstSet, class SecondSet, class... OtherSets>
    auto intersection(const FirstSet& firstSet, const SecondSet& secondSet, const OtherSets&... otherSets)
    {
        return Intersection(self(firstSet), self(secondSet), self(otherSets)...);
    }

    template <class FirstSet, class SecondSet, class... OtherSets>
    auto difference(const FirstSet& firstSet, const SecondSet& secondSet, const OtherSets&... otherSets)
    {
        return Difference(self(firstSet), self(secondSet), self(otherSets)...);
    }

} // namespace samurai
