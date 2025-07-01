// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "intersection_traverser.hpp"
#include "set_base.hpp"
#include "union_traverser.hpp"

namespace samurai
{
    enum class SetOperator
    {
        UNION,
        INTERSECTION
    };

    template <SetOperator op, Set_concept... Sets>
    class SubSet;

    template <Set_concept... Sets>
    struct SetTraits<SubSet<SetOperator::UNION, Sets...>>
    {
        using traverser_t = UnionTraverser<(typename SetTraits<Sets>::traverser_t)...>;
    };

    template <Set_concept... Sets>
    struct SetTraits<SubSet<SetOperator::INTERSECTION, Sets...>>
    {
        using traverser_t = IntersectionTraverser<(typename SetTraits<Sets>::traverser_t)...>;
    };

    template <SetOperator op, Set_concept... Sets>
    class SubSet : public SetBase<SubSet<op, Sets...>>
    {
        static_assert(sizeof...(Sets) >= 2);

        using Base     = SetBase<SubSet<op, Set>>;
        using SetTuple = std::tuple<Set...>;

        static constexpr std::size_t nIntervals = std::tuple_size_v<Childrens>;

      public:

        SubSet(const Sets&... sets)
            : m_sets(sets)
        {
            m_level = std::apply(
                [this](const auto&... set) -> std::size_t
                {
                    return vmin(set.level()...);
                },
                m_sets);
        }

        std::size_t level() const
        {
            return m_level;
        }

        bool exists() const
        {
            return std::apply(
                [this](const auto&... set) -> std::size_t
                {
                    return set.exists() || ...;
                },
                m_sets);
        }

        bool empty() const
        {
            return std::apply(
                [this](const auto&... set) -> std::size_t
                {
                    return set.empty() && ...;
                },
                m_sets);
        }

        template <std::size_t d>
        traverser_t get_traverser(const index_t& index, std::integral_constant<std::size_t d> d_ic) const
        {
            return get_traverser_impl(index, d_ic, std::make_index_sequence<nIntervals>{});
        }

      private:

        template <std::size_t d, std::size_t... Is>
        traverser_t get_traverser_impl(const index_t& index, std::integral_constant<std::size_t d> d_ic, std::index_sequence<Is...>) const
        {
            std::array<std::size_t, nIntervals> shifts{std::size_t(std::get<Is>(m_sets).level() - m_level)...};
            return traverser_t(shifts, (std::get<Is>(m_sets).get_traverser(index, d_ic))...);
        }

        Sets m_sets;
        std::size_t m_level;
    };

    template <Set_concept FirstSet, Set_concept SecondSet, Set_concept... OtherSets>
    SubSet<SetOperator::UNION, FirstSet, SecondSet, OtherSets...>
    union_(const FirstSet& firstSet, const SecondSet& secondSet, const OtherSets&... otherSets)
    {
        return SubSet<SetOperator::UNION, FirstSet, SecondSet, OtherSets...>(firstSet, secondSet, otherSets...);
    }

    template <Set_concept FirstSet, Set_concept SecondSet, Set_concept... OtherSets>
    SubSet<SetOperator::INTERSECTION, FirstSet, SecondSet, OtherSets...>
    intersection(const FirstSet& firstSet, const SecondSet& secondSet, const OtherSets&... otherSets)
    {
        return SubSet<SetOperator::INTERSECTION, FirstSet, SecondSet, OtherSets...>(firstSet, secondSet, otherSets...);
    }

} // namespace samurai
