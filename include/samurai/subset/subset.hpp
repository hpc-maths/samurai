// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "traversers/difference_id_traverser.hpp"
#include "traversers/difference_traverser.hpp"
#include "traversers/intersection_traverser.hpp"
#include "traversers/union_traverser.hpp"

#include "set_base.hpp"

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
        using Childrens = std::tuple<Sets...>;

        template <std::size_t d>
        using traverser_t = UnionTraverser<typename SetTraits<Sets>::template traverser_t<d>...>;

        static constexpr std::size_t dim = SetTraits<std::tuple_element_t<0, Childrens>>::dim;
    };

    template <Set_concept... Sets>
    struct SetTraits<SubSet<SetOperator::INTERSECTION, Sets...>>
    {
        using Childrens = std::tuple<Sets...>;

        template <std::size_t d>
        using traverser_t = IntersectionTraverser<typename SetTraits<Sets>::template traverser_t<d>...>;

        static constexpr std::size_t dim = SetTraits<std::tuple_element_t<0, Childrens>>::dim;
    };

    template <Set_concept... Sets>
    struct SetTraits<SubSet<SetOperator::DIFFERENCE, Sets...>>
    {
        using Childrens = std::tuple<Sets...>;

        template <std::size_t d>
        using traverser_t = std::conditional_t<d == 0,
                                               DifferenceTraverser<typename SetTraits<Sets>::template traverser_t<d>...>,
                                               DifferenceIdTraverser<typename SetTraits<Sets>::template traverser_t<d>...>>;

        static constexpr std::size_t dim = SetTraits<std::tuple_element_t<0, Childrens>>::dim;
    };

    template <SetOperator op, Set_concept... Sets>
    class SubSet : public SetBase<SubSet<op, Sets...>>
    {
        static_assert(sizeof...(Sets) >= 2);
        using Self      = SubSet<op, Sets...>;
        using Base      = SetBase<Self>;
        using Childrens = typename SetTraits<Self>::Childrens;

      public:

        template <std::size_t d>
        using traverser_t = typename Base::template traverser_t<d>;

        static constexpr std::size_t nIntervals = std::tuple_size_v<Childrens>;

        SubSet(const Sets&... sets)
            : m_sets(sets...)
        {
            m_level = std::apply(
                [](const auto&... set) -> std::size_t
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
                [](const auto first_set, const auto&... other_sets) -> std::size_t
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
        traverser_t<d> get_traverser(const index_t& index, std::integral_constant<std::size_t, d> d_ic) const
        {
            return get_traverser_impl(index, d_ic, std::make_index_sequence<nIntervals>{});
        }

      private:

        template <class index_t, std::size_t d, std::size_t... Is>
        traverser_t<d> get_traverser_impl(const index_t& index, std::integral_constant<std::size_t, d> d_ic, std::index_sequence<Is...>) const
        {
            return traverser_t<d>(m_shifts, std::get<Is>(m_sets).get_traverser(index >> m_shifts[Is], d_ic)...);
        }

        Childrens m_sets;
        std::size_t m_level;
        std::array<std::size_t, nIntervals> m_shifts;
    };

    ////////////////////////////////////////////////////////////////////////
    //// functions
    ////////////////////////////////////////////////////////////////////////

    template <class... Sets>
    auto union_(const Sets&... sets)
        requires(sizeof...(Sets) >= 2)
    {
        using Union = SubSet<SetOperator::UNION, std::decay_t<decltype(self(sets))>...>;

        return Union(self(sets)...);
    }

    template <class... Sets>
    auto intersection(const Sets&... sets)
        requires(sizeof...(Sets) >= 2)
    {
        using Intersection = SubSet<SetOperator::INTERSECTION, std::decay_t<decltype(self(sets))>...>;

        return Intersection(self(sets)...);
    }

    template <class... Sets>
    auto difference(const Sets&... sets)
        requires(sizeof...(Sets) >= 2)
    {
        using Difference = SubSet<SetOperator::DIFFERENCE, std::decay_t<decltype(self(sets))>...>;

        return Difference(self(sets)...);
    }

} // namespace samurai
