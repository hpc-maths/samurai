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
    class NArySetOperator;

    template <Set_concept... Sets>
    struct SetTraits<NArySetOperator<SetOperator::UNION, Sets...>>
    {
        using Childrens = std::tuple<Sets...>;

        template <std::size_t d>
        using traverser_t = UnionTraverser<typename Sets::template traverser_t<d>...>;

        static constexpr std::size_t getDim()
        {
            return SetTraits<std::tuple_element_t<0, Childrens>>::getDim();
        }
    };

    template <Set_concept... Sets>
    struct SetTraits<NArySetOperator<SetOperator::INTERSECTION, Sets...>>
    {
        using Childrens = std::tuple<Sets...>;

        template <std::size_t d>
        using traverser_t = IntersectionTraverser<typename SetTraits<Sets>::template traverser_t<d>...>;

        static constexpr std::size_t getDim()
        {
            return SetTraits<std::tuple_element_t<0, Childrens>>::getDim();
        }
    };

    template <Set_concept... Sets>
    struct SetTraits<NArySetOperator<SetOperator::DIFFERENCE, Sets...>>
    {
        using Childrens = std::tuple<Sets...>;

        template <std::size_t d>
        using traverser_t = std::conditional_t<d == 0,
                                               DifferenceTraverser<typename Sets::template traverser_t<d>...>,
                                               DifferenceIdTraverser<typename Sets::template traverser_t<d>...>>;

        static constexpr std::size_t getDim()
        {
            return SetTraits<std::tuple_element_t<0, Childrens>>::getDim();
        }
    };

    template <SetOperator op, Set_concept... Sets>
    class NArySetOperator : public SetBase<NArySetOperator<op, Sets...>>
    {
        using Self      = NArySetOperator<op, Sets...>;
        using Base      = SetBase<Self>;
        using Childrens = typename SetTraits<Self>::Childrens;

      public:

        template <std::size_t d>
        using traverser_t = typename Base::template traverser_t<d>;

        static constexpr std::size_t nIntervals = std::tuple_size_v<Childrens>;

        explicit NArySetOperator(const Sets&... sets)
            : m_sets(sets...)
        {
            m_level_impl = std::apply(
                [](const auto&... set) -> std::size_t
                {
                    return vmax(set.level_impl()...);
                },
                m_sets);

            enumerate_const_items(m_sets,
                                  [this](const auto i, const auto& set)
                                  {
                                      m_shifts[i] = std::size_t(m_level_impl - set.level_impl());
                                  });
        }

        std::size_t level_impl() const
        {
            return m_level_impl;
        }

        bool exist_impl() const
        {
            return std::apply(
                [](const auto first_set, const auto&... other_sets) -> std::size_t
                {
                    if constexpr (op == SetOperator::UNION)
                    {
                        return first_set.exist_impl() || (other_sets.exist_impl() || ...);
                    }
                    else if constexpr (op == SetOperator::INTERSECTION)
                    {
                        return first_set.exist_impl() && (other_sets.exist_impl() && ...);
                    }
                    else
                    {
                        return first_set.exist_impl();
                    }
                },
                m_sets);
        }

        bool empty_impl() const
        {
            return std::apply(
                [this](const auto first_set, const auto&... other_sets) -> std::size_t
                {
                    if constexpr (op == SetOperator::UNION)
                    {
                        return first_set.empty_impl() && (other_sets.empty_impl() && ...);
                    }
                    else if constexpr (op == SetOperator::INTERSECTION)
                    {
                        return first_set.empty_impl() || (other_sets.empty_impl() || ...);
                    }
                    else
                    {
                        return first_set.empty_impl();
                    }
                },
                m_sets);
        }

        template <class index_t, std::size_t d>
        traverser_t<d> get_traverser_impl(const index_t& index, std::integral_constant<std::size_t, d> d_ic) const
        {
            return get_traverser_impl_impl(index, d_ic, std::make_index_sequence<nIntervals>{});
        }

      private:

        template <class index_t, std::size_t d, std::size_t... Is>
        traverser_t<d> get_traverser_impl_impl(const index_t& index, std::integral_constant<std::size_t, d> d_ic, std::index_sequence<Is...>) const
        {
            return traverser_t<d>(m_shifts, std::get<Is>(m_sets).get_traverser_impl(index >> m_shifts[Is], d_ic)...);
        }

        Childrens m_sets;
        std::size_t m_level_impl;
        std::array<std::size_t, nIntervals> m_shifts;
    };

    ////////////////////////////////////////////////////////////////////////
    //// functions
    ////////////////////////////////////////////////////////////////////////

    template <class... Sets>
    auto union_(const Sets&... sets)
        requires(sizeof...(Sets) >= 1)
    {
        //~ using Union = NArySetOperator<SetOperator::UNION, std::decay_t<decltype(self(sets))>...>;
        using Union = NArySetOperator<SetOperator::UNION, typename SelfTraits<Sets>::Type...>;

        return Union(self(sets)...);
    }

    template <class... Sets>
    auto intersection(const Sets&... sets)
        requires(sizeof...(Sets) >= 2)
    {
        using Intersection = NArySetOperator<SetOperator::INTERSECTION, typename SelfTraits<Sets>::Type...>;

        return Intersection(self(sets)...);
    }

    template <class... Sets>
    auto difference(const Sets&... sets)
        requires(sizeof...(Sets) >= 2)
    {
        using Difference = NArySetOperator<SetOperator::DIFFERENCE, typename SelfTraits<Sets>::Type...>;

        return Difference(self(sets)...);
    }

} // namespace samurai
