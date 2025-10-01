// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_base.hpp"
#include "traversers/difference_id_traverser.hpp"
#include "traversers/difference_traverser.hpp"
#include "traversers/intersection_traverser.hpp"
#include "traversers/union_traverser.hpp"
#include "utils.hpp"

namespace samurai
{
    enum class SetOperator
    {
        UNION,
        INTERSECTION,
        DIFFERENCE
    };

    template <SetOperator op, class... Sets>
    class NArySetOperator;

    template <class... Sets>
    struct SetTraits<NArySetOperator<SetOperator::UNION, Sets...>>
    {
        static_assert((IsSet<Sets>::value and ...));

        template <std::size_t d>
        using traverser_t = UnionTraverser<typename Sets::template traverser_t<d>...>;

        static constexpr std::size_t dim = std::tuple_element_t<0, std::tuple<Sets...>>::dim;
    };

    template <class... Sets>
    struct SetTraits<NArySetOperator<SetOperator::INTERSECTION, Sets...>>
    {
        static_assert((IsSet<Sets>::value and ...));

        template <std::size_t d>
        using traverser_t = IntersectionTraverser<typename Sets::template traverser_t<d>...>;

        static constexpr std::size_t dim = std::tuple_element_t<0, std::tuple<Sets...>>::dim;
    };

    template <class... Sets>
    struct SetTraits<NArySetOperator<SetOperator::DIFFERENCE, Sets...>>
    {
        static_assert((IsSet<Sets>::value and ...));

        template <std::size_t d>
        using traverser_t = std::conditional_t<d == 0,
                                               DifferenceTraverser<typename Sets::template traverser_t<d>...>,
                                               DifferenceIdTraverser<typename Sets::template traverser_t<d>...>>;

        static constexpr std::size_t dim = std::tuple_element_t<0, std::tuple<Sets...>>::dim;
    };

    template <SetOperator op, class... Sets>
    class NArySetOperator : public SetBase<NArySetOperator<op, Sets...>>
    {
        using Self = NArySetOperator<op, Sets...>;

      public:

        SAMURAI_SET_TYPEDEFS
        SAMURAI_SET_CONSTEXPRS

        using Childrens = std::tuple<Sets...>;

        static constexpr std::size_t nIntervals = std::tuple_size_v<Childrens>;

        explicit NArySetOperator(const Sets&... sets)
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

        inline std::size_t level_impl() const
        {
            return m_level;
        }

        inline bool exist_impl() const
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

        inline bool empty_impl() const
        {
			return Base::empty_default_impl();
        }

        template <class index_t, std::size_t d>
        inline traverser_t<d> get_traverser_impl(const index_t& index, std::integral_constant<std::size_t, d> d_ic) const
        {
            return get_traverser_impl_detail(index, d_ic, std::make_index_sequence<nIntervals>{});
        }

      private:

        template <class index_t, std::size_t d, std::size_t... Is>
        traverser_t<d>
        get_traverser_impl_detail(const index_t& index, std::integral_constant<std::size_t, d> d_ic, std::index_sequence<Is...>) const
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
        requires(sizeof...(Sets) >= 1)
    {
        using Union = NArySetOperator<SetOperator::UNION, std::decay_t<decltype(self(sets))>...>;

        return Union(self(sets)...);
    }

    template <class... Sets>
    auto intersection(const Sets&... sets)
        requires(sizeof...(Sets) >= 2)
    {
        using Intersection = NArySetOperator<SetOperator::INTERSECTION, std::decay_t<decltype(self(sets))>...>;

        return Intersection(self(sets)...);
    }

    template <class... Sets>
    auto difference(const Sets&... sets)
        requires(sizeof...(Sets) >= 2)
    {
        using Difference = NArySetOperator<SetOperator::DIFFERENCE, std::decay_t<decltype(self(sets))>...>;

        return Difference(self(sets)...);
    }

} // namespace samurai
