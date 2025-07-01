// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <tuple>

#include "set_base.hpp"

namespace samurai
{

    ////////////////////////////////////////////////////////////////////////
    //// Generic Subset class
    ////////////////////////////////////////////////////////////////////////

    template <class Op, Set_concept... Sets>
    class Subset;

    template <class Op, Set_concept FirstSet, Set_concept... OtherSets>
    struct SetTraits<Subset<Op, FirstSet, OtherSets...>>
    {
        using interval_t = typename SetTraits<FirstSet>::interval_t;

        static constexpr std::size_t dim = SetTraits<FirstSet>::dim;
    };

    template <class Op, Set_concept... Sets>
    class Subset : public SetBase<Subset<Op, Sets...>>
    {
        using SetTuple = std::tuple<Sets...>;

        Subset(Op&& op, Sets&&... sets)
            : m_operator(std::move(op))
            , m_sets(std::forward<Sets>(sets)...)
            , m_level(compute_max(s.level()...))
            , m_ref_level(compute_max(sets.ref_level()...))
        {
        }

        std::size_t min_level() const
        {
            return m_min_level;
        }

        std::size_t level() const
        {
            return m_level;
        }

        std::size_t ref_level() const
        {
            return m_ref_level;
        }

        bool exists() const
        {
            return std::apply(
                [this](auto&&... args)
                {
                    return m_operator.exist(args...);
                },
                m_sets);
        }

        bool empty() const
        {
            return std::apply(
                [this](auto&&... args)
                {
                    return m_operator.empty(args...);
                },
                m_sets);
        }

      private:

        Op m_operator;
        SetTuple m_sets;
        std::size_t m_level;
        std::size_t m_min_level;
        std::size_t m_ref_level;
    };
}
