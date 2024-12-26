// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <concepts>
#include <type_traits>

namespace samurai
{
    template <std::size_t dim, class interval_t>
    class LevelCellArray;
}

namespace samurai::experimental
{
    template <class Operator, class S1, class... S>
    class SetOp;

    template <class iterator>
    class IntervalVector;

    template <class T>
    struct is_setop : std::false_type
    {
    };

    template <class... T>
    struct is_setop<SetOp<T...>> : std::true_type
    {
    };

    template <class T>
    constexpr bool is_setop_v{is_setop<std::decay_t<T>>::value};

    template <typename T>
    concept IsSetOp = is_setop_v<T>;

    template <typename T>
    concept IsIntervalVector = std::is_base_of_v<IntervalVector<typename std::decay_t<T>::iterator_t>, std::decay_t<T>>;

    template <typename T>
    concept IsLCA = std::same_as<LevelCellArray<T::dim, typename T::interval_t>, T>;
}