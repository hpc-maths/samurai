#pragma once

#include <algorithm>
#include <functional>
#include <iostream>

#include "func_node.hpp"
#include "interval.hpp"
#include "level_cell_array.hpp"
#include "tuple.hpp"

namespace mure
{
    namespace detail
    {
        template <class T>
        inline bool intersection(const T& a)
        {
                return a;
        }

        template <class T, class... CT>
        inline bool intersection(const T& a, const CT&... b)
        {
            return (a && intersection(b...));
        };

        template <class T>
        inline bool union_(const T& a)
        {
                return a;
        }

        template <class T, class... CT>
        inline bool union_(const T& a, const CT&... b)
        {
            return (a || union_(b...));
        };

        template <class T>
        inline bool not_(const T& a)
        {
                return !a;
        }

        template <class T, class... CT>
        inline bool not_(const T& a, const CT&... b)
        {
                return (!a && not_(b...));
        }

        template <class T, class... CT>
        inline bool difference(const T& a, const CT&... b)
        {
            return (a && not_(b...));
        };
    }

    template <class... T>
    auto intersection(T&&... args)
    {
        return make_func_node([](auto... a) { return detail::intersection(a...); }, std::forward<T>(args)...);
    }

    template <class... T>
    auto union_(T&&... args)
    {
        return make_func_node([](auto... a) { return detail::union_(a...); }, std::forward<T>(args)...);
    }

    template <class... T>
    auto difference(T&&... args)
    {
        return make_func_node([](auto... a) { return detail::difference(a...); }, std::forward<T>(args)...);
    }
}