#pragma once

#include "func_node.hpp"

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
        inline bool difference(std::size_t dim, const T& a, const CT&... b)
        {
            if (dim > 0)
                return (a || union_(b...));
            else
                return (a && not_(b...));
        };
    }

    template <class... T>
    auto intersection(T&&... args)
    {
        return make_func_node([](auto, auto... a) { return detail::intersection(a...); }, std::forward<T>(args)...);
    }

    template <class... T>
    auto union_(T&&... args)
    {
        return make_func_node([](auto, auto... a) { return detail::union_(a...); }, std::forward<T>(args)...);
    }

    template <class... T>
    auto difference(T&&... args)
    {
        return make_func_node([](auto dim, auto... a) { return detail::difference(dim, a...); }, std::forward<T>(args)...);
    }
}