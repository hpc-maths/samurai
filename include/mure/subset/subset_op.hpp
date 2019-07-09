#pragma once

#include "subset_op_base.hpp"

namespace mure
{
    struct intersect_fn
    {
        template<class T>
        inline bool operator()(std::size_t /*dim*/, const T &a) const
        {
            return a;
        }

        template<class T, class... CT>
        inline bool operator()(std::size_t dim, const T &a,
                               const CT &... b) const
        {
            return (a && operator()(dim, b...));
        }
    };

    template<class... T>
    auto intersection(T &&... t)
    {
        return make_subset_operator<intersect_fn>(
            get_arg(std::forward<T>(t))...);
    }

    struct union_fn
    {
        template<class T>
        inline bool operator()(std::size_t /*dim*/, const T &a) const
        {
            return a;
        }

        template<class T, class... CT>
        inline bool operator()(std::size_t dim, const T &a,
                               const CT &... b) const
        {
            return (a || operator()(dim, b...));
        }
    };

    template<class... T>
    auto union_(T &&... t)
    {
        return make_subset_operator<union_fn>(get_arg(std::forward<T>(t))...);
    }

    struct not_fn
    {
        template<class T>
        inline bool operator()(std::size_t /*dim*/, const T &a) const
        {
            return !a;
        }

        template<class T, class... CT>
        inline bool operator()(std::size_t dim, const T &a,
                               const CT &... b) const
        {
            return (!a && operator()(dim, b...));
        }
    };

    struct difference_fn
    {
        template<class T, class... CT>
        inline bool operator()(std::size_t dim, const T &a,
                               const CT &... b) const
        {
            if (dim > 0)
                return (a || union_fn{}(dim, b...));
            else
                return (a && not_fn{}(dim, b...));
        }
    };

    template<class... T>
    auto difference(T &&... t)
    {
        return make_subset_operator<difference_fn>(
            get_arg(std::forward<T>(t))...);
    }
}