// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include "subset_op_base.hpp"

namespace samurai
{

    /////////////////////////////////
    // intersection implementation //
    /////////////////////////////////

    struct intersect_fn
    {
        inline bool operator()(std::size_t /*dim*/, bool a) const
        {
            return a;
        }

        template <class... CT>
        inline bool operator()(std::size_t dim, bool a, CT&&... b) const
        {
            return (a && operator()(dim, std::forward<CT>(b)...));
        }

        inline static bool is_empty(bool a)
        {
            return a;
        }

        template <class... CT>
        inline bool is_empty(bool a, CT&&... b) const
        {
            return (a || is_empty(std::forward<CT>(b)...));
        }
    };

    template <class... T>
    auto intersection(T&&... t)
    {
        return make_subset_operator<intersect_fn>(get_arg(std::forward<T>(t))...);
    }

    //////////////////////////
    // union implementation //
    //////////////////////////

    struct union_fn
    {
        inline bool operator()(std::size_t /*dim*/, bool a) const
        {
            return a;
        }

        template <class... CT>
        inline bool operator()(std::size_t dim, bool a, const CT&... b) const
        {
            return (a || operator()(dim, b...));
        }

        inline static bool is_empty(bool a)
        {
            return a;
        }

        template <class... CT>
        inline bool is_empty(bool a, CT&&... b) const
        {
            return (a && is_empty(std::forward<CT>(b)...));
        }
    };

    template <class... T>
    auto union_(T&&... t)
    {
        return make_subset_operator<union_fn>(get_arg(std::forward<T>(t))...);
    }

    /////////////////////////////
    // negation implementation //
    /////////////////////////////

    struct not_fn
    {
        inline bool operator()(std::size_t /*dim*/, bool a) const
        {
            return !a;
        }

        template <class... CT>
        inline bool operator()(std::size_t dim, bool a, const CT&... b) const
        {
            return (!a && operator()(dim, b...));
        }
    };

    ///////////////////////////////
    // difference implementation //
    ///////////////////////////////

    struct difference_fn
    {
        template <class... CT>
        inline bool operator()(std::size_t dim, bool a, const CT&... b) const
        {
            // Since the algorithm is recursive (d = dim - 1,...,0)
            // we have to construct the union for d > 0 before
            // to find if the difference really exists.
            if (dim > 0)
            {
                return (a || union_fn{}(dim, b...));
            }
            return (a && not_fn{}(dim, b...));
        }

        template <class... CT>
        inline bool is_empty(bool a, CT&&...) const
        {
            return a;
        }
    };

    template <class... T>
    auto difference(T&&... t)
    {
        return make_subset_operator<difference_fn>(get_arg(std::forward<T>(t))...);
    }
} // namespace samurai
