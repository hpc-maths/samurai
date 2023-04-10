// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include "../operators_base.hpp"

namespace samurai
{
    /////////////////////////
    // projection operator //
    /////////////////////////

    template <class TInterval>
    class projection_op_ : public field_operator_base<TInterval>
    {
      public:

        INIT_OPERATOR(projection_op_)

        template <class T1, class T2>
        inline void operator()(Dim<1>, T1& dest, const T2& src) const
        {
            dest(level, i) = .5 * (src(level + 1, 2 * i) + src(level + 1, 2 * i + 1));
        }

        template <class T1, class T2>
        inline void operator()(Dim<2>, T1& dest, const T2& src) const
        {
            dest(level, i, j) = .25
                              * (src(level + 1, 2 * i, 2 * j) + src(level + 1, 2 * i, 2 * j + 1) + src(level + 1, 2 * i + 1, 2 * j)
                                 + src(level + 1, 2 * i + 1, 2 * j + 1));
        }

        template <class T1, class T2>
        inline void operator()(Dim<3>, T1& dest, const T2& src) const
        {
            dest(level, i, j, k) = .125
                                 * (src(level + 1, 2 * i, 2 * j, 2 * k) + src(level + 1, 2 * i + 1, 2 * j, 2 * k)
                                    + src(level + 1, 2 * i, 2 * j + 1, 2 * k) + src(level + 1, 2 * i + 1, 2 * j + 1, 2 * k)
                                    + src(level + 1, 2 * i, 2 * j, 2 * k + 1) + src(level + 1, 2 * i + 1, 2 * j, 2 * k + 1)
                                    + src(level + 1, 2 * i, 2 * j + 1, 2 * k + 1) + src(level + 1, 2 * i + 1, 2 * j + 1, 2 * k + 1));
        }
    };

    template <class TInterval>
    class variadic_projection_op_ : public field_operator_base<TInterval>
    {
      public:

        INIT_OPERATOR(variadic_projection_op_)

        template <std::size_t d>
        inline void operator()(Dim<d>) const
        {
        }

        template <class Head, class... Tail>
        inline void operator()(Dim<1>, Head& source, Tail&... sources) const
        {
            projection_op_<interval_t>(level, i)(Dim<1>{}, source, source);
            this->operator()(Dim<1>{}, sources...);
        }

        template <class Head, class... Tail>
        inline void operator()(Dim<2>, Head& source, Tail&... sources) const
        {
            projection_op_<interval_t>(level, i, j)(Dim<2>{}, source, source);
            this->operator()(Dim<2>{}, sources...);
        }

        template <class Head, class... Tail>
        inline void operator()(Dim<3>, Head& source, Tail&... sources) const
        {
            projection_op_<interval_t>(level, i, j, k)(Dim<3>{}, source, source);
            this->operator()(Dim<3>{}, sources...);
        }
    };

    template <class T>
    inline auto projection(T&& field)
    {
        return make_field_operator_function<projection_op_>(std::forward<T>(field), std::forward<T>(field));
    }

    template <class... T>
    inline auto variadic_projection(T&&... fields)
    {
        return make_field_operator_function<variadic_projection_op_>(std::forward<T>(fields)...);
    }

    template <class T1, class T2>
    inline auto projection(T1&& field_dest, T2&& field_src)
    {
        return make_field_operator_function<projection_op_>(std::forward<T1>(field_dest), std::forward<T2>(field_src));
    }
}
