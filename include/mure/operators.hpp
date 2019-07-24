#pragma once

#include <xtensor/xfixed.hpp>

#include "utils.hpp"

namespace mure
{
    template<template<class T> class OP, class... CT>
    class field_operator_function {
      public:
        static constexpr std::size_t dim = detail::compute_dim<CT...>();

        field_operator_function(CT &&... e) : m_e{std::forward<CT>(e)...}
        {}

        template<class interval_t, class index_t>
        void operator()(std::size_t level, index_t index, interval_t i)
        {
            OP<interval_t> op(level, index, i);
            apply(op);
        }

      private:
        template<class interval_t>
        void apply(OP<interval_t> &op) const
        {
            apply_impl(std::make_index_sequence<sizeof...(CT)>(), op);
        }

        template<std::size_t... I, class interval_t>
        void apply_impl(std::index_sequence<I...>, OP<interval_t> &op) const
        {
            op(std::get<I>(m_e)..., std::integral_constant<std::size_t, dim>{});
        }

        std::tuple<CT...> m_e;
    };

    template<template<class T> class OP, class... CT>
    auto make_field_operator_function(CT &&... e)
    {
        return field_operator_function<OP, CT...>(std::forward<CT>(e)...);
    }

    template<class TInterval>
    class field_operator_base {
      public:
        using interval_t = TInterval;
        using coord_index_t = typename interval_t::coord_index_t;

        std::size_t level;
        interval_t i;
        coord_index_t j, k;

      protected:
        template<std::size_t dim>
        field_operator_base(
            std::size_t level,
            xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> index,
            interval_t interval)
            : level{level}, i{interval}
        {
            if (dim > 0)
                j = index[0];
            if (dim > 1)
                k = index[1];
        }
    };

#define INIT_OPERATOR(NAME)                                                    \
    using interval_t = TInterval;                                              \
    using coord_index_t = typename interval_t::coord_index_t;                  \
                                                                               \
    using base = field_operator_base<interval_t>;                              \
    using base::i;                                                             \
    using base::j;                                                             \
    using base::k;                                                             \
    using base::level;                                                         \
                                                                               \
    template<std::size_t dim>                                                  \
    NAME(std::size_t level,                                                    \
         xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> index,              \
         interval_t interval)                                                  \
        : base(level, index, interval)                                         \
    {}

    /***********************
     * projection operator *
     ***********************/

    template<class TInterval>
    class projection_op_ : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(projection_op_)

        template<class T>
        void operator()(T &field, Dim<1>) const
        {
            field(level, i) =
                .5 * (field(level + 1, 2 * i) + field(level + 1, 2 * i + 1));
        }

        template<class T>
        void operator()(T &field, Dim<2>) const
        {
            field(level, i, j) = .25 * (field(level + 1, 2 * i, 2 * j) +
                                        field(level + 1, 2 * i, 2 * j + 1) +
                                        field(level + 1, 2 * i + 1, 2 * j) +
                                        field(level + 1, 2 * i + 1, 2 * j + 1));
        }

        template<class T>
        void operator()(T &field, Dim<3>) const
        {
            field(level - 1, i, j, k) =
                .125 * (field(level, 2 * i, 2 * j, 2 * k) +
                        field(level, 2 * i + 1, 2 * j, 2 * k) +
                        field(level, 2 * i, 2 * j + 1, 2 * k) +
                        field(level, 2 * i + 1, 2 * j + 1, 2 * k) +
                        field(level, 2 * i, 2 * j + 1, 2 * k + 1) +
                        field(level, 2 * i + 1, 2 * j + 1, 2 * k + 1));
        }
    };

    template<class T>
    auto projection(T &&field)
    {
        return make_field_operator_function<projection_op_>(
            std::forward<T>(field));
    }

    /********************
     * maximum operator *
     ********************/

    template<class TInterval>
    class maximum_op : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(maximum_op)

        template<class T>
        void operator()(T &field, Dim<1>) const
        {
            xt::xtensor<bool, 1> mask =
                field(level + 1, 2 * i) | field(level + 1, 2 * i + 1);

            xt::masked_view(field(level + 1, 2 * i), mask) = true;
            xt::masked_view(field(level + 1, 2 * i + 1), mask) = true;

            field(level, i) =
                field(level + 1, 2 * i) | field(level + 1, 2 * i + 1);
        }

        template<class T>
        void operator()(T &field, Dim<2>) const
        {
            xt::xtensor<bool, 1> mask = field(level + 1, 2 * i, 2 * j) |
                                        field(level + 1, 2 * i + 1, 2 * j) |
                                        field(level + 1, 2 * i, 2 * j + 1) |
                                        field(level + 1, 2 * i + 1, 2 * j + 1);

            xt::masked_view(field(level + 1, 2 * i, 2 * j), mask) = true;
            xt::masked_view(field(level + 1, 2 * i + 1, 2 * j), mask) = true;
            xt::masked_view(field(level + 1, 2 * i, 2 * j + 1), mask) = true;
            xt::masked_view(field(level + 1, 2 * i + 1, 2 * j + 1), mask) =
                true;

            field(level, i, j) = field(level + 1, 2 * i, 2 * j) |
                                 field(level + 1, 2 * i + 1, 2 * j) |
                                 field(level + 1, 2 * i, 2 * j + 1) |
                                 field(level + 1, 2 * i + 1, 2 * j + 1);
        }
    };

    template<class T>
    auto maximum(T &&field)
    {
        return make_field_operator_function<maximum_op>(std::forward<T>(field));
    }

    /*******************
     * graded operator *
     *******************/

    template<class TInterval>
    class graded_op : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(graded_op)

        template<class T>
        void operator()(T &field, Dim<1>) const
        {
            field(level, i + 1) |= field(level, i);
            field(level, i - 1) |= field(level, i);
        }

        template<class T>
        void operator()(T &field, Dim<2>) const
        {
            coord_index_t ii_start = -1, ii_end = 1;
            coord_index_t jj_start = -1, jj_end = 1;

            for (coord_index_t jj = jj_start; jj <= jj_end; ++jj)
                for (coord_index_t ii = ii_start; ii <= ii_end; ++ii)
                    field(level, i + ii, j + jj) |= field(level, i, j);
        }
    };

    template<class T>
    auto graded(T &&field)
    {
        return make_field_operator_function<graded_op>(std::forward<T>(field));
    }

    /*****************
     * copy operator *
     *****************/

    template<class TInterval>
    class copy_op : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(copy_op)

        template<class T>
        void operator()(T &dest, const T &src, Dim<1>) const
        {
            dest(level, i) = src(level, i);
        }

        template<class T>
        void operator()(T &dest, const T &src, Dim<2>) const
        {
            dest(level, i, j) = src(level, i, j);
        }

        template<class T>
        void operator()(T &dest, const T &src, Dim<3>) const
        {
            dest(level, i, j, k) = src(level, i, j, k);
        }
    };

    template<class T>
    auto copy(T &&dest, T &&src)
    {
        return make_field_operator_function<copy_op>(std::forward<T>(dest),
                                                     std::forward<T>(src));
    }

    /***************************
     * compute detail operator *
     ***************************/

    template<class TInterval>
    class compute_detail_op : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(compute_detail_op)

        template<class T>
        void operator()(T &detail, const T &field, Dim<1>) const
        {
            detail(level + 1, 2 * i) =
                field(level + 1, 2 * i) -
                (field(level, i) -
                 1. / 8 * (field(level, i + 1) - field(level, i - 1)));
            detail(level + 1, 2 * i + 1) =
                field(level + 1, 2 * i + 1) -
                (field(level, i) +
                 1. / 8 * (field(level, i + 1) - field(level, i - 1)));
        }

        template<class T>
        void operator()(T &detail, const T &field, Dim<2>) const
        {
            detail(level + 1, 2 * i, 2 * j) =
                field(level + 1, 2 * i, 2 * j) -
                (field(level, i, j) -
                 1. / 8 * (field(level, i + 1, j) - field(level, i - 1, j)) -
                 1. / 8 * (field(level, i, j + 1) - field(level, i, j - 1)) -
                 1. / 64 *
                     (field(level, i + 1, j + 1) - field(level, i - 1, j + 1) +
                      field(level, i - 1, j - 1) - field(level, i + 1, j - 1)));

            detail(level + 1, 2 * i, 2 * j + 1) =
                field(level + 1, 2 * i, 2 * j + 1) -
                (field(level, i, j) -
                 1. / 8 * (field(level, i + 1, j) - field(level, i - 1, j)) +
                 1. / 8 * (field(level, i, j + 1) - field(level, i, j - 1)) +
                 1. / 64 *
                     (field(level, i + 1, j + 1) - field(level, i - 1, j + 1) +
                      field(level, i - 1, j - 1) - field(level, i + 1, j - 1)));

            detail(level + 1, 2 * i + 1, 2 * j) =
                field(level + 1, 2 * i + 1, 2 * j) -
                (field(level, i, j) +
                 1. / 8 * (field(level, i + 1, j) - field(level, i - 1, j)) -
                 1. / 8 * (field(level, i, j + 1) - field(level, i, j - 1)) +
                 1. / 64 *
                     (field(level, i + 1, j + 1) - field(level, i - 1, j + 1) +
                      field(level, i - 1, j - 1) - field(level, i + 1, j - 1)));

            detail(level + 1, 2 * i + 1, 2 * j + 1) =
                field(level + 1, 2 * i + 1, 2 * j + 1) -
                (field(level, i, j) +
                 1. / 8 * (field(level, i + 1, j) - field(level, i - 1, j)) +
                 1. / 8 * (field(level, i, j + 1) - field(level, i, j - 1)) -
                 1. / 64 *
                     (field(level, i + 1, j + 1) - field(level, i - 1, j + 1) +
                      field(level, i - 1, j - 1) - field(level, i + 1, j - 1)));
        }
    };

    template<class T>
    auto compute_detail(T &&detail, T &&field)
    {
        return make_field_operator_function<compute_detail_op>(
            std::forward<T>(detail), std::forward<T>(field));
    }

    /*******************************
     * compute max detail operator *
     *******************************/

    template<class TInterval>
    class compute_max_detail_op : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(compute_max_detail_op)

        template<class T, class U>
        void operator()(const U &detail, T &max_detail, Dim<1>) const
        {
            auto ii = 2 * i;
            ii.step = 1;
            max_detail[level + 1] =
                std::max(max_detail[level + 1],
                         xt::amax(xt::abs(detail(level + 1, ii)))[0]);
        }

        template<class T, class U>
        void operator()(const U &detail, T &max_detail, Dim<2>) const
        {
            auto ii = 2 * i;
            ii.step = 1;
            max_detail[level + 1] =
                std::max(max_detail[level + 1],
                         xt::amax(xt::maximum(
                             xt::abs(detail(level + 1, ii, 2 * j)),
                             xt::abs(detail(level + 1, ii, 2 * j + 1))))[0]);
        }
    };

    template<class T, class U>
    auto compute_max_detail(U &&detail, T &&max_detail)
    {
        return make_field_operator_function<compute_max_detail_op>(
            std::forward<U>(detail), std::forward<T>(max_detail));
    }

    /***********************
     * to_coarsen operator *
     ***********************/

    template<class TInterval>
    class to_coarsen_op : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(to_coarsen_op)

        template<class T, class U, class V>
        void operator()(T &keep, const U &detail, const V &max_detail,
                        double eps, Dim<1>) const
        {
            auto mask = (.5 *
                         (xt::abs(detail(level + 1, 2 * i)) +
                          xt::abs(detail(level + 1, 2 * i + 1))) /
                         max_detail[level + 1]) < eps;
            xt::masked_view(keep(level + 1, 2 * i), mask) = false;
            xt::masked_view(keep(level + 1, 2 * i + 1), mask) = false;
        }

        template<class T, class U, class V>
        void operator()(T &keep, const U &detail, const V &max_detail,
                        double eps, Dim<2>) const
        {
            auto mask = (0.25 *
                         (xt::abs(detail(level + 1, 2 * i, 2 * j)) +
                          xt::abs(detail(level + 1, 2 * i + 1, 2 * j)) +
                          xt::abs(detail(level + 1, 2 * i, 2 * j + 1)) +
                          xt::abs(detail(level + 1, 2 * i + 1, 2 * j + 1))) /
                         max_detail[level + 1]) < eps;
            xt::masked_view(keep(level + 1, 2 * i, 2 * j), mask) = false;
            xt::masked_view(keep(level + 1, 2 * i + 1, 2 * j), mask) = false;
            xt::masked_view(keep(level + 1, 2 * i, 2 * j + 1), mask) = false;
            xt::masked_view(keep(level + 1, 2 * i + 1, 2 * j + 1), mask) =
                false;
        }
    };

    template<class... CT>
    auto to_coarsen(CT &&... e)
    {
        return make_field_operator_function<to_coarsen_op>(
            std::forward<CT>(e)...);
    }

    /******************
     * clean operator *
     ******************/

    template<class TInterval>
    class clean_op : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(clean_op)

        template<class T>
        void operator()(T &field, Dim<1>) const
        {
            field(level, i) = false;
        }

        template<class T>
        void operator()(T &field, Dim<2>) const
        {
            field(level, i, j) = false;
        }

        template<class T>
        void operator()(T &field, Dim<3>) const
        {
            field(level, i, j, k) = false;
        }
    };

    template<class... CT>
    auto clean(CT &&... e)
    {
        return make_field_operator_function<clean_op>(std::forward<CT>(e)...);
    }

    /********************
     * to_keep operator *
     ********************/

    template<class TInterval>
    class to_keep_op : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(to_keep_op)

        template<class T>
        void operator()(T &field, Dim<1>) const
        {
            xt::xtensor<bool, 1> mask =
                field(level, 2 * i) | field(level, 2 * i + 1);

            xt::masked_view(field(level - 1, i), !mask) = true;
        }

        template<class T>
        void operator()(T &field, Dim<2>) const
        {
            xt::xtensor<bool, 1> mask = field(level, 2 * i, 2 * j) |
                                        field(level, 2 * i + 1, 2 * j) |
                                        field(level, 2 * i, 2 * j + 1) |
                                        field(level, 2 * i + 1, 2 * j + 1);

            xt::masked_view(field(level - 1, i, j), !mask) = true;
        }
    };

    template<class... CT>
    auto to_keep(CT &&... e)
    {
        return make_field_operator_function<to_keep_op>(std::forward<CT>(e)...);
    }
}