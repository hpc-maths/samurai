#pragma once

#include <xtensor/xfixed.hpp>

#include "field_expression.hpp"
#include "utils.hpp"

namespace mure
{
    template<template<class T> class OP, class... CT>
    class field_operator_function
        : public field_expression<field_operator_function<OP, CT...>> {
      public:
        static constexpr std::size_t dim = detail::compute_dim<CT...>();

        field_operator_function(CT &&... e) : m_e{std::forward<CT>(e)...}
        {}

        template<class interval_t, class... index_t>
        auto operator()(std::size_t level, interval_t i, index_t... index) const
        {
            OP<interval_t> op(level, i, index...);
            return apply(op);
        }

        template<class interval_t, class coord_index_t>
        auto operator()(
            std::size_t level, interval_t i,
            xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> index) const
        {
            OP<interval_t> op(level, i, index);
            return apply(op);
        }

      private:
        template<class interval_t>
        auto apply(OP<interval_t> &op) const
        {
            return apply_impl(std::make_index_sequence<sizeof...(CT)>(), op);
        }

        template<std::size_t... I, class interval_t>
        auto apply_impl(std::index_sequence<I...>, OP<interval_t> &op) const
        {
            return op(std::integral_constant<std::size_t, dim>{},
                      std::get<I>(m_e)...);
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
        double dx;

      protected:
        template<std::size_t dim>
        field_operator_base(
            std::size_t level, interval_t interval,
            xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> index)
            : level{level}, i{interval}, dx{1. / (1 << level)}
        {
            if (dim > 0)
                j = index[0];
            if (dim > 1)
                k = index[1];
        }

        field_operator_base(std::size_t level, interval_t interval)
            : level{level}, i{interval}, dx{1. / (1 << level)}
        {}

        field_operator_base(std::size_t level, interval_t interval,
                            coord_index_t j_)
            : level{level}, dx{1. / (1 << level)}, i{interval}, j{j_}
        {}

        field_operator_base(std::size_t level, interval_t interval,
                            coord_index_t j_, coord_index_t k_)
            : level{level}, dx{1. / (1 << level)}, i{interval}, j{j_}, k{k_}
        {}
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
    using base::dx;                                                            \
                                                                               \
    template<std::size_t dim>                                                  \
    NAME(std::size_t level, interval_t interval,                               \
         xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> index)              \
        : base(level, interval, index)                                         \
    {}                                                                         \
    template<class... index_t>                                                 \
    NAME(std::size_t level, interval_t interval, index_t... index)             \
        : base(level, interval, index...)                                      \
    {}

    /***********************
     * projection operator *
     ***********************/

    template<class TInterval>
    class projection_op_ : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(projection_op_)

        template<class T>
        void operator()(Dim<1>, T &field) const
        {
            field(level, i) =
                .5 * (field(level + 1, 2 * i) + field(level + 1, 2 * i + 1));
        }

        template<class T>
        void operator()(Dim<2>, T &field) const
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

    /***********************
     * prediction operator *
     ***********************/

    template<class TInterval>
    class prediction_op : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(prediction_op)

        template<class T>
        void operator()(Dim<2>, T &field) const
        {
            for (coord_index_t iii = i.start; iii < i.end; ++iii)
            {
                auto tmp = iii >> 1;
                auto jj = j >> 1;
                interval_t ii{tmp, tmp + 1};
                interval_t iv{iii, iii + 1};
                if (!(iii & 1) and !(j & 1))
                {
                    field(level, iv, j) =
                        field(level - 1, ii, jj) -
                        1. / 8 *
                            (field(level - 1, ii + 1, jj) -
                             field(level - 1, ii - 1, jj)) -
                        1. / 8 *
                            (field(level - 1, ii, jj + 1) -
                             field(level - 1, ii, jj - 1)) -
                        1. / 64 *
                            (field(level - 1, ii + 1, jj + 1) -
                             field(level - 1, ii - 1, jj + 1) +
                             field(level - 1, ii - 1, jj - 1) -
                             field(level - 1, ii + 1, jj - 1));
                }
                if (!(iii & 1) and (j & 1))
                {
                    field(level, iv, j) =
                        field(level - 1, ii, jj) -
                        1. / 8 *
                            (field(level - 1, ii + 1, jj) -
                             field(level - 1, ii - 1, jj)) +
                        1. / 8 *
                            (field(level - 1, ii, jj + 1) -
                             field(level - 1, ii, jj - 1)) +
                        1. / 64 *
                            (field(level - 1, ii + 1, jj + 1) -
                             field(level - 1, ii - 1, jj + 1) +
                             field(level - 1, ii - 1, jj - 1) -
                             field(level - 1, ii + 1, jj - 1));
                }
                if ((iii & 1) and !(j & 1))
                {
                    field(level, iv, j) =
                        (field(level - 1, ii, jj) +
                         1. / 8 *
                             (field(level - 1, ii + 1, jj) -
                              field(level - 1, ii - 1, jj)) -
                         1. / 8 *
                             (field(level - 1, ii, jj + 1) -
                              field(level - 1, ii, jj - 1)) +
                         1. / 64 *
                             (field(level - 1, ii + 1, jj + 1) -
                              field(level - 1, ii - 1, jj + 1) +
                              field(level - 1, ii - 1, jj - 1) -
                              field(level - 1, ii + 1, jj - 1)));
                }
                if ((iii & 1) and (j & 1))
                {
                    field(level, iv, j) =
                        (field(level - 1, ii, jj) +
                         1. / 8 *
                             (field(level - 1, ii + 1, jj) -
                              field(level - 1, ii - 1, jj)) +
                         1. / 8 *
                             (field(level - 1, ii, jj + 1) -
                              field(level - 1, ii, jj - 1)) -
                         1. / 64 *
                             (field(level - 1, ii + 1, jj + 1) -
                              field(level - 1, ii - 1, jj + 1) +
                              field(level - 1, ii - 1, jj - 1) -
                              field(level - 1, ii + 1, jj - 1)));
                }
            }
        }
    };

    template<class T>
    auto prediction(T &&field)
    {
        return make_field_operator_function<prediction_op>(
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
        void operator()(Dim<1>, T &field) const
        {
            xt::xtensor<bool, 1> mask =
                field(level + 1, 2 * i) | field(level + 1, 2 * i + 1);

            xt::masked_view(field(level + 1, 2 * i), mask) = true;
            xt::masked_view(field(level + 1, 2 * i + 1), mask) = true;

            xt::masked_view(field(level, i), mask) = true;
        }

        template<class T>
        void operator()(Dim<2>, T &field) const
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

            xt::masked_view(field(level, i, j), mask) = true;
        }
    };

    template<class T>
    auto maximum(T &&field)
    {
        return make_field_operator_function<maximum_op>(std::forward<T>(field));
    }

    /*****************
     * copy operator *
     *****************/

    template<class TInterval>
    class copy_op : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(copy_op)

        template<class T>
        void operator()(Dim<1>, T &dest, const T &src) const
        {
            dest(level, i) = src(level, i);
        }

        template<class T>
        void operator()(Dim<2>, T &dest, const T &src) const
        {
            dest(level, i, j) = src(level, i, j);
        }

        template<class T>
        void operator()(Dim<3>, T &dest, const T &src) const
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
        void operator()(Dim<1>, T &detail, const T &field) const
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
        void operator()(Dim<2>, T &detail, const T &field) const
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
        void operator()(Dim<1>, const U &detail, T &max_detail) const
        {
            auto ii = 2 * i;
            ii.step = 1;
            max_detail[level + 1] =
                std::max(max_detail[level + 1],
                         xt::amax(xt::abs(detail(level + 1, ii)))[0]);
        }

        template<class T, class U>
        void operator()(Dim<2>, const U &detail, T &max_detail) const
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
        void operator()(Dim<1>, T &keep, const U &detail, const V &max_detail,
                        double eps) const
        {
            auto mask = (.5 *
                         (xt::abs(detail(level + 1, 2 * i)) +
                          xt::abs(detail(level + 1, 2 * i + 1))) /
                         max_detail[level + 1]) < eps;
            xt::masked_view(keep(level + 1, 2 * i), mask) = false;
            xt::masked_view(keep(level + 1, 2 * i + 1), mask) = false;
        }

        template<class T, class U, class V>
        void operator()(Dim<2>, T &keep, const U &detail, const V &max_detail,
                        double eps) const
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

    /***********************
     * to_refine operator *
     ***********************/

    template<class TInterval>
    class to_refine_op : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(to_refine_op)

        template<class T, class U, class V>
        void operator()(Dim<1>, T &refine, const U &detail, const V &max_detail,
                        double eps) const
        {
            auto mask = (.5 *
                         (xt::abs(detail(level + 1, 2 * i)) +
                          xt::abs(detail(level + 1, 2 * i + 1))) /
                         max_detail[level + 1]) >= 2 * eps;
            xt::masked_view(refine(level + 1, 2 * i), mask) = true;
            xt::masked_view(refine(level + 1, 2 * i + 1), mask) = true;
        }

        template<class T, class U, class V>
        void operator()(Dim<2>, T &refine, const U &detail, const V &max_detail,
                        double eps) const
        {
            auto mask =
                (xt::abs(detail(level, i, j)) / max_detail[level]) >= 2 * eps;
            xt::masked_view(refine(level, i, j), mask) = true;
        }
    };

    template<class... CT>
    auto to_refine(CT &&... e)
    {
        return make_field_operator_function<to_refine_op>(
            std::forward<CT>(e)...);
    }

    /***********************
     * apply_expr operator *
     ***********************/

    template<class TInterval>
    class apply_expr_op : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(apply_expr_op)

        template<class T, class E>
        void operator()(Dim<1>, T &field, const field_expression<E> &e) const
        {
            field(level, i) = e.derived_cast()(level, i);
        }

        template<class T, class E>
        void operator()(Dim<2>, T &field, const field_expression<E> &e) const
        {
            field(level, i, j) = e.derived_cast()(level, i, j);
        }

        template<class T, class E>
        void operator()(Dim<3>, T &field, const field_expression<E> &e) const
        {
            field(level, i, j, k) = e.derived_cast()(level, i, j, k);
        }
    };

    template<class... CT>
    auto apply_expr(CT &&... e)
    {
        return make_field_operator_function<apply_expr_op>(
            std::forward<CT>(e)...);
    }
}