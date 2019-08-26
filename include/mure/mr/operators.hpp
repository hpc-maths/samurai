#pragma once

#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include "../operators_base.hpp"
#include "prediction.hpp"

namespace mure
{
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
            field(level, i, j, k) =
                .125 * (field(level - 1, 2 * i, 2 * j, 2 * k) +
                        field(level - 1, 2 * i + 1, 2 * j, 2 * k) +
                        field(level - 1, 2 * i, 2 * j + 1, 2 * k) +
                        field(level - 1, 2 * i + 1, 2 * j + 1, 2 * k) +
                        field(level - 1, 2 * i, 2 * j + 1, 2 * k + 1) +
                        field(level - 1, 2 * i + 1, 2 * j + 1, 2 * k + 1));
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

                auto qs_i = Qs_i<1>(field, level - 1, ii, jj);
                auto qs_j = Qs_j<1>(field, level - 1, ii, jj);
                auto qs_ij = Qs_ij<1>(field, level - 1, ii, jj);

                if (!(iii & 1) and !(j & 1))
                {
                    field(level, iv, j) =
                        field(level - 1, ii, jj) + qs_i + qs_j - qs_ij;
                }
                if (!(iii & 1) and (j & 1))
                {
                    field(level, iv, j) =
                        field(level - 1, ii, jj) + qs_i - qs_j + qs_ij;
                }
                if ((iii & 1) and !(j & 1))
                {
                    field(level, iv, j) =
                        field(level - 1, ii, jj) - qs_i + qs_j + qs_ij;
                }
                if ((iii & 1) and (j & 1))
                {
                    field(level, iv, j) =
                        field(level - 1, ii, jj) - qs_i - qs_j - qs_ij;
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
            auto qs_i = xt::eval(Qs_i<1>(field, level, i, j));

            detail(level + 1, 2 * i) =
                field(level + 1, 2 * i) - (field(level, i) + qs_i);

            detail(level + 1, 2 * i + 1) =
                field(level + 1, 2 * i + 1) - (field(level, i) - qs_i);
        }

        template<class T>
        void operator()(Dim<2>, T &detail, const T &field) const
        {
            auto qs_i = Qs_i<1>(field, level, i, j);
            auto qs_j = Qs_j<1>(field, level, i, j);
            auto qs_ij = Qs_ij<1>(field, level, i, j);

            detail(level + 1, 2 * i, 2 * j) =
                field(level + 1, 2 * i, 2 * j) -
                (field(level, i, j) + qs_i + qs_j - qs_ij);

            detail(level + 1, 2 * i + 1, 2 * j) =
                field(level + 1, 2 * i + 1, 2 * j) -
                (field(level, i, j) - qs_i + qs_j + qs_ij);

            detail(level + 1, 2 * i, 2 * j + 1) =
                field(level + 1, 2 * i, 2 * j + 1) -
                (field(level, i, j) + qs_i - qs_j + qs_ij);

            detail(level + 1, 2 * i + 1, 2 * j + 1) =
                field(level + 1, 2 * i + 1, 2 * j + 1) -
                (field(level, i, j) - qs_i - qs_j - qs_ij);

            std::cout << "##################################\n";
            std::cout << level + 1 << " " << 2 * i << " " << 2 * j << "\n";
            std::cout << field(level + 1, 2 * i, 2 * j) << "\n";
            std::cout << field(level, i, j) << "\n";
            std::cout << detail(level + 1, 2 * i, 2 * j) << "\n";
            std::cout << qs_i << "\n";
            std::cout << qs_j << "\n";
            std::cout << qs_ij << "\n";
            std::cout << "##################################\n";
        }
    };

    template<class T>
    auto compute_detail(T &&detail, T &&field)
    {
        return make_field_operator_function<compute_detail_op>(
            std::forward<T>(detail), std::forward<T>(field));
    }

    /***************************
     * compute detail operator *
     ***************************/

    template<class TInterval>
    class compute_detail_op_ : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(compute_detail_op_)

        template<class T>
        void operator()(Dim<2>, T &detail, const T &field) const
        {
            for (int ii = i.start; ii < i.end; ++ii)
            {
                auto i_level = interval_t{ii, ii + 1};
                auto j_level = j;
                auto i_levelm1 = i_level / 2;
                auto j_levelm1 = j >> 1;

                auto qs_i = Qs_i<1>(field, level - 1, i_levelm1, j_levelm1);
                auto qs_j = Qs_j<1>(field, level - 1, i_levelm1, j_levelm1);
                auto qs_ij = Qs_ij<1>(field, level - 1, i_levelm1, j_levelm1);

                if (!(ii & 1) and !(j & 1))
                {
                    detail(level, i_level, j_level) =
                        field(level, i_level, j_level) -
                        (field(level - 1, i_levelm1, j_levelm1) + qs_i + qs_j -
                         qs_ij);
                }

                if ((ii & 1) and !(j & 1))
                {
                    detail(level, i_level, j_level) =
                        field(level, i_level, j_level) -
                        (field(level - 1, i_levelm1, j_levelm1) - qs_i + qs_j +
                         qs_ij);
                }
                if (!(ii & 1) and (j & 1))
                {
                    detail(level, i_level, j_level) =
                        field(level, i_level, j_level) -
                        (field(level - 1, i_levelm1, j_levelm1) + qs_i - qs_j +
                         qs_ij);
                }
                if ((ii & 1) and (j & 1))
                {
                    detail(level, i_level, j_level) =
                        field(level, i_level, j_level) -
                        (field(level - 1, i_levelm1, j_levelm1) - qs_i - qs_j -
                         qs_ij);
                }
            }
        }
    };

    template<class T>
    auto compute_detail_(T &&detail, T &&field)
    {
        return make_field_operator_function<compute_detail_op_>(
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

    /*******************************
     * compute max detail operator *
     *******************************/

    template<class TInterval>
    class compute_max_detail_op_ : public field_operator_base<TInterval> {
      public:
        INIT_OPERATOR(compute_max_detail_op_)

        template<class T, class U>
        void operator()(Dim<2>, const U &detail, T &max_detail) const
        {
            max_detail[level] = std::max(
                max_detail[level], xt::amax(xt::abs(detail(level, i, j)))[0]);
        }
    };

    template<class T, class U>
    auto compute_max_detail_(U &&detail, T &&max_detail)
    {
        return make_field_operator_function<compute_max_detail_op_>(
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
            // auto mask = (0.25 *
            //              (xt::abs(detail(level + 1, 2 * i, 2 * j)) +
            //               xt::abs(detail(level + 1, 2 * i + 1, 2 * j)) +
            //               xt::abs(detail(level + 1, 2 * i, 2 * j + 1)) +
            //               xt::abs(detail(level + 1, 2 * i + 1, 2 * j + 1))) /
            //              max_detail[level + 1]) < eps;
            auto mask =
                (0.25 * (xt::abs(detail(level + 1, 2 * i, 2 * j)) +
                         xt::abs(detail(level + 1, 2 * i + 1, 2 * j)) +
                         xt::abs(detail(level + 1, 2 * i, 2 * j + 1)) +
                         xt::abs(detail(level + 1, 2 * i + 1, 2 * j + 1)))) <
                eps;
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
            // auto mask =
            //     (xt::abs(detail(level, i, j)) / max_detail[level]) >= 2 *
            //     eps;
            std::cout << eps << " " << xt::abs(detail(level, i, j)) << "\n";
            auto mask = xt::abs(detail(level, i, j)) >= 4 * eps;
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