// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <xtensor/xmasked_view.hpp>
#include <xtensor/xview.hpp>

#include "../cell_flag.hpp"
#include "../operators_base.hpp"

namespace samurai
{

    template <class TInterval>
    class to_coarsen_mr_op : public field_operator_base<TInterval>
    {
      public:

        INIT_OPERATOR(to_coarsen_mr_op)

        template <class T1, class T2>
        // inline void operator()(Dim<1>, const T1& detail, const T3&
        // max_detail, T2 &tag, double eps, std::size_t min_lev) const
        inline void operator()(Dim<1>, const T1& detail, T2& tag, double eps, std::size_t min_lev) const

        {
            constexpr auto size    = T1::size;
            std::size_t fine_level = level + 1;

            if (fine_level > min_lev)
            {
                // auto maxd = xt::view(max_detail, level);

                if constexpr (size == 1)
                {
                    // auto mask = xt::abs(detail(level, 2*i))/maxd < eps;
                    auto mask = xt::abs(detail(fine_level, 2 * i)) < eps; // NO normalization

                    xt::masked_view(tag(fine_level, 2 * i), mask)     = static_cast<int>(CellFlag::coarsen);
                    xt::masked_view(tag(fine_level, 2 * i + 1), mask) = static_cast<int>(CellFlag::coarsen);
                }
                else
                {
                    // auto mask = xt::sum((xt::abs(detail(level, 2*i))/maxd <
                    // eps), {1}) > (size-1);
                    auto mask = xt::sum((xt::abs(detail(fine_level, 2 * i)) < eps), {detail.is_soa ? 0 : 1}) > (size - 1); // No
                                                                                                                           // normalization

                    xt::masked_view(tag(fine_level, 2 * i), mask)     = static_cast<int>(CellFlag::coarsen);
                    xt::masked_view(tag(fine_level, 2 * i + 1), mask) = static_cast<int>(CellFlag::coarsen);
                }
            }
        }

        template <class T1, class T2>
        inline void operator()(Dim<2>, const T1& detail, T2& tag, double eps, std::size_t min_lev) const
        {
            constexpr auto size    = T1::size;
            std::size_t fine_level = level + 1;

            if (fine_level > min_lev)
            {
                // auto maxd = xt::view(max_detail, level);

                if constexpr (size == 1)
                {
                    // auto mask = (xt::abs(detail(level, 2*i  ,   2*j))/maxd <
                    // eps) &&
                    //             (xt::abs(detail(level, 2*i+1,   2*j))/maxd <
                    //             eps) && (xt::abs(detail(level, 2*i  ,
                    //             2*j+1))/maxd < eps) && (xt::abs(detail(level,
                    //             2*i+1, 2*j+1))/maxd < eps);
                    auto mask = (xt::abs(detail(fine_level, 2 * i, 2 * j)) < eps) && (xt::abs(detail(fine_level, 2 * i + 1, 2 * j)) < eps)
                             && (xt::abs(detail(fine_level, 2 * i, 2 * j + 1)) < eps)
                             && (xt::abs(detail(fine_level, 2 * i + 1, 2 * j + 1)) < eps);

                    xt::masked_view(tag(fine_level, 2 * i, 2 * j), mask)         = static_cast<int>(CellFlag::coarsen);
                    xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j), mask)     = static_cast<int>(CellFlag::coarsen);
                    xt::masked_view(tag(fine_level, 2 * i, 2 * j + 1), mask)     = static_cast<int>(CellFlag::coarsen);
                    xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j + 1), mask) = static_cast<int>(CellFlag::coarsen);
                }
                else
                {
                    // auto mask = xt::sum((xt::abs(detail(level, 2*i  ,
                    // 2*j))/maxd < eps) &&
                    //                     (xt::abs(detail(level, 2*i+1,
                    //                     2*j))/maxd < eps) &&
                    //                     (xt::abs(detail(level, 2*i  ,
                    //                     2*j+1))/maxd < eps) &&
                    //                     (xt::abs(detail(level, 2*i+1,
                    //                     2*j+1))/maxd < eps), {1}) > (size-1);
                    auto mask = xt::sum((xt::abs(detail(fine_level, 2 * i, 2 * j)) < eps)
                                            && (xt::abs(detail(fine_level, 2 * i + 1, 2 * j)) < eps)
                                            && (xt::abs(detail(fine_level, 2 * i, 2 * j + 1)) < eps)
                                            && (xt::abs(detail(fine_level, 2 * i + 1, 2 * j + 1)) < eps),
                                        {detail.is_soa ? 0 : 1})
                              > (size - 1);

                    xt::masked_view(tag(fine_level, 2 * i, 2 * j), mask)         = static_cast<int>(CellFlag::coarsen);
                    xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j), mask)     = static_cast<int>(CellFlag::coarsen);
                    xt::masked_view(tag(fine_level, 2 * i, 2 * j + 1), mask)     = static_cast<int>(CellFlag::coarsen);
                    xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j + 1), mask) = static_cast<int>(CellFlag::coarsen);
                }
            }
        }

        template <class T1, class T2>
        inline void operator()(Dim<3>, const T1& detail, T2& tag, double eps, std::size_t min_lev) const
        {
            constexpr auto size    = T1::size;
            std::size_t fine_level = level + 1;

            if (fine_level > min_lev)
            {
                // auto maxd = xt::view(max_detail, level);

                if constexpr (size == 1)
                {
                    // auto mask = (xt::abs(detail(level, 2*i  ,   2*j))/maxd <
                    // eps) and
                    //             (xt::abs(detail(level, 2*i+1,   2*j))/maxd <
                    //             eps) and (xt::abs(detail(level, 2*i  ,
                    //             2*j+1))/maxd < eps) and
                    //             (xt::abs(detail(level, 2*i+1, 2*j+1))/maxd <
                    //             eps);
                    auto mask = (xt::abs(detail(fine_level, 2 * i, 2 * j, 2 * k)) < eps)
                            and (xt::abs(detail(fine_level, 2 * i + 1, 2 * j, 2 * k)) < eps)
                            and (xt::abs(detail(fine_level, 2 * i, 2 * j + 1, 2 * k)) < eps)
                            and (xt::abs(detail(fine_level, 2 * i + 1, 2 * j + 1, 2 * k)) < eps)
                            and (xt::abs(detail(fine_level, 2 * i, 2 * j, 2 * k + 1)) < eps)
                            and (xt::abs(detail(fine_level, 2 * i + 1, 2 * j, 2 * k + 1)) < eps)
                            and (xt::abs(detail(fine_level, 2 * i, 2 * j + 1, 2 * k + 1)) < eps)
                            and (xt::abs(detail(fine_level, 2 * i + 1, 2 * j + 1, 2 * k + 1)) < eps);

                    xt::masked_view(tag(fine_level, 2 * i, 2 * j, 2 * k), mask)             = static_cast<int>(CellFlag::coarsen);
                    xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j, 2 * k), mask)         = static_cast<int>(CellFlag::coarsen);
                    xt::masked_view(tag(fine_level, 2 * i, 2 * j + 1, 2 * k), mask)         = static_cast<int>(CellFlag::coarsen);
                    xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j + 1, 2 * k), mask)     = static_cast<int>(CellFlag::coarsen);
                    xt::masked_view(tag(fine_level, 2 * i, 2 * j, 2 * k + 1), mask)         = static_cast<int>(CellFlag::coarsen);
                    xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j, 2 * k + 1), mask)     = static_cast<int>(CellFlag::coarsen);
                    xt::masked_view(tag(fine_level, 2 * i, 2 * j + 1, 2 * k + 1), mask)     = static_cast<int>(CellFlag::coarsen);
                    xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j + 1, 2 * k + 1), mask) = static_cast<int>(CellFlag::coarsen);
                }
                else
                {
                    // auto mask = xt::sum((xt::abs(detail(level, 2*i  ,
                    // 2*j))/maxd < eps) and
                    //                     (xt::abs(detail(level, 2*i+1,
                    //                     2*j))/maxd < eps) and
                    //                     (xt::abs(detail(level, 2*i  ,
                    //                     2*j+1))/maxd < eps) and
                    //                     (xt::abs(detail(level, 2*i+1,
                    //                     2*j+1))/maxd < eps), {1}) > (size-1);
                    auto mask = xt::sum((xt::abs(detail(fine_level, 2 * i, 2 * j, 2 * k)) < eps)
                                            and (xt::abs(detail(fine_level, 2 * i + 1, 2 * j, 2 * k)) < eps)
                                            and (xt::abs(detail(fine_level, 2 * i, 2 * j + 1, 2 * k)) < eps)
                                            and (xt::abs(detail(fine_level, 2 * i + 1, 2 * j + 1, 2 * k)) < eps)
                                            and (xt::abs(detail(fine_level, 2 * i, 2 * j, 2 * k + 1)) < eps)
                                            and (xt::abs(detail(fine_level, 2 * i + 1, 2 * j, 2 * k + 1)) < eps)
                                            and (xt::abs(detail(fine_level, 2 * i, 2 * j + 1, 2 * k + 1)) < eps)
                                            and (xt::abs(detail(fine_level, 2 * i + 1, 2 * j + 1, 2 * k + 1)) < eps),
                                        {1})
                              > (size - 1);

                    xt::masked_view(tag(fine_level, 2 * i, 2 * j, 2 * k), mask)             = static_cast<int>(CellFlag::coarsen);
                    xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j, 2 * k), mask)         = static_cast<int>(CellFlag::coarsen);
                    xt::masked_view(tag(fine_level, 2 * i, 2 * j + 1, 2 * k), mask)         = static_cast<int>(CellFlag::coarsen);
                    xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j + 1, 2 * k), mask)     = static_cast<int>(CellFlag::coarsen);
                    xt::masked_view(tag(fine_level, 2 * i, 2 * j, 2 * k + 1), mask)         = static_cast<int>(CellFlag::coarsen);
                    xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j, 2 * k + 1), mask)     = static_cast<int>(CellFlag::coarsen);
                    xt::masked_view(tag(fine_level, 2 * i, 2 * j + 1, 2 * k + 1), mask)     = static_cast<int>(CellFlag::coarsen);
                    xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j + 1, 2 * k + 1), mask) = static_cast<int>(CellFlag::coarsen);
                }
            }
        }
    };

    template <class... CT>
    inline auto to_coarsen_mr(CT&&... e)
    {
        return make_field_operator_function<to_coarsen_mr_op>(std::forward<CT>(e)...);
    }

    // Uses the details in the way suggested in
    // the paper by Bihari && Harten [1997]
    template <class TInterval>
    class to_coarsen_mr_BH_op : public field_operator_base<TInterval>
    {
      public:

        INIT_OPERATOR(to_coarsen_mr_BH_op)

        template <class T1, class T2, class T3>
        inline void operator()(Dim<2>, const T1& detail, const T3& max_detail, T2& tag, double eps, std::size_t min_lev) const
        {
            constexpr auto size    = T1::size;
            std::size_t fine_level = level + 1;

            // CAVEAT : THIS CONTROL IS NECESSARY...MANY PROBLEMS WITHOUT
            if (fine_level > min_lev)
            {
                const double C_fourth_term = 2.0;
                auto maxd                  = xt::view(max_detail, fine_level);

                if constexpr (size == 1)
                {
                    auto mask = (xt::abs(.25
                                         * (detail(fine_level, 2 * i, 2 * j) - detail(fine_level, 2 * i + 1, 2 * j)
                                            - detail(fine_level, 2 * i, 2 * j + 1) + detail(fine_level, 2 * i + 1, 2 * j + 1))
                                         / maxd * C_fourth_term)
                                 < eps)
                             && (xt::abs(.25
                                         * (-detail(fine_level, 2 * i, 2 * j) + detail(fine_level, 2 * i + 1, 2 * j)
                                            - detail(fine_level, 2 * i, 2 * j + 1) + detail(fine_level, 2 * i + 1, 2 * j + 1))
                                         / maxd))
                                    < eps
                             && (xt::abs(.25
                                         * (-detail(fine_level, 2 * i, 2 * j) - detail(fine_level, 2 * i + 1, 2 * j)
                                            + detail(fine_level, 2 * i, 2 * j + 1) + detail(fine_level, 2 * i + 1, 2 * j + 1))
                                         / maxd))
                                    < eps;

                    xt::masked_view(tag(fine_level, 2 * i, 2 * j), mask)         = static_cast<int>(CellFlag::coarsen);
                    xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j), mask)     = static_cast<int>(CellFlag::coarsen);
                    xt::masked_view(tag(fine_level, 2 * i, 2 * j + 1), mask)     = static_cast<int>(CellFlag::coarsen);
                    xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j + 1), mask) = static_cast<int>(CellFlag::coarsen);
                }
                else
                {
                    auto mask = xt::sum((xt::abs(.25
                                                 * (detail(fine_level, 2 * i, 2 * j) - detail(fine_level, 2 * i + 1, 2 * j)
                                                    - detail(fine_level, 2 * i, 2 * j + 1) + detail(fine_level, 2 * i + 1, 2 * j + 1))
                                                 / maxd * C_fourth_term)
                                         < eps)
                                            && (xt::abs(.25
                                                        * (-detail(fine_level, 2 * i, 2 * j) + detail(fine_level, 2 * i + 1, 2 * j)
                                                           - detail(fine_level, 2 * i, 2 * j + 1) + detail(fine_level, 2 * i + 1, 2 * j + 1))
                                                        / maxd))
                                                   < eps
                                            && (xt::abs(.25
                                                        * (-detail(fine_level, 2 * i, 2 * j) - detail(fine_level, 2 * i + 1, 2 * j)
                                                           + detail(fine_level, 2 * i, 2 * j + 1) + detail(fine_level, 2 * i + 1, 2 * j + 1))
                                                        / maxd))
                                                   < eps,
                                        {detail.is_soa ? 0 : 1})
                              > (size - 1);

                    xt::masked_view(tag(fine_level, 2 * i, 2 * j), mask)         = static_cast<int>(CellFlag::coarsen);
                    xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j), mask)     = static_cast<int>(CellFlag::coarsen);
                    xt::masked_view(tag(fine_level, 2 * i, 2 * j + 1), mask)     = static_cast<int>(CellFlag::coarsen);
                    xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j + 1), mask) = static_cast<int>(CellFlag::coarsen);
                }
            }
        }
    };

    template <class... CT>
    inline auto to_coarsen_mr_BH(CT&&... e)
    {
        return make_field_operator_function<to_coarsen_mr_BH_op>(std::forward<CT>(e)...);
    }

    template <class TInterval>
    class to_refine_mr_op : public field_operator_base<TInterval>
    {
      public:

        INIT_OPERATOR(to_refine_mr_op)

        template <class T1, class T2>
        inline void operator()(Dim<1>, const T1& detail, T2& tag, double eps, std::size_t max_level) const
        {
            constexpr auto size    = T1::size;
            std::size_t fine_level = level + 1;

            if (fine_level < max_level)
            {
                // auto maxd = xt::view(max_detail, fine_level);
                if constexpr (size == 1)
                {
                    // auto mask = ((xt::abs(detail(fine_level, 2*i))/maxd) >
                    // eps) ||
                    //             ((xt::abs(detail(fine_level, 2*i+1))/maxd) >
                    //             eps);

                    auto mask = ((xt::abs(detail(fine_level, 2 * i))) > eps)
                             || ((xt::abs(detail(fine_level, 2 * i + 1))) > eps); // No normalization

                    xt::masked_view(tag(fine_level, 2 * i), mask)     = static_cast<int>(CellFlag::refine);
                    xt::masked_view(tag(fine_level, 2 * i + 1), mask) = static_cast<int>(CellFlag::refine);
                }
                else
                {
                    // auto mask = xt::sum(((xt::abs(detail(fine_level, 2*i
                    // ))/maxd) > eps) ||
                    //                     ((xt::abs(detail(fine_level,
                    //                     2*i+1))/maxd) > eps), {1}) > 0;

                    auto mask = xt::sum(((xt::abs(detail(fine_level, 2 * i))) > eps) || ((xt::abs(detail(fine_level, 2 * i + 1))) > eps),
                                        {detail.is_soa ? 0 : 1})
                              > 0; // No normalization

                    xt::masked_view(tag(fine_level, 2 * i), mask)     = static_cast<int>(CellFlag::refine);
                    xt::masked_view(tag(fine_level, 2 * i + 1), mask) = static_cast<int>(CellFlag::refine);
                }
            }
        }

        template <class T1, class T2>
        inline void operator()(Dim<2>, const T1& detail, T2& tag, double eps, std::size_t max_level) const
        {
            constexpr auto size    = T1::size;
            std::size_t fine_level = level + 1;

            if (fine_level < max_level)
            {
                // auto maxd = xt::view(max_detail, fine_level);
                if constexpr (size == 1)
                {
                    // auto mask = ((xt::abs(detail(fine_level, 2*i  , 2*j
                    // ))/maxd) > eps) ||
                    //             ((xt::abs(detail(fine_level, 2*i+1, 2*j
                    //             ))/maxd) > eps) ||
                    //             ((xt::abs(detail(fine_level, 2*i  ,
                    //             2*j+1))/maxd) > eps) ||
                    //             ((xt::abs(detail(fine_level, 2*i+1,
                    //             2*j+1))/maxd) > eps);
                    auto mask = (xt::abs(detail(fine_level, 2 * i, 2 * j)) > eps) || (xt::abs(detail(fine_level, 2 * i + 1, 2 * j)) > eps)
                             || (xt::abs(detail(fine_level, 2 * i, 2 * j + 1)) > eps)
                             || (xt::abs(detail(fine_level, 2 * i + 1, 2 * j + 1)) > eps);

                    xt::masked_view(tag(fine_level, 2 * i, 2 * j), mask)         = static_cast<int>(CellFlag::refine);
                    xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j), mask)     = static_cast<int>(CellFlag::refine);
                    xt::masked_view(tag(fine_level, 2 * i, 2 * j + 1), mask)     = static_cast<int>(CellFlag::refine);
                    xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j + 1), mask) = static_cast<int>(CellFlag::refine);
                }
                else
                {
                    // auto mask = xt::sum(((xt::abs(detail(fine_level, 2*i  ,
                    // 2*j  ))/maxd) > eps) ||
                    //                     ((xt::abs(detail(fine_level, 2*i+1,
                    //                     2*j  ))/maxd) > eps) ||
                    //                     ((xt::abs(detail(fine_level, 2*i  ,
                    //                     2*j+1))/maxd) > eps) ||
                    //                     ((xt::abs(detail(fine_level, 2*i+1,
                    //                     2*j+1))/maxd) > eps), {1}) > 0;
                    auto mask = xt::sum((xt::abs(detail(fine_level, 2 * i, 2 * j)) > eps)
                                            || (xt::abs(detail(fine_level, 2 * i + 1, 2 * j)) > eps)
                                            || (xt::abs(detail(fine_level, 2 * i, 2 * j + 1)) > eps)
                                            || (xt::abs(detail(fine_level, 2 * i + 1, 2 * j + 1)) > eps),
                                        {detail.is_soa ? 0 : 1})
                              > 0;

                    xt::masked_view(tag(fine_level, 2 * i, 2 * j), mask)         = static_cast<int>(CellFlag::refine);
                    xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j), mask)     = static_cast<int>(CellFlag::refine);
                    xt::masked_view(tag(fine_level, 2 * i, 2 * j + 1), mask)     = static_cast<int>(CellFlag::refine);
                    xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j + 1), mask) = static_cast<int>(CellFlag::refine);
                }
            }
        }

        template <class T1, class T2>
        inline void operator()(Dim<3>, const T1& detail, T2& tag, double eps, std::size_t max_level) const
        {
            constexpr auto size    = T1::size;
            std::size_t fine_level = level + 1;

            if (fine_level < max_level)
            {
                // auto maxd = xt::view(max_detail, fine_level);
                if constexpr (size == 1)
                {
                    // auto mask = ((xt::abs(detail(fine_level, 2*i  , 2*j
                    // ))/maxd) > eps) or
                    //             ((xt::abs(detail(fine_level, 2*i+1, 2*j
                    //             ))/maxd) > eps) or
                    //             ((xt::abs(detail(fine_level, 2*i  ,
                    //             2*j+1))/maxd) > eps) or
                    //             ((xt::abs(detail(fine_level, 2*i+1,
                    //             2*j+1))/maxd) > eps);
                    auto mask = (xt::abs(detail(fine_level, 2 * i, 2 * j, 2 * k)) > eps)
                             or (xt::abs(detail(fine_level, 2 * i + 1, 2 * j, 2 * k)) > eps)
                             or (xt::abs(detail(fine_level, 2 * i, 2 * j + 1, 2 * k)) > eps)
                             or (xt::abs(detail(fine_level, 2 * i + 1, 2 * j + 1, 2 * k)) > eps)
                             or (xt::abs(detail(fine_level, 2 * i, 2 * j, 2 * k + 1)) > eps)
                             or (xt::abs(detail(fine_level, 2 * i + 1, 2 * j, 2 * k + 1)) > eps)
                             or (xt::abs(detail(fine_level, 2 * i, 2 * j + 1, 2 * k + 1)) > eps)
                             or (xt::abs(detail(fine_level, 2 * i + 1, 2 * j + 1, 2 * k + 1)) > eps);

                    xt::masked_view(tag(fine_level, 2 * i, 2 * j, 2 * k), mask)             = static_cast<int>(CellFlag::refine);
                    xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j, 2 * k), mask)         = static_cast<int>(CellFlag::refine);
                    xt::masked_view(tag(fine_level, 2 * i, 2 * j + 1, 2 * k), mask)         = static_cast<int>(CellFlag::refine);
                    xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j + 1, 2 * k), mask)     = static_cast<int>(CellFlag::refine);
                    xt::masked_view(tag(fine_level, 2 * i, 2 * j, 2 * k + 1), mask)         = static_cast<int>(CellFlag::refine);
                    xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j, 2 * k + 1), mask)     = static_cast<int>(CellFlag::refine);
                    xt::masked_view(tag(fine_level, 2 * i, 2 * j + 1, 2 * k + 1), mask)     = static_cast<int>(CellFlag::refine);
                    xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j + 1, 2 * k + 1), mask) = static_cast<int>(CellFlag::refine);
                }
                else
                {
                    // auto mask = xt::sum(((xt::abs(detail(fine_level, 2*i  ,
                    // 2*j  ))/maxd) > eps) or
                    //                     ((xt::abs(detail(fine_level, 2*i+1,
                    //                     2*j  ))/maxd) > eps) or
                    //                     ((xt::abs(detail(fine_level, 2*i  ,
                    //                     2*j+1))/maxd) > eps) or
                    //                     ((xt::abs(detail(fine_level, 2*i+1,
                    //                     2*j+1))/maxd) > eps), {1}) > 0;
                    auto mask = xt::sum((xt::abs(detail(fine_level, 2 * i, 2 * j, 2 * k)) > eps)
                                            or (xt::abs(detail(fine_level, 2 * i + 1, 2 * j, 2 * k)) > eps)
                                            or (xt::abs(detail(fine_level, 2 * i, 2 * j + 1, 2 * k)) > eps)
                                            or (xt::abs(detail(fine_level, 2 * i + 1, 2 * j + 1, 2 * k)) > eps)
                                            or (xt::abs(detail(fine_level, 2 * i, 2 * j, 2 * k + 1)) > eps)
                                            or (xt::abs(detail(fine_level, 2 * i + 1, 2 * j, 2 * k + 1)) > eps)
                                            or (xt::abs(detail(fine_level, 2 * i, 2 * j + 1, 2 * k + 1)) > eps)
                                            or (xt::abs(detail(fine_level, 2 * i + 1, 2 * j + 1, 2 * k + 1)) > eps),
                                        {1})
                              > 0;

                    xt::masked_view(tag(fine_level, 2 * i, 2 * j, 2 * k), mask)             = static_cast<int>(CellFlag::refine);
                    xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j, 2 * k), mask)         = static_cast<int>(CellFlag::refine);
                    xt::masked_view(tag(fine_level, 2 * i, 2 * j + 1, 2 * k), mask)         = static_cast<int>(CellFlag::refine);
                    xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j + 1, 2 * k), mask)     = static_cast<int>(CellFlag::refine);
                    xt::masked_view(tag(fine_level, 2 * i, 2 * j, 2 * k + 1), mask)         = static_cast<int>(CellFlag::refine);
                    xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j, 2 * k + 1), mask)     = static_cast<int>(CellFlag::refine);
                    xt::masked_view(tag(fine_level, 2 * i, 2 * j + 1, 2 * k + 1), mask)     = static_cast<int>(CellFlag::refine);
                    xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j + 1, 2 * k + 1), mask) = static_cast<int>(CellFlag::refine);
                }
            }
        }
    };

    template <class... CT>
    inline auto to_refine_mr(CT&&... e)
    {
        return make_field_operator_function<to_refine_mr_op>(std::forward<CT>(e)...);
    }

    template <class TInterval>
    class to_refine_mr_BH_op : public field_operator_base<TInterval>
    {
      public:

        INIT_OPERATOR(to_refine_mr_BH_op)

        template <class T1, class T2, class T3>
        inline void operator()(Dim<2>, const T1& detail, const T3& max_detail, T2& tag, double eps, std::size_t max_level) const
        {
            constexpr auto size    = T1::size;
            std::size_t fine_level = level + 1;

            if (fine_level < max_level)
            {
                const double C_fourth_term = 2.0;
                auto maxd                  = xt::view(max_detail, fine_level);

                if constexpr (size == 1)
                {
                    auto mask = (xt::abs(.25
                                         * (detail(fine_level, 2 * i, 2 * j) - detail(fine_level, 2 * i + 1, 2 * j)
                                            - detail(fine_level, 2 * i, 2 * j + 1) + detail(fine_level, 2 * i + 1, 2 * j + 1))
                                         / maxd * C_fourth_term)
                                 > eps)
                             || (xt::abs(.25
                                         * (-detail(fine_level, 2 * i, 2 * j) + detail(fine_level, 2 * i + 1, 2 * j)
                                            - detail(fine_level, 2 * i, 2 * j + 1) + detail(fine_level, 2 * i + 1, 2 * j + 1))
                                         / maxd))
                                    > eps
                             || (xt::abs(.25
                                         * (-detail(fine_level, 2 * i, 2 * j) - detail(fine_level, 2 * i + 1, 2 * j)
                                            + detail(fine_level, 2 * i, 2 * j + 1) + detail(fine_level, 2 * i + 1, 2 * j + 1))
                                         / maxd))
                                    > eps;

                    xt::masked_view(tag(fine_level, 2 * i, 2 * j), mask)         = static_cast<int>(CellFlag::refine);
                    xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j), mask)     = static_cast<int>(CellFlag::refine);
                    xt::masked_view(tag(fine_level, 2 * i, 2 * j + 1), mask)     = static_cast<int>(CellFlag::refine);
                    xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j + 1), mask) = static_cast<int>(CellFlag::refine);
                }
                else
                {
                    auto mask = xt::sum((xt::abs(.25
                                                 * (detail(fine_level, 2 * i, 2 * j) - detail(fine_level, 2 * i + 1, 2 * j)
                                                    - detail(fine_level, 2 * i, 2 * j + 1) + detail(fine_level, 2 * i + 1, 2 * j + 1))
                                                 / maxd * C_fourth_term)
                                         > eps)
                                            || (xt::abs(.25
                                                        * (-detail(fine_level, 2 * i, 2 * j) + detail(fine_level, 2 * i + 1, 2 * j)
                                                           - detail(fine_level, 2 * i, 2 * j + 1) + detail(fine_level, 2 * i + 1, 2 * j + 1))
                                                        / maxd))
                                                   > eps
                                            || (xt::abs(.25
                                                        * (-detail(fine_level, 2 * i, 2 * j) - detail(fine_level, 2 * i + 1, 2 * j)
                                                           + detail(fine_level, 2 * i, 2 * j + 1) + detail(fine_level, 2 * i + 1, 2 * j + 1))
                                                        / maxd))
                                                   > eps,
                                        {detail.is_soa ? 0 : 1})
                              > 0;

                    xt::masked_view(tag(fine_level, 2 * i, 2 * j), mask)         = static_cast<int>(CellFlag::refine);
                    xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j), mask)     = static_cast<int>(CellFlag::refine);
                    xt::masked_view(tag(fine_level, 2 * i, 2 * j + 1), mask)     = static_cast<int>(CellFlag::refine);
                    xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j + 1), mask) = static_cast<int>(CellFlag::refine);
                }
            }
        }
    };

    template <class... CT>
    inline auto to_refine_mr_BH(CT&&... e)
    {
        return make_field_operator_function<to_refine_mr_BH_op>(std::forward<CT>(e)...);
    }

    template <class TInterval>
    class max_detail_mr_op : public field_operator_base<TInterval>
    {
      public:

        INIT_OPERATOR(max_detail_mr_op)

        template <class T1>
        inline void operator()(Dim<2>, const T1& detail, double& max_detail) const
        {
            auto ii = 2 * i;
            ii.step = 1;

            max_detail = std::max(max_detail,
                                  xt::amax(xt::maximum(xt::abs(detail(level + 1, ii, 2 * j)), xt::abs(detail(level + 1, ii, 2 * j + 1))))[0]);
        }
    };

    template <class... CT>
    inline auto max_detail_mr(CT&&... e)
    {
        return make_field_operator_function<max_detail_mr_op>(std::forward<CT>(e)...);
    }
} // namespace samurai
