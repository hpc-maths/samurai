// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../cell_flag.hpp"
#include "../operators_base.hpp"
#include "../utils.hpp"

namespace samurai
{

    template <std::size_t dim, class TInterval>
    class to_coarsen_mr_op : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(to_coarsen_mr_op)

        template <class T1, class T2>
        // SAMURAI_INLINE void operator()(Dim<1>, const T1& detail, const T3&
        // max_detail, T2 &tag, double eps, std::size_t min_lev) const
        SAMURAI_INLINE void operator()(Dim<1>, const T1& detail, T2& tag, double eps, std::size_t min_lev) const
        {
            using namespace math;
            std::size_t fine_level = level + 1;

            if (fine_level > min_lev)
            {
                // auto maxd = xt::view(max_detail, level);

                if constexpr (T1::is_scalar)
                {
                    // auto mask = abs(detail(level, 2*i))/maxd < eps;
                    auto mask = abs(detail(fine_level, 2 * i)) < eps; // NO normalization

                    apply_on_masked(mask,
                                    [&](auto imask)
                                    {
                                        tag(fine_level, 2 * i)(imask)     = static_cast<int>(CellFlag::coarsen);
                                        tag(fine_level, 2 * i + 1)(imask) = static_cast<int>(CellFlag::coarsen);
                                    });
                }
                else
                {
                    constexpr auto n_comp = T1::n_comp;

                    // auto mask = xt::sum((abs(detail(level, 2*i))/maxd <
                    // eps), {1}) > (n_comp-1);
                    constexpr std::size_t axis = detail::static_size_first_v<n_comp, detail::is_soa_v<T1>, T1::is_scalar, T1::static_layout>
                                                   ? 0
                                                   : 1;

                    auto mask = sum<axis>((abs(detail(fine_level, 2 * i)) < eps)) > (n_comp - 1); // No normalization

                    apply_on_masked(mask,
                                    [&](auto imask)
                                    {
                                        tag(fine_level, 2 * i)(imask)     = static_cast<int>(CellFlag::coarsen);
                                        tag(fine_level, 2 * i + 1)(imask) = static_cast<int>(CellFlag::coarsen);
                                    });
                }
            }
        }

        template <class T1, class T2>
        SAMURAI_INLINE void operator()(Dim<2>, const T1& detail, T2& tag, double eps, std::size_t min_lev) const
        {
            using namespace math;

            std::size_t fine_level = level + 1;

            if (fine_level > min_lev)
            {
                if constexpr (T1::is_scalar)
                {
                    auto mask = (abs(detail(fine_level, 2 * i, 2 * j)) < eps) && (abs(detail(fine_level, 2 * i + 1, 2 * j)) < eps)
                             && (abs(detail(fine_level, 2 * i, 2 * j + 1)) < eps) && (abs(detail(fine_level, 2 * i + 1, 2 * j + 1)) < eps);

                    apply_on_masked(
                        mask,
                        [&](auto imask)
                        {
                            static_nested_loop<dim - 1, 0, 2>(
                                [&](auto stencil)
                                {
                                    for (int ii = 0; ii < 2; ++ii)
                                    {
                                        tag(fine_level, 2 * i + ii, 2 * index + stencil)(imask) = static_cast<int>(CellFlag::coarsen);
                                    }
                                });
                        });
                    // xt::masked_view(tag(fine_level, 2 * i, 2 * j), mask)         = static_cast<int>(CellFlag::coarsen);
                    // xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j), mask)     = static_cast<int>(CellFlag::coarsen);
                    // xt::masked_view(tag(fine_level, 2 * i, 2 * j + 1), mask)     = static_cast<int>(CellFlag::coarsen);
                    // xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j + 1), mask) = static_cast<int>(CellFlag::coarsen);
                }
                else
                {
                    constexpr auto n_comp = T1::n_comp;

                    // auto mask = xt::sum((abs(detail(level, 2*i  ,
                    // 2*j))/maxd < eps) &&
                    //                     (abs(detail(level, 2*i+1,
                    //                     2*j))/maxd < eps) &&
                    //                     (abs(detail(level, 2*i  ,
                    //                     2*j+1))/maxd < eps) &&
                    //                     (abs(detail(level, 2*i+1,
                    //                     2*j+1))/maxd < eps), {1}) > (n_comp-1);
                    constexpr std::size_t axis = detail::static_size_first_v<n_comp, detail::is_soa_v<T1>, T1::is_scalar, T1::static_layout>
                                                   ? 0
                                                   : 1;

                    auto mask = all_true<axis, n_comp>(
                        (abs(detail(fine_level, 2 * i, 2 * j)) < eps) && (abs(detail(fine_level, 2 * i + 1, 2 * j)) < eps)
                        && (abs(detail(fine_level, 2 * i, 2 * j + 1)) < eps) && (abs(detail(fine_level, 2 * i + 1, 2 * j + 1)) < eps));

                    apply_on_masked(
                        mask,
                        [&](auto imask)
                        {
                            static_nested_loop<dim - 1, 0, 2>(
                                [&](auto stencil)
                                {
                                    for (int ii = 0; ii < 2; ++ii)
                                    {
                                        tag(fine_level, 2 * i + ii, 2 * index + stencil)(imask) = static_cast<int>(CellFlag::coarsen);
                                    }
                                });
                        });
                    // xt::masked_view(tag(fine_level, 2 * i, 2 * j), mask)         = static_cast<int>(CellFlag::coarsen);
                    // xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j), mask)     = static_cast<int>(CellFlag::coarsen);
                    // xt::masked_view(tag(fine_level, 2 * i, 2 * j + 1), mask)     = static_cast<int>(CellFlag::coarsen);
                    // xt::masked_view(tag(fine_level, 2 * i + 1, 2 * j + 1), mask) = static_cast<int>(CellFlag::coarsen);
                }
            }
        }

        template <class T1, class T2>
        SAMURAI_INLINE void operator()(Dim<3>, const T1& detail, T2& tag, double eps, std::size_t min_lev) const
        {
            using namespace math;

            std::size_t fine_level = level + 1;

            if (fine_level > min_lev)
            {
                // auto maxd = xt::view(max_detail, level);

                if constexpr (T1::is_scalar)
                {
                    // auto mask = (abs(detail(level, 2*i  ,   2*j))/maxd <
                    // eps) and
                    //             (abs(detail(level, 2*i+1,   2*j))/maxd <
                    //             eps) and (abs(detail(level, 2*i  ,
                    //             2*j+1))/maxd < eps) and
                    //             (abs(detail(level, 2*i+1, 2*j+1))/maxd <
                    //             eps);
                    auto mask = eval((abs(detail(fine_level, 2 * i, 2 * j, 2 * k)) < eps)
                                     && (abs(detail(fine_level, 2 * i + 1, 2 * j, 2 * k)) < eps)
                                     && (abs(detail(fine_level, 2 * i, 2 * j + 1, 2 * k)) < eps)
                                     && (abs(detail(fine_level, 2 * i + 1, 2 * j + 1, 2 * k)) < eps)
                                     && (abs(detail(fine_level, 2 * i, 2 * j, 2 * k + 1)) < eps)
                                     && (abs(detail(fine_level, 2 * i + 1, 2 * j, 2 * k + 1)) < eps)
                                     && (abs(detail(fine_level, 2 * i, 2 * j + 1, 2 * k + 1)) < eps)
                                     && (abs(detail(fine_level, 2 * i + 1, 2 * j + 1, 2 * k + 1)) < eps));

                    apply_on_masked(mask,
                                    [&](auto imask)
                                    {
                                        tag(fine_level, 2 * i, 2 * j, 2 * k)(imask)             = static_cast<int>(CellFlag::coarsen);
                                        tag(fine_level, 2 * i + 1, 2 * j, 2 * k)(imask)         = static_cast<int>(CellFlag::coarsen);
                                        tag(fine_level, 2 * i, 2 * j + 1, 2 * k)(imask)         = static_cast<int>(CellFlag::coarsen);
                                        tag(fine_level, 2 * i + 1, 2 * j + 1, 2 * k)(imask)     = static_cast<int>(CellFlag::coarsen);
                                        tag(fine_level, 2 * i, 2 * j, 2 * k + 1)(imask)         = static_cast<int>(CellFlag::coarsen);
                                        tag(fine_level, 2 * i + 1, 2 * j, 2 * k + 1)(imask)     = static_cast<int>(CellFlag::coarsen);
                                        tag(fine_level, 2 * i, 2 * j + 1, 2 * k + 1)(imask)     = static_cast<int>(CellFlag::coarsen);
                                        tag(fine_level, 2 * i + 1, 2 * j + 1, 2 * k + 1)(imask) = static_cast<int>(CellFlag::coarsen);
                                    });
                }
                else
                {
                    constexpr auto n_comp = T1::n_comp;

                    // auto mask = xt::sum((abs(detail(level, 2*i  ,
                    // 2*j))/maxd < eps) and
                    //                     (abs(detail(level, 2*i+1,
                    //                     2*j))/maxd < eps) and
                    //                     (abs(detail(level, 2*i  ,
                    //                     2*j+1))/maxd < eps) and
                    //                     (abs(detail(level, 2*i+1,
                    //                     2*j+1))/maxd < eps), {1}) > (n_comp-1);

                    constexpr std::size_t axis = detail::static_size_first_v<n_comp, detail::is_soa_v<T1>, T1::is_scalar, T1::static_layout>
                                                   ? 0
                                                   : 1;

                    auto mask = sum<axis>((abs(detail(fine_level, 2 * i, 2 * j, 2 * k)) < eps)
                                          && (abs(detail(fine_level, 2 * i + 1, 2 * j, 2 * k)) < eps)
                                          && (abs(detail(fine_level, 2 * i, 2 * j + 1, 2 * k)) < eps)
                                          && (abs(detail(fine_level, 2 * i + 1, 2 * j + 1, 2 * k)) < eps)
                                          && (abs(detail(fine_level, 2 * i, 2 * j, 2 * k + 1)) < eps)
                                          && (abs(detail(fine_level, 2 * i + 1, 2 * j, 2 * k + 1)) < eps)
                                          && (abs(detail(fine_level, 2 * i, 2 * j + 1, 2 * k + 1)) < eps)
                                          && (abs(detail(fine_level, 2 * i + 1, 2 * j + 1, 2 * k + 1)) < eps))
                              > (n_comp - 1);

                    apply_on_masked(mask,
                                    [&](auto imask)
                                    {
                                        tag(fine_level, 2 * i, 2 * j, 2 * k)(imask)             = static_cast<int>(CellFlag::coarsen);
                                        tag(fine_level, 2 * i + 1, 2 * j, 2 * k)(imask)         = static_cast<int>(CellFlag::coarsen);
                                        tag(fine_level, 2 * i, 2 * j + 1, 2 * k)(imask)         = static_cast<int>(CellFlag::coarsen);
                                        tag(fine_level, 2 * i + 1, 2 * j + 1, 2 * k)(imask)     = static_cast<int>(CellFlag::coarsen);
                                        tag(fine_level, 2 * i, 2 * j, 2 * k + 1)(imask)         = static_cast<int>(CellFlag::coarsen);
                                        tag(fine_level, 2 * i + 1, 2 * j, 2 * k + 1)(imask)     = static_cast<int>(CellFlag::coarsen);
                                        tag(fine_level, 2 * i, 2 * j + 1, 2 * k + 1)(imask)     = static_cast<int>(CellFlag::coarsen);
                                        tag(fine_level, 2 * i + 1, 2 * j + 1, 2 * k + 1)(imask) = static_cast<int>(CellFlag::coarsen);
                                    });
                }
            }
        }
    };

    template <class... CT>
    SAMURAI_INLINE auto to_coarsen_mr(CT&&... e)
    {
        return make_field_operator_function<to_coarsen_mr_op>(std::forward<CT>(e)...);
    }

    template <std::size_t dim, class TInterval>
    class to_refine_mr_op : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(to_refine_mr_op)

        template <std::size_t n_comp, bool is_soa, class T1>
        SAMURAI_INLINE auto get_mask(const T1& detail_view, double eps) const
        {
            using namespace math;

            if constexpr (n_comp == 1)
            {
                return eval(abs(detail_view) > eps); // No normalization
            }
            else
            {
                constexpr std::size_t axis = detail::static_size_first_v<n_comp, is_soa, false, SAMURAI_DEFAULT_LAYOUT> ? 0 : 1;
                return eval(sum<axis>(abs(detail_view) > eps) > 0);
            }
        }

        template <class T1, class T2>
        SAMURAI_INLINE void operator()(Dim<dim>, const T1& detail, T2& tag, double eps, std::size_t max_level) const
        {
            using namespace math;
            constexpr auto n_comp  = T1::n_comp;
            std::size_t fine_level = level + 1;

            auto mask_ghost = get_mask<n_comp, detail::is_soa_v<T1>>(detail(fine_level - 1, i, index), eps / (1 << dim));

            apply_on_masked(mask_ghost,
                            [&](auto imask)
                            {
                                static_nested_loop<dim - 1, 0, 2>(
                                    [&](auto stencil)
                                    {
                                        tag(fine_level, 2 * i, 2 * index + stencil)(imask) |= static_cast<int>(CellFlag::keep);
                                        tag(fine_level, 2 * i + 1, 2 * index + stencil)(imask) |= static_cast<int>(CellFlag::keep);
                                    });
                            });

            if (fine_level < max_level)
            {
                static_nested_loop<dim - 1, 0, 2>(
                    [&](auto stencil)
                    {
                        for (int ii = 0; ii < 2; ++ii)
                        {
                            auto mask = get_mask<n_comp, detail::is_soa_v<T1>>(detail(fine_level, 2 * i + ii, 2 * index + stencil), eps);

                            apply_on_masked(tag(fine_level, 2 * i + ii, 2 * index + stencil),
                                            mask,
                                            [](auto& e)
                                            {
                                                e |= static_cast<int>(CellFlag::refine);
                                            });
                        }
                    });
            }
        }
    };

    template <class... CT>
    SAMURAI_INLINE auto to_refine_mr(CT&&... e)
    {
        return make_field_operator_function<to_refine_mr_op>(std::forward<CT>(e)...);
    }

    template <std::size_t dim, class TInterval>
    class max_detail_mr_op : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(max_detail_mr_op)

        template <class T1>
        SAMURAI_INLINE void operator()(Dim<2>, const T1& detail, double& max_detail) const
        {
            auto ii = 2 * i;
            ii.step = 1;

            max_detail = std::max(max_detail,
                                  xt::amax(xt::maximum(abs(detail(level + 1, ii, 2 * j)), abs(detail(level + 1, ii, 2 * j + 1))))[0]);
        }
    };

    template <class... CT>
    SAMURAI_INLINE auto max_detail_mr(CT&&... e)
    {
        return make_field_operator_function<max_detail_mr_op>(std::forward<CT>(e)...);
    }
} // namespace samurai
