// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xmasked_view.hpp>
#include <xtensor/views/xview.hpp>

#include "../cell_flag.hpp"
#include "../field.hpp"
#include "../numeric/prediction.hpp"
#include "../operators_base.hpp"
#include "../utils.hpp"

namespace samurai
{
    /********************
     * maximum operator *
     ********************/

    template <std::size_t dim, class TInterval>
    class maximum_op : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(maximum_op)

        template <class T>
        SAMURAI_INLINE void operator()(Dim<1>, T& field) const
        {
            auto mask = (field(level + 1, 2 * i) & static_cast<int>(CellFlag::keep))
                      | (field(level + 1, 2 * i + 1) & static_cast<int>(CellFlag::keep));

            apply_on_masked(mask,
                            [&](auto imask)
                            {
                                field(level + 1, 2 * i)(imask) |= static_cast<int>(CellFlag::keep);
                                field(level + 1, 2 * i + 1)(imask) |= static_cast<int>(CellFlag::keep);
                                field(level, i)(imask) |= static_cast<int>(CellFlag::keep);
                            });

            auto coarsen_mask = ((field(level + 1, 2 * i) & static_cast<int>(CellFlag::coarsen))
                                 & (field(level + 1, 2 * i + 1) & static_cast<int>(CellFlag::coarsen)));

            apply_on_masked(field(level, i),
                            coarsen_mask,
                            [&](auto& e)
                            {
                                e |= static_cast<int>(CellFlag::keep);
                            });
        }

        template <class T>
        SAMURAI_INLINE void operator()(Dim<2>, T& field) const
        {
            auto mask_keep = eval((field(level + 1, 2 * i, 2 * j) & static_cast<int>(CellFlag::keep))
                                  | (field(level + 1, 2 * i + 1, 2 * j) & static_cast<int>(CellFlag::keep))
                                  | (field(level + 1, 2 * i, 2 * j + 1) & static_cast<int>(CellFlag::keep))
                                  | (field(level + 1, 2 * i + 1, 2 * j + 1) & static_cast<int>(CellFlag::keep)));

            // version 1
            // xt::masked_view(field(level + 1, 2 * i, 2 * j), mask_keep) |= static_cast<int>(CellFlag::keep);
            // xt::masked_view(field(level + 1, 2 * i + 1, 2 * j), mask_keep) |= static_cast<int>(CellFlag::keep);
            // xt::masked_view(field(level + 1, 2 * i, 2 * j + 1), mask_keep) |= static_cast<int>(CellFlag::keep);
            // xt::masked_view(field(level + 1, 2 * i + 1, 2 * j + 1), mask_keep) |= static_cast<int>(CellFlag::keep);

            // xt::masked_view(field(level, i, j), mask_keep) |= static_cast<int>(CellFlag::keep);

            // version 2
            // static_nested_loop<dim - 1, 0, 2>(
            //     [&](auto stencil)
            //     {
            //         for (int ii = 0; ii < 2; ++ii)
            //         {
            //             field(level + 1, 2 * i + ii, 2 * index + stencil) |= mask_keep * static_cast<int>(CellFlag::keep);
            //         }
            //     });
            // field(level, i, index) |= mask_keep * static_cast<int>(CellFlag::keep);

            // version 3
            apply_on_masked(mask_keep,
                            [&](auto imask)
                            {
                                field(level + 1, 2 * i, 2 * j)(imask) |= static_cast<int>(CellFlag::keep);
                                field(level + 1, 2 * i + 1, 2 * j)(imask) |= static_cast<int>(CellFlag::keep);
                                field(level + 1, 2 * i, 2 * j + 1)(imask) |= static_cast<int>(CellFlag::keep);
                                field(level + 1, 2 * i + 1, 2 * j + 1)(imask) |= static_cast<int>(CellFlag::keep);
                                field(level, i, j)(imask) |= static_cast<int>(CellFlag::keep);
                            });

            auto mask_coarsen = eval((field(level + 1, 2 * i, 2 * j) & static_cast<int>(CellFlag::coarsen))
                                     & (field(level + 1, 2 * i + 1, 2 * j) & static_cast<int>(CellFlag::coarsen))
                                     & (field(level + 1, 2 * i, 2 * j + 1) & static_cast<int>(CellFlag::coarsen))
                                     & (field(level + 1, 2 * i + 1, 2 * j + 1) & static_cast<int>(CellFlag::coarsen)));

            // version 1
            // xt::masked_view(field(level + 1, 2 * i, 2 * j), !mask_coarsen) &= ~static_cast<unsigned int>(CellFlag::coarsen);
            // xt::masked_view(field(level + 1, 2 * i + 1, 2 * j), !mask_coarsen) &= ~static_cast<unsigned int>(CellFlag::coarsen);
            // xt::masked_view(field(level + 1, 2 * i, 2 * j + 1), !mask_coarsen) &= ~static_cast<unsigned int>(CellFlag::coarsen);
            // xt::masked_view(field(level + 1, 2 * i + 1, 2 * j + 1), !mask_coarsen) &= ~static_cast<unsigned int>(CellFlag::coarsen);
            // xt::masked_view(field(level, i, j), mask_coarsen) |= static_cast<int>(CellFlag::keep);

            // version 2
            // static_nested_loop<dim - 1, 0, 2>(
            //     [&](auto stencil)
            //     {
            //         for (int ii = 0; ii < 2; ++ii)
            //         {
            //             noalias(field(level + 1, 2 * i + ii, 2 * index + stencil)) = (!mask_coarsen)
            //                                                                            * (field(level + 1, 2 * i + ii, 2 * index +
            //                                                                            stencil)
            //                                                                               & (~static_cast<unsigned
            //                                                                               int>(CellFlag::coarsen)))
            //                                                                        + mask_coarsen
            //                                                                              * field(level + 1, 2 * i + ii, 2 * index +
            //                                                                              stencil);
            //         }
            //     });

            // field(level, i, j) |= mask_coarsen * static_cast<int>(CellFlag::keep);

            // version 3
            apply_on_masked(!mask_coarsen,
                            [&](auto imask)
                            {
                                field(level + 1, 2 * i, 2 * j)(imask) &= ~static_cast<int>(CellFlag::coarsen);
                                field(level + 1, 2 * i + 1, 2 * j)(imask) &= ~static_cast<int>(CellFlag::coarsen);
                                field(level + 1, 2 * i, 2 * j + 1)(imask) &= ~static_cast<int>(CellFlag::coarsen);
                                field(level + 1, 2 * i + 1, 2 * j + 1)(imask) &= ~static_cast<int>(CellFlag::coarsen);
                            });
            apply_on_masked(field(level, i, j),
                            mask_coarsen,
                            [](auto& e)
                            {
                                e |= static_cast<int>(CellFlag::keep);
                            });
        }

        template <class T>
        SAMURAI_INLINE void operator()(Dim<3>, T& field) const
        {
            auto mask1 = (field(level + 1, 2 * i, 2 * j, 2 * k) & static_cast<int>(CellFlag::keep))
                       | (field(level + 1, 2 * i + 1, 2 * j, 2 * k) & static_cast<int>(CellFlag::keep))
                       | (field(level + 1, 2 * i, 2 * j + 1, 2 * k) & static_cast<int>(CellFlag::keep))
                       | (field(level + 1, 2 * i + 1, 2 * j + 1, 2 * k) & static_cast<int>(CellFlag::keep))
                       | (field(level + 1, 2 * i, 2 * j, 2 * k + 1) & static_cast<int>(CellFlag::keep))
                       | (field(level + 1, 2 * i + 1, 2 * j, 2 * k + 1) & static_cast<int>(CellFlag::keep))
                       | (field(level + 1, 2 * i, 2 * j + 1, 2 * k + 1) & static_cast<int>(CellFlag::keep))
                       | (field(level + 1, 2 * i + 1, 2 * j + 1, 2 * k + 1) & static_cast<int>(CellFlag::keep));

            apply_on_masked(mask1,
                            [&](auto imask)
                            {
                                field(level + 1, 2 * i, 2 * j, 2 * k)(imask) |= static_cast<int>(CellFlag::keep);
                                field(level + 1, 2 * i + 1, 2 * j, 2 * k)(imask) |= static_cast<int>(CellFlag::keep);
                                field(level + 1, 2 * i, 2 * j + 1, 2 * k)(imask) |= static_cast<int>(CellFlag::keep);
                                field(level + 1, 2 * i + 1, 2 * j + 1, 2 * k)(imask) |= static_cast<int>(CellFlag::keep);
                                field(level + 1, 2 * i, 2 * j, 2 * k + 1)(imask) |= static_cast<int>(CellFlag::keep);
                                field(level + 1, 2 * i + 1, 2 * j, 2 * k + 1)(imask) |= static_cast<int>(CellFlag::keep);
                                field(level + 1, 2 * i, 2 * j + 1, 2 * k + 1)(imask) |= static_cast<int>(CellFlag::keep);
                                field(level + 1, 2 * i + 1, 2 * j + 1, 2 * k + 1)(imask) |= static_cast<int>(CellFlag::keep);
                                field(level, i, j, k)(imask) |= static_cast<int>(CellFlag::keep);
                            });

            auto mask2 = (field(level + 1, 2 * i, 2 * j, 2 * k) & static_cast<int>(CellFlag::coarsen))
                       & (field(level + 1, 2 * i + 1, 2 * j, 2 * k) & static_cast<int>(CellFlag::coarsen))
                       & (field(level + 1, 2 * i, 2 * j + 1, 2 * k) & static_cast<int>(CellFlag::coarsen))
                       & (field(level + 1, 2 * i + 1, 2 * j + 1, 2 * k) & static_cast<int>(CellFlag::coarsen))
                       & (field(level + 1, 2 * i, 2 * j, 2 * k + 1) & static_cast<int>(CellFlag::coarsen))
                       & (field(level + 1, 2 * i + 1, 2 * j, 2 * k + 1) & static_cast<int>(CellFlag::coarsen))
                       & (field(level + 1, 2 * i, 2 * j + 1, 2 * k + 1) & static_cast<int>(CellFlag::coarsen))
                       & (field(level + 1, 2 * i + 1, 2 * j + 1, 2 * k + 1) & static_cast<int>(CellFlag::coarsen));

            apply_on_masked(!mask2,
                            [&](auto imask)
                            {
                                field(level + 1, 2 * i, 2 * j, 2 * k)(imask) &= ~static_cast<int>(CellFlag::coarsen);
                                field(level + 1, 2 * i + 1, 2 * j, 2 * k)(imask) &= ~static_cast<int>(CellFlag::coarsen);
                                field(level + 1, 2 * i, 2 * j + 1, 2 * k)(imask) &= ~static_cast<int>(CellFlag::coarsen);
                                field(level + 1, 2 * i + 1, 2 * j + 1, 2 * k)(imask) &= ~static_cast<int>(CellFlag::coarsen);
                                field(level + 1, 2 * i, 2 * j, 2 * k + 1)(imask) &= ~static_cast<int>(CellFlag::coarsen);
                                field(level + 1, 2 * i + 1, 2 * j, 2 * k + 1)(imask) &= ~static_cast<int>(CellFlag::coarsen);
                                field(level + 1, 2 * i, 2 * j + 1, 2 * k + 1)(imask) &= ~static_cast<int>(CellFlag::coarsen);
                                field(level + 1, 2 * i + 1, 2 * j + 1, 2 * k + 1)(imask) &= ~static_cast<int>(CellFlag::coarsen);
                            });

            apply_on_masked(field(level, i, j, k),
                            mask2,
                            [](auto& e)
                            {
                                e |= static_cast<int>(CellFlag::keep);
                            });
        }
    };

    template <class T>
    SAMURAI_INLINE auto maximum(T&& field)
    {
        return make_field_operator_function<maximum_op>(std::forward<T>(field));
    }

    /**************$$$$$$$$***
     * balance_2to1 operator *
     ****************$$$$$$$$*/

    template <std::size_t dim, class TInterval>
    class balance_2to1_op : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(balance_2to1_op)

        template <class T, class stencil_t>
        SAMURAI_INLINE void operator()(Dim<1>, T& cell_flag, const stencil_t& stencil) const
        {
            cell_flag(level, i - stencil[0]) |= (cell_flag(level, i) & static_cast<int>(samurai::CellFlag::keep));
        }

        template <class T, class stencil_t>
        SAMURAI_INLINE void operator()(Dim<2>, T& cell_flag, const stencil_t& stencil) const
        {
            cell_flag(level, i - stencil[0], j - stencil[1]) |= (cell_flag(level, i, j) & static_cast<int>(samurai::CellFlag::keep));
        }

        template <class T, class stencil_t>
        SAMURAI_INLINE void operator()(Dim<3>, T& cell_flag, const stencil_t& stencil) const
        {
            cell_flag(level, i - stencil[0], j - stencil[1], k - stencil[2]) |= (cell_flag(level, i, j, k)
                                                                                 & static_cast<int>(samurai::CellFlag::keep));
        }
    };

    template <class T, class stencil_t>
    SAMURAI_INLINE auto balance_2to1(T&& cell_flag, stencil_t&& stencil)
    {
        return make_field_operator_function<balance_2to1_op>(std::forward<T>(cell_flag), std::forward<stencil_t>(stencil));
    }

    /***************************
     * compute detail operator *
     ***************************/

    template <std::size_t dim, class TInterval>
    class compute_detail_op : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(compute_detail_op)

        template <class T1, class T2, std::size_t order = T2::mesh_t::config::prediction_stencil_radius>
        SAMURAI_INLINE void operator()(Dim<1>, T1& detail, const T2& field) const
        {
            if constexpr (order == 0)
            {
                detail(level + 1, 2 * i)     = field(level + 1, 2 * i) - field(level, i);
                detail(level + 1, 2 * i + 1) = field(level + 1, 2 * i + 1) - field(level, i);
            }
            else
            {
                using value_t    = typename TInterval::value_t;
                auto sorder      = static_cast<value_t>(order);
                auto interp_even = interp_coeffs<2 * order + 1>(1.);
                auto interp_odd  = interp_coeffs<2 * order + 1>(-1.);

                detail(level + 1, 2 * i)     = field(level + 1, 2 * i);
                detail(level + 1, 2 * i + 1) = field(level + 1, 2 * i + 1);

                auto detail_1 = detail(level + 1, 2 * i);
                auto detail_2 = detail(level + 1, 2 * i + 1);

                for (value_t ki = 0; ki < 2 * sorder + 1; ++ki)
                {
                    std::size_t uki = static_cast<std::size_t>(ki);
                    auto field_ik   = field(level, i + ki - sorder);
                    detail_1 -= interp_even[uki] * field_ik;
                    detail_2 -= interp_odd[uki] * field_ik;
                }
            }
        }

        template <std::size_t order>
        SAMURAI_INLINE auto get_indices(Dim<2>, const auto& mesh) const
        {
            using value_t                     = typename TInterval::value_t;
            constexpr std::size_t interp_size = 2 * order + 1;

            std::array<std::size_t, interp_size * interp_size> indices;

            for (std::size_t kj = 0; kj < interp_size; ++kj)
            {
                auto ind_l_i_jk = static_cast<std::size_t>(
                    mesh.get_index(level, i.start - static_cast<value_t>(order), j + static_cast<value_t>(kj) - static_cast<value_t>(order)));
                for (std::size_t ki = 0; ki < interp_size; ++ki)
                {
                    auto idx     = ki + kj * interp_size;
                    indices[idx] = ind_l_i_jk + ki;
                }
            }

            return indices;
        }

        template <std::size_t order>
        SAMURAI_INLINE auto get_indices(Dim<3>, const auto& mesh) const
        {
            using value_t                     = typename TInterval::value_t;
            constexpr std::size_t interp_size = 2 * order + 1;

            std::array<std::size_t, interp_size * interp_size * interp_size> indices;

            for (std::size_t kk = 0; kk < interp_size; ++kk)
            {
                for (std::size_t kj = 0; kj < interp_size; ++kj)
                {
                    auto ind_l_i_jk = static_cast<std::size_t>(mesh.get_index(level,
                                                                              i.start - static_cast<value_t>(order),
                                                                              j + static_cast<value_t>(kj) - static_cast<value_t>(order),
                                                                              k + static_cast<value_t>(kk) - static_cast<value_t>(order)));
                    for (std::size_t ki = 0; ki < interp_size; ++ki)
                    {
                        auto idx     = ki + kj * interp_size + kk * interp_size * interp_size;
                        indices[idx] = ind_l_i_jk + ki;
                    }
                }
            }

            return indices;
        }

        template <class T1, class T2, std::size_t order = T2::mesh_t::config::prediction_stencil_radius>
        SAMURAI_INLINE void operator()(Dim<2>, T1& detail, const T2& field) const
        {
            if constexpr (order == 0)
            {
                detail(level + 1, 2 * i, 2 * j)         = field(level + 1, 2 * i, 2 * j) - field(level, i, j);
                detail(level + 1, 2 * i + 1, 2 * j)     = field(level + 1, 2 * i + 1, 2 * j) - field(level, i, j);
                detail(level + 1, 2 * i, 2 * j + 1)     = field(level + 1, 2 * i, 2 * j + 1) - field(level, i, j);
                detail(level + 1, 2 * i + 1, 2 * j + 1) = field(level + 1, 2 * i + 1, 2 * j + 1) - field(level, i, j);
            }
            else
            {
                auto interp_even = interp_coeffs<2 * order + 1>(1.);
                auto interp_odd  = interp_coeffs<2 * order + 1>(-1.);

                auto indices = get_indices<order>(Dim<2>{}, field.mesh());

                const auto* data = field.data();

                auto* detail_data = detail.data();

                auto ind1 = static_cast<std::size_t>(field.mesh().get_index(level + 1, 2 * i.start, 2 * j));
                auto ind2 = static_cast<std::size_t>(field.mesh().get_index(level + 1, 2 * i.start, 2 * j + 1));

                constexpr std::size_t interp_size = 2 * order + 1;

                // TODO: this implementation only works for AOS layout. We need to implement the SOA version and pay attention that detail
                // is always AOS but not necessarily field.
                // TODO: see if the separable implementation is faster.
                for (std::size_t ii = 0, i_f = 0; ii < i.size(); ++ii, i_f += 2)
                {
                    if constexpr (T2::is_scalar)
                    {
                        double d1 = data[ind1 + i_f];
                        double d2 = data[ind1 + i_f + 1];
                        double d3 = data[ind2 + i_f];
                        double d4 = data[ind2 + i_f + 1];

                        for (std::size_t kj = 0; kj < interp_size; ++kj)
                        {
                            for (std::size_t ki = 0; ki < interp_size; ++ki)
                            {
                                auto idx         = ki + kj * interp_size;
                                const double src = data[indices[idx] + ii];

                                d1 -= interp_even[ki] * interp_even[kj] * src;
                                d2 -= interp_odd[ki] * interp_even[kj] * src;
                                d3 -= interp_even[ki] * interp_odd[kj] * src;
                                d4 -= interp_odd[ki] * interp_odd[kj] * src;
                            }
                        }

                        detail_data[ind1 + i_f]     = d1;
                        detail_data[ind1 + i_f + 1] = d2;
                        detail_data[ind2 + i_f]     = d3;
                        detail_data[ind2 + i_f + 1] = d4;
                    }
                    else
                    {
                        for (std::size_t nc = 0; nc < T2::n_comp; ++nc)
                        {
                            double d1 = data[(ind1 + i_f) * T2::n_comp + nc];
                            double d2 = data[(ind1 + i_f + 1) * T2::n_comp + nc];
                            double d3 = data[(ind2 + i_f) * T2::n_comp + nc];
                            double d4 = data[(ind2 + i_f + 1) * T2::n_comp + nc];

                            for (std::size_t kj = 0; kj < interp_size; ++kj)
                            {
                                for (std::size_t ki = 0; ki < interp_size; ++ki)
                                {
                                    auto idx         = (indices[ki + kj * interp_size] + ii) * T2::n_comp;
                                    const double src = data[idx + nc];

                                    d1 -= interp_even[ki] * interp_even[kj] * src;
                                    d2 -= interp_odd[ki] * interp_even[kj] * src;
                                    d3 -= interp_even[ki] * interp_odd[kj] * src;
                                    d4 -= interp_odd[ki] * interp_odd[kj] * src;
                                }
                            }

                            detail_data[(ind1 + i_f) * T2::n_comp + nc]     = d1;
                            detail_data[(ind1 + i_f + 1) * T2::n_comp + nc] = d2;
                            detail_data[(ind2 + i_f) * T2::n_comp + nc]     = d3;
                            detail_data[(ind2 + i_f + 1) * T2::n_comp + nc] = d4;
                        }
                    }
                }
            }
        }

        template <class T1, class T2, std::size_t order = T2::mesh_t::config::prediction_stencil_radius>
        SAMURAI_INLINE void operator()(Dim<3>, T1& detail, const T2& field) const
        {
            if constexpr (order == 0)
            {
                detail(level + 1, 2 * i, 2 * j, 2 * k)             = field(level + 1, 2 * i, 2 * j, 2 * k) - field(level, i, j, k);
                detail(level + 1, 2 * i + 1, 2 * j, 2 * k)         = field(level + 1, 2 * i + 1, 2 * j, 2 * k) - field(level, i, j, k);
                detail(level + 1, 2 * i, 2 * j + 1, 2 * k)         = field(level + 1, 2 * i, 2 * j + 1, 2 * k) - field(level, i, j, k);
                detail(level + 1, 2 * i + 1, 2 * j + 1, 2 * k)     = field(level + 1, 2 * i + 1, 2 * j + 1, 2 * k) - field(level, i, j, k);
                detail(level + 1, 2 * i, 2 * j, 2 * k + 1)         = field(level + 1, 2 * i, 2 * j, 2 * k + 1) - field(level, i, j, k);
                detail(level + 1, 2 * i + 1, 2 * j, 2 * k + 1)     = field(level + 1, 2 * i + 1, 2 * j, 2 * k + 1) - field(level, i, j, k);
                detail(level + 1, 2 * i, 2 * j + 1, 2 * k + 1)     = field(level + 1, 2 * i, 2 * j + 1, 2 * k + 1) - field(level, i, j, k);
                detail(level + 1, 2 * i + 1, 2 * j + 1, 2 * k + 1) = field(level + 1, 2 * i + 1, 2 * j + 1, 2 * k + 1)
                                                                   - field(level, i, j, k);
            }
            else
            {
                auto interp_even = interp_coeffs<2 * order + 1>(1.);
                auto interp_odd  = interp_coeffs<2 * order + 1>(-1.);

                auto indices     = get_indices<order>(Dim<3>{}, field.mesh());
                const auto* data = field.array().data();

                auto field_1 = field(level + 1, 2 * i, 2 * j, 2 * k);
                auto field_2 = field(level + 1, 2 * i + 1, 2 * j, 2 * k);
                auto field_3 = field(level + 1, 2 * i, 2 * j + 1, 2 * k);
                auto field_4 = field(level + 1, 2 * i + 1, 2 * j + 1, 2 * k);
                auto field_5 = field(level + 1, 2 * i, 2 * j, 2 * k + 1);
                auto field_6 = field(level + 1, 2 * i + 1, 2 * j, 2 * k + 1);
                auto field_7 = field(level + 1, 2 * i, 2 * j + 1, 2 * k + 1);
                auto field_8 = field(level + 1, 2 * i + 1, 2 * j + 1, 2 * k + 1);

                auto detail_1 = detail(level + 1, 2 * i, 2 * j, 2 * k);
                auto detail_2 = detail(level + 1, 2 * i + 1, 2 * j, 2 * k);
                auto detail_3 = detail(level + 1, 2 * i, 2 * j + 1, 2 * k);
                auto detail_4 = detail(level + 1, 2 * i + 1, 2 * j + 1, 2 * k);
                auto detail_5 = detail(level + 1, 2 * i, 2 * j, 2 * k + 1);
                auto detail_6 = detail(level + 1, 2 * i + 1, 2 * j, 2 * k + 1);
                auto detail_7 = detail(level + 1, 2 * i, 2 * j + 1, 2 * k + 1);
                auto detail_8 = detail(level + 1, 2 * i + 1, 2 * j + 1, 2 * k + 1);

                constexpr std::size_t interp_size = 2 * order + 1;

                // TODO: this implementation only works for AOS layout. We need to implement the SOA version and pay attention that detail
                // is always AOS but not necessarily field.
                // TODO: see if the separable implementation is faster.
                for (std::size_t ii = 0, i_f = 0; ii < i.size(); ++ii, i_f += 2)
                {
                    if constexpr (T2::is_scalar)
                    {
                        double d1 = field_1(ii);
                        double d2 = field_2(ii);
                        double d3 = field_3(ii);
                        double d4 = field_4(ii);
                        double d5 = field_5(ii);
                        double d6 = field_6(ii);
                        double d7 = field_7(ii);
                        double d8 = field_8(ii);

                        for (std::size_t kk = 0; kk < interp_size; ++kk)
                        {
                            for (std::size_t kj = 0; kj < interp_size; ++kj)
                            {
                                for (std::size_t ki = 0; ki < interp_size; ++ki)
                                {
                                    auto idx         = ki + kj * interp_size + kk * interp_size * interp_size;
                                    const double src = data[indices[idx] + ii];

                                    d1 -= interp_even[ki] * interp_even[kj] * interp_even[kk] * src;
                                    d2 -= interp_odd[ki] * interp_even[kj] * interp_even[kk] * src;
                                    d3 -= interp_even[ki] * interp_odd[kj] * interp_even[kk] * src;
                                    d4 -= interp_odd[ki] * interp_odd[kj] * interp_even[kk] * src;
                                    d5 -= interp_even[ki] * interp_even[kj] * interp_odd[kk] * src;
                                    d6 -= interp_odd[ki] * interp_even[kj] * interp_odd[kk] * src;
                                    d7 -= interp_even[ki] * interp_odd[kj] * interp_odd[kk] * src;
                                    d8 -= interp_odd[ki] * interp_odd[kj] * interp_odd[kk] * src;
                                }
                            }
                        }

                        detail_1(ii) = d1;
                        detail_2(ii) = d2;
                        detail_3(ii) = d3;
                        detail_4(ii) = d4;
                        detail_5(ii) = d5;
                        detail_6(ii) = d6;
                        detail_7(ii) = d7;
                        detail_8(ii) = d8;
                    }
                    else
                    {
                        for (std::size_t nc = 0; nc < T2::n_comp; ++nc)
                        {
                            double d1 = field_1(ii, nc);
                            double d2 = field_2(ii, nc);
                            double d3 = field_3(ii, nc);
                            double d4 = field_4(ii, nc);
                            double d5 = field_5(ii, nc);
                            double d6 = field_6(ii, nc);
                            double d7 = field_7(ii, nc);
                            double d8 = field_8(ii, nc);

                            for (std::size_t kk = 0; kk < interp_size; ++kk)
                            {
                                for (std::size_t kj = 0; kj < interp_size; ++kj)
                                {
                                    for (std::size_t ki = 0; ki < interp_size; ++ki)
                                    {
                                        auto idx = (indices[ki + kj * interp_size + kk * interp_size * interp_size] + ii) * T2::n_comp;
                                        const double src = data[idx + nc];

                                        d1 -= interp_even[ki] * interp_even[kj] * interp_even[kk] * src;
                                        d2 -= interp_odd[ki] * interp_even[kj] * interp_even[kk] * src;
                                        d3 -= interp_even[ki] * interp_odd[kj] * interp_even[kk] * src;
                                        d4 -= interp_odd[ki] * interp_odd[kj] * interp_even[kk] * src;
                                        d5 -= interp_even[ki] * interp_even[kj] * interp_odd[kk] * src;
                                        d6 -= interp_odd[ki] * interp_even[kj] * interp_odd[kk] * src;
                                        d7 -= interp_even[ki] * interp_odd[kj] * interp_odd[kk] * src;
                                        d8 -= interp_odd[ki] * interp_odd[kj] * interp_odd[kk] * src;
                                    }
                                }
                            }

                            detail_1(ii, nc) = d1;
                            detail_2(ii, nc) = d2;
                            detail_3(ii, nc) = d3;
                            detail_4(ii, nc) = d4;
                            detail_5(ii, nc) = d5;
                            detail_6(ii, nc) = d6;
                            detail_7(ii, nc) = d7;
                            detail_8(ii, nc) = d8;
                        }
                    }
                }
            }
        }
    };

    template <class T1, class T2, std::enable_if_t<detail::is_field_type_v<T2>, int> = 0>
    SAMURAI_INLINE auto compute_detail(T1&& detail, T2&& field)
    {
        return make_field_operator_function<compute_detail_op>(std::forward<T1>(detail), std::forward<T2>(field));
    }

    namespace detail
    {
        template <bool transpose, class Field>
        class detail_range
        {
          public:

            static constexpr std::size_t dim    = Field::dim;
            static constexpr std::size_t n_comp = Field::n_comp;
            static constexpr bool is_soa        = detail::is_soa_v<Field>;
            static constexpr bool is_scalar     = false;

            using interval_t    = typename Field::interval_t;
            using coord_index_t = typename interval_t::coord_index_t;
            using array_index_t = xt::xtensor_fixed<coord_index_t, xt::xshape<dim - 1>>;

            detail_range(Field& detail, std::size_t beg, std::size_t end)
                : m_detail(detail)
                , m_beg(beg)
                , m_end(end)
            {
            }

            template <class... T>
            auto operator()(std::size_t level, const interval_t& i, const T... index)
            {
                if constexpr (transpose)
                {
                    return math::transpose(m_detail(m_beg, m_end, level, i, index...));
                }
                else
                {
                    return m_detail(m_beg, m_end, level, i, index...);
                }
            }

            auto operator()(std::size_t level, const interval_t& i, const array_index_t& index)
            {
                if constexpr (transpose)
                {
                    return math::transpose(m_detail(m_beg, m_end, level, i, index));
                }
                else
                {
                    return m_detail(m_beg, m_end, level, i, index);
                }
            }

            auto data()
            {
                return m_detail.data() + m_beg * m_detail.n_comp * m_detail.mesh().nb_cells();
            }

            auto data() const
            {
                return m_detail.data() + m_beg * m_detail.n_comp * m_detail.mesh().nb_cells();
            }

          private:

            Field& m_detail;
            std::size_t m_beg;
            std::size_t m_end;
        };

        template <bool transpose, class Field>
        auto make_detail_range(Field& field, std::size_t beg, std::size_t end)
        {
            return detail_range<transpose, Field>(field, beg, end);
        }
    }

    template <std::size_t dim, class TInterval>
    class compute_detail_on_tuple_op : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(compute_detail_on_tuple_op)

        template <class Ranges, class T1, class T2>
        SAMURAI_INLINE void compute_detail_impl(Dim<dim> d, std::size_t i_r, const Ranges& ranges, T1& detail, const T2& field) const
        {
            auto dest_shape = shape(detail(ranges[i_r], ranges[i_r + 1], level + 1, 2 * i, 2 * index));
            auto src_shape  = shape(field(level + 1, 2 * i, 2 * index));

            if (xt::same_shape(dest_shape, src_shape))
            {
                auto detail_range = detail::make_detail_range<false>(detail, ranges[i_r], ranges[i_r + 1]);
                compute_detail_op<dim, TInterval> compute_detail_(level, i, index);
                compute_detail_(d, detail_range, field);
            }
            else
            {
                auto detail_range = detail::make_detail_range<true>(detail, ranges[i_r], ranges[i_r + 1]);
                compute_detail_op<dim, TInterval> compute_detail_(level, i, index);
                compute_detail_(d, detail_range, field);
            }
        }

        template <class T1, class T2, std::size_t... Is>
        SAMURAI_INLINE void compute_detail_impl(Dim<dim>, T1& detail, const T2& fields, std::index_sequence<Is...>) const
        {
            std::array<std::size_t, std::tuple_size_v<T2> + 1> ranges;
            ranges[0]      = 0;
            std::size_t ir = 1;
            (
                [&]()
                {
                    ranges[ir] = ranges[ir - 1] + std::get<Is>(fields).n_comp;
                    ++ir;
                }(),
                ...);
            (compute_detail_impl(Dim<dim>(), Is, ranges, detail, std::get<Is>(fields)), ...);
        }

        template <class T1, class T2>
        SAMURAI_INLINE void operator()(Dim<dim>, T1& detail, const T2& fields) const
        {
            compute_detail_impl(Dim<dim>(), detail, fields.elements(), std::make_index_sequence<std::tuple_size_v<typename T2::tuple_type>>{});
        }
    };

    template <class Field, class... T>
    SAMURAI_INLINE auto compute_detail(Field& detail, const Field_tuple<T...>& fields)
    {
        return make_field_operator_function<compute_detail_on_tuple_op>(detail, fields);
    }

    /*******************************
     * compute max detail operator *
     *******************************/

    template <std::size_t dim, class TInterval>
    class compute_max_detail_op : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(compute_max_detail_op)

        template <class T, class U>
        SAMURAI_INLINE void operator()(Dim<1>, const U& detail, T& max_detail) const
        {
            auto ii       = 2 * i;
            ii.step       = 1;
            auto max_view = xt::view(max_detail, level + 1);

            max_view = xt::maximum(max_view, xt::amax(xt::abs(detail(level + 1, ii)), {0}));
        }

        template <class T, class U>
        SAMURAI_INLINE void operator()(Dim<2>, const U& detail, T& max_detail) const
        {
            auto ii       = 2 * i;
            ii.step       = 1;
            auto max_view = xt::view(max_detail, level + 1);

            max_view = xt::maximum(
                max_view,
                xt::amax(xt::maximum(xt::abs(detail(level + 1, ii, 2 * j)), xt::abs(detail(level + 1, ii, 2 * j + 1))), {0}));
        }

        template <class T, class U>
        SAMURAI_INLINE void operator()(Dim<3>, const U& detail, T& max_detail) const
        {
            auto ii       = 2 * i;
            ii.step       = 1;
            auto max_view = xt::view(max_detail, level + 1);

            max_view = xt::maximum(max_view,
                                   xt::amax(xt::maximum(xt::maximum(xt::abs(detail(level + 1, ii, 2 * j, 2 * k)),
                                                                    xt::abs(detail(level + 1, ii, 2 * j + 1, 2 * k))),
                                                        xt::maximum(xt::abs(detail(level + 1, ii, 2 * j, 2 * k + 1)),
                                                                    xt::abs(detail(level + 1, ii, 2 * j + 1, 2 * k + 1)))),
                                            {0}));
        }
    };

    template <class T, class U>
    SAMURAI_INLINE auto compute_max_detail(U&& detail, T&& max_detail)
    {
        return make_field_operator_function<compute_max_detail_op>(std::forward<U>(detail), std::forward<T>(max_detail));
    }

    /*******************************
     * compute max detail operator *
     *******************************/

    template <std::size_t dim, class TInterval>
    class compute_max_detail_op_ : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(compute_max_detail_op_)

        template <class T, class U>
        SAMURAI_INLINE void operator()(Dim<1>, const U& detail, T& max_detail) const
        {
            max_detail[level] = std::max(max_detail[level], xt::amax(xt::abs(detail(level, i)))[0]);
        }

        template <class T, class U>
        SAMURAI_INLINE void operator()(Dim<2>, const U& detail, T& max_detail) const
        {
            max_detail[level] = std::max(max_detail[level], xt::amax(xt::abs(detail(level, i, j)))[0]);
        }

        template <class T, class U>
        SAMURAI_INLINE void operator()(Dim<3>, const U& detail, T& max_detail) const
        {
            max_detail[level] = std::max(max_detail[level], xt::amax(xt::abs(detail(level, i, j, k)))[0]);
        }
    };

    template <class T, class U>
    SAMURAI_INLINE auto compute_max_detail_(U&& detail, T&& max_detail)
    {
        return make_field_operator_function<compute_max_detail_op_>(std::forward<U>(detail), std::forward<T>(max_detail));
    }

    /***********************
     * to_coarsen operator *
     ***********************/

    template <std::size_t dim, class TInterval>
    class to_coarsen_op : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(to_coarsen_op)

        template <class T, class U, class V>
        SAMURAI_INLINE void operator()(Dim<1>, T& keep, const U& detail, double eps) const
        {
            auto mask = abs(detail(level + 1, 2 * i)) < eps;
            // auto mask = (.5 *
            //              (xt::abs(detail(level + 1, 2 * i)) +
            //               xt::abs(detail(level + 1, 2 * i + 1))) /
            //              max_detail[level + 1]) < eps;

            apply_on_masked(mask,
                            [&](auto imask)
                            {
                                keep(level + 1, 2 * i)(imask)     = static_cast<int>(CellFlag::coarsen);
                                keep(level + 1, 2 * i + 1)(imask) = static_cast<int>(CellFlag::coarsen);
                            });
        }

        template <class T, class U, class V>
        SAMURAI_INLINE void operator()(Dim<2>, T& keep, const U& detail, double eps) const
        {
            auto mask = abs(detail(level + 1, 2 * i, 2 * j)) < eps;

            // auto mask = (0.25 *
            //              (xt::abs(detail(level + 1, 2 * i, 2 * j)) +
            //               xt::abs(detail(level + 1, 2 * i + 1, 2 * j)) +
            //               xt::abs(detail(level + 1, 2 * i, 2 * j + 1)) +
            //               xt::abs(detail(level + 1, 2 * i + 1, 2 * j + 1))) /
            //              max_detail[level + 1]) < eps;

            apply_on_masked(mask,
                            [&](auto imask)
                            {
                                for (coord_index_t jj = 0; jj < 2; ++jj)
                                {
                                    for (coord_index_t ii = 0; ii < 2; ++ii)
                                    {
                                        keep(level + 1, 2 * i + ii, 2 * j + jj)(imask) = static_cast<int>(CellFlag::coarsen);
                                    }
                                }
                            });
        }

        template <class T, class U, class V>
        SAMURAI_INLINE void operator()(Dim<3>, T& keep, const U& detail, double eps) const
        {
            auto mask = abs(detail(level + 1, 2 * i, 2 * j, 2 * k)) < eps;

            apply_on_masked(mask,
                            [&](auto imask)
                            {
                                for (coord_index_t kk = 0; kk < 2; ++kk)
                                {
                                    for (coord_index_t jj = 0; jj < 2; ++jj)
                                    {
                                        for (coord_index_t ii = 0; ii < 2; ++ii)
                                        {
                                            keep(level + 1, 2 * i + ii, 2 * j + jj, 2 * k + kk)(imask) = static_cast<int>(CellFlag::coarsen);
                                        }
                                    }
                                }
                            });
        }
    };

    template <class... CT>
    SAMURAI_INLINE auto to_coarsen(CT&&... e)
    {
        return make_field_operator_function<to_coarsen_op>(std::forward<CT>(e)...);
    }

    /*************************
     * refine_ghost operator *
     *************************/

    template <std::size_t dim, class TInterval>
    class refine_ghost_op : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(refine_ghost_op)

        template <class T>
        SAMURAI_INLINE void operator()(Dim<1>, T& flag) const
        {
            auto mask = flag(level + 1, i) & static_cast<int>(CellFlag::keep);
            apply_on_masked(flag(level, i / 2),
                            mask,
                            [](auto& e)
                            {
                                e = static_cast<int>(CellFlag::refine);
                            });
        }

        template <class T>
        SAMURAI_INLINE void operator()(Dim<2>, T& flag) const
        {
            auto mask = flag(level + 1, i, j) & static_cast<int>(CellFlag::keep);
            apply_on_masked(flag(level, i / 2, j / 2),
                            mask,
                            [](auto& e)
                            {
                                e = static_cast<int>(CellFlag::refine);
                            });
        }

        template <class T>
        SAMURAI_INLINE void operator()(Dim<3>, T& flag) const
        {
            auto mask = flag(level + 1, i, j, k) & static_cast<int>(CellFlag::keep);
            apply_on_masked(flag(level, i / 2, j / 2, k / 2),
                            mask,
                            [](auto& e)
                            {
                                e = static_cast<int>(CellFlag::refine);
                            });
        }
    };

    template <class... CT>
    SAMURAI_INLINE auto refine_ghost(CT&&... e)
    {
        return make_field_operator_function<refine_ghost_op>(std::forward<CT>(e)...);
    }

    /********************
     * enlarge operator *
     ********************/

    template <std::size_t dim, class TInterval>
    class enlarge_op : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(enlarge_op)

        template <class T>
        SAMURAI_INLINE void operator()(Dim<1>, T& cell_flag) const
        {
            auto keep_mask = cell_flag(level, i) & static_cast<int>(CellFlag::keep);

            apply_on_masked(keep_mask,
                            [&](auto imask)
                            {
                                for (int ii = -1; ii < 2; ++ii)
                                {
                                    cell_flag(level, i + ii)(imask) |= static_cast<int>(CellFlag::enlarge);
                                }
                            });
        }

        template <class T>
        SAMURAI_INLINE void operator()(Dim<2>, T& cell_flag) const
        {
            auto keep_mask = cell_flag(level, i, j) & static_cast<int>(CellFlag::keep);

            apply_on_masked(keep_mask,
                            [&](auto imask)
                            {
                                for (int jj = -1; jj < 2; ++jj)
                                {
                                    for (int ii = -1; ii < 2; ++ii)
                                    {
                                        cell_flag(level, i + ii, j + jj)(imask) |= static_cast<int>(CellFlag::enlarge);
                                    }
                                }
                            });
        }

        template <class T>
        SAMURAI_INLINE void operator()(Dim<3>, T& cell_flag) const
        {
            auto keep_mask = cell_flag(level, i, j, k) & static_cast<int>(CellFlag::keep);

            apply_on_masked(keep_mask,
                            [&](auto imask)
                            {
                                for (int kk = -1; kk < 2; ++kk)
                                {
                                    for (int jj = -1; jj < 2; ++jj)
                                    {
                                        for (int ii = -1; ii < 2; ++ii)
                                        {
                                            cell_flag(level, i + ii, j + jj, k + kk)(imask) |= static_cast<int>(CellFlag::enlarge);
                                        }
                                    }
                                }
                            });
        }
    };

    template <class... CT>
    SAMURAI_INLINE auto enlarge(CT&&... e)
    {
        return make_field_operator_function<enlarge_op>(std::forward<CT>(e)...);
    }

    /*******************************
     * keep_around_refine operator *
     *******************************/

    template <std::size_t dim, class TInterval>
    class keep_around_refine_op : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(keep_around_refine_op)

        template <class T>
        SAMURAI_INLINE void operator()(Dim<1>, T& cell_flag) const
        {
            auto refine_mask = cell_flag(level, i) & static_cast<int>(CellFlag::refine);

            apply_on_masked(refine_mask,
                            [&](auto imask)
                            {
                                for (int ii = -1; ii < 2; ++ii)
                                {
                                    cell_flag(level, i + ii)(imask) |= static_cast<int>(CellFlag::keep);
                                }
                            });
        }

        template <class T>
        SAMURAI_INLINE void operator()(Dim<2>, T& cell_flag) const
        {
            auto refine_mask = cell_flag(level, i, j) & static_cast<int>(CellFlag::refine);

            apply_on_masked(refine_mask,
                            [&](auto imask)
                            {
                                static_nested_loop<dim - 1, -1, 2>(
                                    [&](auto stencil)
                                    {
                                        for (int ii = -1; ii < 2; ++ii)
                                        {
                                            cell_flag(level, i + ii, index + stencil)(imask) |= static_cast<int>(CellFlag::keep);
                                        }
                                    });
                            });
        }

        template <class T>
        SAMURAI_INLINE void operator()(Dim<3>, T& cell_flag) const
        {
            auto refine_mask = cell_flag(level, i, j, k) & static_cast<int>(CellFlag::refine);

            apply_on_masked(refine_mask,
                            [&](auto imask)
                            {
                                for (int kk = -1; kk < 2; ++kk)
                                {
                                    for (int jj = -1; jj < 2; ++jj)
                                    {
                                        for (int ii = -1; ii < 2; ++ii)
                                        {
                                            cell_flag(level, i + ii, j + jj, k + kk)(imask) |= static_cast<int>(CellFlag::keep);
                                        }
                                    }
                                }
                            });
        }
    };

    template <class... CT>
    SAMURAI_INLINE auto keep_around_refine(CT&&... e)
    {
        return make_field_operator_function<keep_around_refine_op>(std::forward<CT>(e)...);
    }

    /***********************
     * apply_expr operator *
     ***********************/

    template <std::size_t dim, class TInterval>
    class apply_expr_op : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(apply_expr_op)

        template <class T, class E>
        SAMURAI_INLINE void operator()(Dim<1>, T& field, const field_expression<E>& e) const
        {
            field(level, i) = e.derived_cast()(level, i);
        }

        template <class T, class E>
        SAMURAI_INLINE void operator()(Dim<2>, T& field, const field_expression<E>& e) const
        {
            field(level, i, j) = e.derived_cast()(level, i, j);
        }

        template <class T, class E>
        SAMURAI_INLINE void operator()(Dim<3>, T& field, const field_expression<E>& e) const
        {
            field(level, i, j, k) = e.derived_cast()(level, i, j, k);
        }
    };

    template <class... CT>
    SAMURAI_INLINE auto apply_expr(CT&&... e)
    {
        return make_field_operator_function<apply_expr_op>(std::forward<CT>(e)...);
    }

    /*******************
     * extend operator *
     *******************/
    template <std::size_t dim, class TInterval>
    class extend_op : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(extend_op)

        template <class T>
        SAMURAI_INLINE void operator()(Dim<1>, T& tag) const
        {
            auto refine_mask = tag(level, i) & static_cast<int>(samurai::CellFlag::refine);

            const int added_cells = 1; // 1 by default

            apply_on_masked(refine_mask,
                            [&](auto imask)
                            {
                                for (int ii = -added_cells; ii < added_cells + 1; ++ii)
                                {
                                    tag(level, i + ii)(imask) |= static_cast<int>(samurai::CellFlag::keep);
                                }
                            });
        }

        template <class T>
        SAMURAI_INLINE void operator()(Dim<2>, T& tag) const
        {
            auto refine_mask = tag(level, i, j) & static_cast<int>(samurai::CellFlag::refine);

            const int added_cells = 1; // 1 by default

            apply_on_masked(refine_mask,
                            [&](auto imask)
                            {
                                static_nested_loop<dim - 1, -added_cells, added_cells + 1>(
                                    [&](auto stencil)
                                    {
                                        for (int ii = -added_cells; ii < added_cells + 1; ++ii)
                                        {
                                            tag(level, i + ii, index + stencil)(imask) |= static_cast<int>(samurai::CellFlag::keep);
                                        }
                                    });
                            });
            // for (int jj = -1; jj < 2; ++jj)
            // {
            //     for (int ii = -1; ii < 2; ++ii)
            //     {
            //         xt::masked_view(tag(level, i + ii, j + jj), refine_mask) |= static_cast<int>(samurai::CellFlag::keep);
            //     }
            // }
        }

        template <class T>
        SAMURAI_INLINE void operator()(Dim<3>, T& tag) const
        {
            auto refine_mask = tag(level, i, j, k) & static_cast<int>(samurai::CellFlag::refine);

            const int added_cells = 1; // 1 by default

            apply_on_masked(refine_mask,
                            [&](auto imask)
                            {
                                for (int kk = -added_cells; kk < added_cells + 1; ++kk)
                                {
                                    for (int jj = -added_cells; jj < added_cells + 1; ++jj)
                                    {
                                        for (int ii = -added_cells; ii < added_cells + 1; ++ii)
                                        {
                                            tag(level, i + ii, j + jj, k + kk)(imask) |= static_cast<int>(samurai::CellFlag::keep);
                                        }
                                    }
                                }
                            });
        }
    };

    template <class... CT>
    SAMURAI_INLINE auto extend(CT&&... e)
    {
        return make_field_operator_function<extend_op>(std::forward<CT>(e)...);
    }

    /****************************
     * make_graduation operator *
     ****************************/

    template <std::size_t dim, class TInterval>
    class make_graduation_op : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(make_graduation_op)

        template <class T>
        SAMURAI_INLINE void operator()(Dim<1>, T& tag) const
        {
            auto i_even = i.even_elements();
            if (i_even.is_valid())
            {
                auto mask = tag(level, i_even) & static_cast<int>(CellFlag::keep);
                apply_on_masked(tag(level - 1, i_even >> 1),
                                mask,
                                [](auto& e)
                                {
                                    e |= static_cast<int>(CellFlag::refine);
                                });
            }

            auto i_odd = i.odd_elements();
            if (i_odd.is_valid())
            {
                auto mask = tag(level, i_odd) & static_cast<int>(CellFlag::keep);
                apply_on_masked(tag(level - 1, i_odd >> 1),
                                mask,
                                [](auto& e)
                                {
                                    e |= static_cast<int>(CellFlag::refine);
                                });
            }
        }

        template <class T>
        SAMURAI_INLINE void operator()(Dim<2>, T& tag) const
        {
            auto i_even = i.even_elements();
            if (i_even.is_valid())
            {
                auto mask = tag(level, i_even, j) & static_cast<int>(CellFlag::keep);
                apply_on_masked(tag(level - 1, i_even >> 1, j >> 1),
                                mask,
                                [](auto& e)
                                {
                                    e |= static_cast<int>(CellFlag::refine);
                                });

                // xt::masked_view(tag(level - 1, i_even >> 1, j >> 1), mask) |= static_cast<int>(CellFlag::refine);
            }

            auto i_odd = i.odd_elements();
            if (i_odd.is_valid())
            {
                auto mask = tag(level, i_odd, j) & static_cast<int>(CellFlag::keep);
                apply_on_masked(tag(level - 1, i_odd >> 1, j >> 1),
                                mask,
                                [](auto& e)
                                {
                                    e |= static_cast<int>(CellFlag::refine);
                                });
                // xt::masked_view(tag(level - 1, i_odd >> 1, j >> 1), mask) |= static_cast<int>(CellFlag::refine);
            }
        }

        template <class T>
        SAMURAI_INLINE void operator()(Dim<3>, T& tag) const
        {
            auto i_even = i.even_elements();
            if (i_even.is_valid())
            {
                auto mask = tag(level, i_even, j, k) & static_cast<int>(CellFlag::keep);
                apply_on_masked(tag(level - 1, i_even >> 1, j >> 1, k >> 1),
                                mask,
                                [](auto& e)
                                {
                                    e |= static_cast<int>(CellFlag::refine);
                                });
            }

            auto i_odd = i.odd_elements();
            if (i_odd.is_valid())
            {
                auto mask = tag(level, i_odd, j, k) & static_cast<int>(CellFlag::keep);
                apply_on_masked(tag(level - 1, i_odd >> 1, j >> 1, k >> 1),
                                mask,
                                [](auto& e)
                                {
                                    e |= static_cast<int>(CellFlag::refine);
                                });
            }
        }
    };

    template <class... CT>
    SAMURAI_INLINE auto make_graduation_(CT&&... e)
    {
        return make_field_operator_function<make_graduation_op>(std::forward<CT>(e)...);
    }
} // namespace samurai
