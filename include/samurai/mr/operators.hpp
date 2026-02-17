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
                auto qs_i = Qs_i<order>(field, level, i);

                detail(level + 1, 2 * i)     = field(level + 1, 2 * i) - (field(level, i) + qs_i);
                detail(level + 1, 2 * i + 1) = field(level + 1, 2 * i + 1) - (field(level, i) - qs_i);
            }
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
                auto qs_i  = Qs_i<order>(field, level, i, j);
                auto qs_j  = Qs_j<order>(field, level, i, j);
                auto qs_ij = Qs_ij<order>(field, level, i, j);

#ifdef SAMURAI_CHECK_NAN
                if constexpr (T1::is_scalar)
                {
                    for (std::size_t ii = 0; ii < i.size(); ++ii)
                    {
                        if (std::isnan(qs_i(ii)) || std::isnan(qs_j(ii)) || std::isnan(qs_ij(ii)))
                        {
                            local_save(fs::current_path(), "check_nan", {true, true}, field.mesh(), field);
                            throw std::runtime_error(fmt::format("NaN detected during the computation of details at level {}.", level));
                        }
                    }
                }
                else
                {
                    if (xt::any(xt::isnan(qs_ij)))
                    {
                        local_save(fs::current_path(), "check_nan", {true, true}, field.mesh(), field);
                        throw std::runtime_error(fmt::format("NaN detected during the computation of details at level {}.", level));
                    }
                }
#endif

                detail(level + 1, 2 * i, 2 * j)     = field(level + 1, 2 * i, 2 * j) - (field(level, i, j) + qs_i + qs_j - qs_ij);
                detail(level + 1, 2 * i + 1, 2 * j) = field(level + 1, 2 * i + 1, 2 * j) - (field(level, i, j) - qs_i + qs_j + qs_ij);
                detail(level + 1, 2 * i, 2 * j + 1) = field(level + 1, 2 * i, 2 * j + 1) - (field(level, i, j) + qs_i - qs_j + qs_ij);
                detail(level + 1, 2 * i + 1, 2 * j + 1) = field(level + 1, 2 * i + 1, 2 * j + 1) - (field(level, i, j) - qs_i - qs_j - qs_ij);
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
                auto qs_i   = Qs_i<order>(field, level, i, j, k);
                auto qs_j   = Qs_j<order>(field, level, i, j, k);
                auto qs_k   = Qs_k<order>(field, level, i, j, k);
                auto qs_ij  = Qs_ij<order>(field, level, i, j, k);
                auto qs_ik  = Qs_ik<order>(field, level, i, j, k);
                auto qs_jk  = Qs_jk<order>(field, level, i, j, k);
                auto qs_ijk = Qs_ijk<order>(field, level, i, j, k);

                detail(level + 1, 2 * i, 2 * j, 2 * k) = field(level + 1, 2 * i, 2 * j, 2 * k)
                                                       - (field(level, i, j, k) + qs_i + qs_j + qs_k - qs_ij - qs_ik - qs_jk + qs_ijk);
                detail(level + 1, 2 * i + 1, 2 * j, 2 * k) = field(level + 1, 2 * i + 1, 2 * j, 2 * k)
                                                           - (field(level, i, j, k) - qs_i + qs_j + qs_k + qs_ij + qs_ik - qs_jk - qs_ijk);
                detail(level + 1, 2 * i, 2 * j + 1, 2 * k) = field(level + 1, 2 * i, 2 * j + 1, 2 * k)
                                                           - (field(level, i, j, k) + qs_i - qs_j + qs_k + qs_ij - qs_ik + qs_jk - qs_ijk);
                detail(level + 1, 2 * i + 1, 2 * j + 1, 2 * k) = field(level + 1, 2 * i + 1, 2 * j + 1, 2 * k)
                                                               - (field(level, i, j, k) - qs_i - qs_j + qs_k - qs_ij + qs_ik + qs_jk + qs_ijk);
                detail(level + 1, 2 * i, 2 * j, 2 * k + 1) = field(level + 1, 2 * i, 2 * j, 2 * k + 1)
                                                           - (field(level, i, j, k) + qs_i + qs_j - qs_k - qs_ij + qs_ik + qs_jk - qs_ijk);
                detail(level + 1, 2 * i + 1, 2 * j, 2 * k + 1) = field(level + 1, 2 * i + 1, 2 * j, 2 * k + 1)
                                                               - (field(level, i, j, k) - qs_i + qs_j - qs_k + qs_ij - qs_ik + qs_jk + qs_ijk);
                detail(level + 1, 2 * i, 2 * j + 1, 2 * k + 1) = field(level + 1, 2 * i, 2 * j + 1, 2 * k + 1)
                                                               - (field(level, i, j, k) + qs_i - qs_j - qs_k + qs_ij + qs_ik - qs_jk + qs_ijk);
                detail(level + 1, 2 * i + 1, 2 * j + 1, 2 * k + 1) = field(level + 1, 2 * i + 1, 2 * j + 1, 2 * k + 1)
                                                                   - (field(level, i, j, k) - qs_i - qs_j - qs_k - qs_ij - qs_ik - qs_jk
                                                                      - qs_ijk);
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
