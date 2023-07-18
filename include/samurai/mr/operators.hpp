// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <xtensor/xmasked_view.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include "../cell_flag.hpp"
#include "../field.hpp"
#include "../numeric/prediction.hpp"
#include "../operators_base.hpp"

namespace samurai
{
    /********************
     * maximum operator *
     ********************/

    template <class TInterval>
    class maximum_op : public field_operator_base<TInterval>
    {
      public:

        INIT_OPERATOR(maximum_op)

        template <class T>
        inline void operator()(Dim<1>, T& field) const
        {
            xt::xtensor<bool, 1> mask = (field(level + 1, 2 * i) & static_cast<int>(CellFlag::keep))
                                      | (field(level + 1, 2 * i + 1) & static_cast<int>(CellFlag::keep));

            xt::masked_view(field(level + 1, 2 * i), mask) |= static_cast<int>(CellFlag::keep);
            xt::masked_view(field(level + 1, 2 * i + 1), mask) |= static_cast<int>(CellFlag::keep);

            xt::masked_view(field(level, i), mask) |= static_cast<int>(CellFlag::keep);

            mask = (field(level + 1, 2 * i) & static_cast<int>(CellFlag::coarsen))
                 & (field(level + 1, 2 * i + 1) & static_cast<int>(CellFlag::coarsen));

            xt::masked_view(field(level + 1, 2 * i), !mask) &= ~static_cast<unsigned int>(CellFlag::coarsen);
            xt::masked_view(field(level + 1, 2 * i + 1), !mask) &= ~static_cast<unsigned int>(CellFlag::coarsen);
            xt::masked_view(field(level, i), mask) |= static_cast<int>(CellFlag::keep);
        }

        template <class T>
        inline void operator()(Dim<2>, T& field) const
        {
            xt::xtensor<bool, 1> mask = (field(level + 1, 2 * i, 2 * j) & static_cast<int>(CellFlag::keep))
                                      | (field(level + 1, 2 * i + 1, 2 * j) & static_cast<int>(CellFlag::keep))
                                      | (field(level + 1, 2 * i, 2 * j + 1) & static_cast<int>(CellFlag::keep))
                                      | (field(level + 1, 2 * i + 1, 2 * j + 1) & static_cast<int>(CellFlag::keep));

            xt::masked_view(field(level + 1, 2 * i, 2 * j), mask) |= static_cast<int>(CellFlag::keep);
            xt::masked_view(field(level + 1, 2 * i + 1, 2 * j), mask) |= static_cast<int>(CellFlag::keep);
            xt::masked_view(field(level + 1, 2 * i, 2 * j + 1), mask) |= static_cast<int>(CellFlag::keep);
            xt::masked_view(field(level + 1, 2 * i + 1, 2 * j + 1), mask) |= static_cast<int>(CellFlag::keep);

            xt::masked_view(field(level, i, j), mask) |= static_cast<int>(CellFlag::keep);

            mask = (field(level + 1, 2 * i, 2 * j) & static_cast<int>(CellFlag::coarsen))
                 & (field(level + 1, 2 * i + 1, 2 * j) & static_cast<int>(CellFlag::coarsen))
                 & (field(level + 1, 2 * i, 2 * j + 1) & static_cast<int>(CellFlag::coarsen))
                 & (field(level + 1, 2 * i + 1, 2 * j + 1) & static_cast<int>(CellFlag::coarsen));

            xt::masked_view(field(level + 1, 2 * i, 2 * j), !mask) &= ~static_cast<unsigned int>(CellFlag::coarsen);
            xt::masked_view(field(level + 1, 2 * i + 1, 2 * j), !mask) &= ~static_cast<unsigned int>(CellFlag::coarsen);
            xt::masked_view(field(level + 1, 2 * i, 2 * j + 1), !mask) &= ~static_cast<unsigned int>(CellFlag::coarsen);
            xt::masked_view(field(level + 1, 2 * i + 1, 2 * j + 1), !mask) &= ~static_cast<unsigned int>(CellFlag::coarsen);
            xt::masked_view(field(level, i, j), mask) |= static_cast<int>(CellFlag::keep);
        }

        template <class T>
        inline void operator()(Dim<3>, T& field) const
        {
            xt::xtensor<bool, 1> mask = (field(level + 1, 2 * i, 2 * j, 2 * k) & static_cast<int>(CellFlag::keep))
                                      | (field(level + 1, 2 * i + 1, 2 * j, 2 * k) & static_cast<int>(CellFlag::keep))
                                      | (field(level + 1, 2 * i, 2 * j + 1, 2 * k) & static_cast<int>(CellFlag::keep))
                                      | (field(level + 1, 2 * i + 1, 2 * j + 1, 2 * k) & static_cast<int>(CellFlag::keep))
                                      | (field(level + 1, 2 * i, 2 * j, 2 * k + 1) & static_cast<int>(CellFlag::keep))
                                      | (field(level + 1, 2 * i + 1, 2 * j, 2 * k + 1) & static_cast<int>(CellFlag::keep))
                                      | (field(level + 1, 2 * i, 2 * j + 1, 2 * k + 1) & static_cast<int>(CellFlag::keep))
                                      | (field(level + 1, 2 * i + 1, 2 * j + 1, 2 * k + 1) & static_cast<int>(CellFlag::keep));

            xt::masked_view(field(level + 1, 2 * i, 2 * j, 2 * k), mask) |= static_cast<int>(CellFlag::keep);
            xt::masked_view(field(level + 1, 2 * i + 1, 2 * j, 2 * k), mask) |= static_cast<int>(CellFlag::keep);
            xt::masked_view(field(level + 1, 2 * i, 2 * j + 1, 2 * k), mask) |= static_cast<int>(CellFlag::keep);
            xt::masked_view(field(level + 1, 2 * i + 1, 2 * j + 1, 2 * k), mask) |= static_cast<int>(CellFlag::keep);
            xt::masked_view(field(level + 1, 2 * i, 2 * j, 2 * k + 1), mask) |= static_cast<int>(CellFlag::keep);
            xt::masked_view(field(level + 1, 2 * i + 1, 2 * j, 2 * k + 1), mask) |= static_cast<int>(CellFlag::keep);
            xt::masked_view(field(level + 1, 2 * i, 2 * j + 1, 2 * k + 1), mask) |= static_cast<int>(CellFlag::keep);
            xt::masked_view(field(level + 1, 2 * i + 1, 2 * j + 1, 2 * k + 1), mask) |= static_cast<int>(CellFlag::keep);

            xt::masked_view(field(level, i, j, k), mask) |= static_cast<int>(CellFlag::keep);

            mask = (field(level + 1, 2 * i, 2 * j, 2 * k) & static_cast<int>(CellFlag::coarsen))
                 & (field(level + 1, 2 * i + 1, 2 * j, 2 * k) & static_cast<int>(CellFlag::coarsen))
                 & (field(level + 1, 2 * i, 2 * j + 1, 2 * k) & static_cast<int>(CellFlag::coarsen))
                 & (field(level + 1, 2 * i + 1, 2 * j + 1, 2 * k) & static_cast<int>(CellFlag::coarsen))
                 & (field(level + 1, 2 * i, 2 * j, 2 * k + 1) & static_cast<int>(CellFlag::coarsen))
                 & (field(level + 1, 2 * i + 1, 2 * j, 2 * k + 1) & static_cast<int>(CellFlag::coarsen))
                 & (field(level + 1, 2 * i, 2 * j + 1, 2 * k + 1) & static_cast<int>(CellFlag::coarsen))
                 & (field(level + 1, 2 * i + 1, 2 * j + 1, 2 * k + 1) & static_cast<int>(CellFlag::coarsen));

            xt::masked_view(field(level + 1, 2 * i, 2 * j, 2 * k), !mask) &= ~static_cast<unsigned int>(CellFlag::coarsen);
            xt::masked_view(field(level + 1, 2 * i + 1, 2 * j, 2 * k), !mask) &= ~static_cast<unsigned int>(CellFlag::coarsen);
            xt::masked_view(field(level + 1, 2 * i, 2 * j + 1, 2 * k), !mask) &= ~static_cast<unsigned int>(CellFlag::coarsen);
            xt::masked_view(field(level + 1, 2 * i + 1, 2 * j + 1, 2 * k), !mask) &= ~static_cast<unsigned int>(CellFlag::coarsen);
            xt::masked_view(field(level + 1, 2 * i, 2 * j, 2 * k + 1), !mask) &= ~static_cast<unsigned int>(CellFlag::coarsen);
            xt::masked_view(field(level + 1, 2 * i + 1, 2 * j, 2 * k + 1), !mask) &= ~static_cast<unsigned int>(CellFlag::coarsen);
            xt::masked_view(field(level + 1, 2 * i, 2 * j + 1, 2 * k + 1), !mask) &= ~static_cast<unsigned int>(CellFlag::coarsen);
            xt::masked_view(field(level + 1, 2 * i + 1, 2 * j + 1, 2 * k + 1), !mask) &= ~static_cast<unsigned int>(CellFlag::coarsen);
            xt::masked_view(field(level, i, j, k), mask) |= static_cast<int>(CellFlag::keep);
        }
    };

    template <class T>
    inline auto maximum(T&& field)
    {
        return make_field_operator_function<maximum_op>(std::forward<T>(field));
    }

    /**************$$$$$$$$***
     * balance_2to1 operator *
     ****************$$$$$$$$*/

    template <class TInterval>
    class balance_2to1_op : public field_operator_base<TInterval>
    {
      public:

        INIT_OPERATOR(balance_2to1_op)

        template <class T, class stencil_t>
        inline void operator()(Dim<1>, T& cell_flag, const stencil_t& stencil) const
        {
            cell_flag(level, i - stencil[0]) |= (cell_flag(level, i) & static_cast<int>(samurai::CellFlag::keep));
        }

        template <class T, class stencil_t>
        inline void operator()(Dim<2>, T& cell_flag, const stencil_t& stencil) const
        {
            cell_flag(level, i - stencil[0], j - stencil[1]) |= (cell_flag(level, i, j) & static_cast<int>(samurai::CellFlag::keep));
        }

        template <class T, class stencil_t>
        inline void operator()(Dim<3>, T& cell_flag, const stencil_t& stencil) const
        {
            cell_flag(level, i - stencil[0], j - stencil[1], k - stencil[2]) |= (cell_flag(level, i, j, k)
                                                                                 & static_cast<int>(samurai::CellFlag::keep));
        }
    };

    template <class T, class stencil_t>
    inline auto balance_2to1(T&& cell_flag, stencil_t&& stencil)
    {
        return make_field_operator_function<balance_2to1_op>(std::forward<T>(cell_flag), std::forward<stencil_t>(stencil));
    }

    /***************************
     * compute detail operator *
     ***************************/

    template <class TInterval>
    class compute_detail_op : public field_operator_base<TInterval>
    {
      public:

        INIT_OPERATOR(compute_detail_op)

        template <class T, std::size_t order = T::mesh_t::config::prediction_order>
        inline void operator()(Dim<1>, T& detail, const T& field) const
        {
            auto qs_i = xt::eval(Qs_i<order>(field, level, i));

            detail(level + 1, 2 * i) = field(level + 1, 2 * i) - (field(level, i) + qs_i);

            detail(level + 1, 2 * i + 1) = field(level + 1, 2 * i + 1) - (field(level, i) - qs_i);
        }

        template <class T, std::size_t order = T::mesh_t::config::prediction_order>
        inline void operator()(Dim<2>, T& detail, const T& field) const
        {
            auto qs_i  = Qs_i<order>(field, level, i, j);
            auto qs_j  = Qs_j<order>(field, level, i, j);
            auto qs_ij = Qs_ij<order>(field, level, i, j);

            detail(level + 1, 2 * i, 2 * j) = field(level + 1, 2 * i, 2 * j) - (field(level, i, j) + qs_i + qs_j - qs_ij);

            detail(level + 1, 2 * i + 1, 2 * j) = field(level + 1, 2 * i + 1, 2 * j) - (field(level, i, j) - qs_i + qs_j + qs_ij);

            detail(level + 1, 2 * i, 2 * j + 1) = field(level + 1, 2 * i, 2 * j + 1) - (field(level, i, j) + qs_i - qs_j + qs_ij);

            detail(level + 1, 2 * i + 1, 2 * j + 1) = field(level + 1, 2 * i + 1, 2 * j + 1) - (field(level, i, j) - qs_i - qs_j - qs_ij);

            // This is what is done by Bihari and Harten 1999
            // // It seems the good choice.
            // detail(level + 1, 2 * i, 2 * j) =
            //     field(level + 1, 2 * i, 2 * j) -
            //     (field(level, i, j) - qs_i - qs_j + qs_ij);

            // detail(level + 1, 2 * i + 1, 2 * j) =
            //     field(level + 1, 2 * i + 1, 2 * j) -
            //     (field(level, i, j) + qs_i - qs_j - qs_ij);

            // detail(level + 1, 2 * i, 2 * j + 1) =
            //     field(level + 1, 2 * i, 2 * j + 1) -
            //     (field(level, i, j) - qs_i + qs_j - qs_ij);

            // detail(level + 1, 2 * i + 1, 2 * j + 1) =
            //     field(level + 1, 2 * i + 1, 2 * j + 1) -
            //     (field(level, i, j) + qs_i + qs_j + qs_ij);
        }

        template <class T, std::size_t order = T::mesh_t::config::prediction_order>
        inline void operator()(Dim<3>, T& detail, const T& field) const
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
                                                               - (field(level, i, j, k) - qs_i - qs_j - qs_k - qs_ij - qs_ik - qs_jk - qs_ijk);
        }
    };

    template <class T>
    inline auto compute_detail(T&& detail, T&& field)
    {
        return make_field_operator_function<compute_detail_op>(std::forward<T>(detail), std::forward<T>(field));
    }

    template <class TInterval>
    class compute_detail_on_tuple_op : public field_operator_base<TInterval>
    {
      public:

        INIT_OPERATOR(compute_detail_on_tuple_op)

        template <class T1, class T2>
        inline void compute_detail_impl(Dim<1>, std::size_t index, T1& detail, const T2& field) const
        {
            static constexpr std::size_t order = T1::mesh_t::config::prediction_order;
            auto qs_i                          = xt::eval(Qs_i<order>(field, level, i));

            detail(index, level + 1, 2 * i) = field(level + 1, 2 * i) - (field(level, i) + qs_i);

            detail(index, level + 1, 2 * i + 1) = field(level + 1, 2 * i + 1) - (field(level, i) - qs_i);
        }

        template <class T1, class T2>
        inline void compute_detail_impl(Dim<2>, std::size_t index, T1& detail, const T2& field) const
        {
            static constexpr std::size_t order = T1::mesh_t::config::prediction_order;
            auto qs_i                          = Qs_i<order>(field, level, i, j);
            auto qs_j                          = Qs_j<order>(field, level, i, j);
            auto qs_ij                         = Qs_ij<order>(field, level, i, j);

            detail(index, level + 1, 2 * i, 2 * j) = field(level + 1, 2 * i, 2 * j) - (field(level, i, j) + qs_i + qs_j - qs_ij);

            detail(index, level + 1, 2 * i + 1, 2 * j) = field(level + 1, 2 * i + 1, 2 * j) - (field(level, i, j) - qs_i + qs_j + qs_ij);

            detail(index, level + 1, 2 * i, 2 * j + 1) = field(level + 1, 2 * i, 2 * j + 1) - (field(level, i, j) + qs_i - qs_j + qs_ij);

            detail(index, level + 1, 2 * i + 1, 2 * j + 1) = field(level + 1, 2 * i + 1, 2 * j + 1)
                                                           - (field(level, i, j) - qs_i - qs_j - qs_ij);
        }

        template <class T1, class T2>
        inline void compute_detail_impl(Dim<3>, std::size_t index, T1& detail, const T2& field) const
        {
            static constexpr std::size_t order = T1::mesh_t::config::prediction_order;
            auto qs_i                          = Qs_i<order>(field, level, i, j, k);
            auto qs_j                          = Qs_j<order>(field, level, i, j, k);
            auto qs_k                          = Qs_k<order>(field, level, i, j, k);
            auto qs_ij                         = Qs_ij<order>(field, level, i, j, k);
            auto qs_ik                         = Qs_ik<order>(field, level, i, j, k);
            auto qs_jk                         = Qs_jk<order>(field, level, i, j, k);
            auto qs_ijk                        = Qs_ijk<order>(field, level, i, j, k);

            detail(index, level + 1, 2 * i, 2 * j, 2 * k) = field(level + 1, 2 * i, 2 * j, 2 * k)
                                                          - (field(level, i, j, k) + qs_i + qs_j + qs_k - qs_ij - qs_ik - qs_jk + qs_ijk);

            detail(index, level + 1, 2 * i + 1, 2 * j, 2 * k) = field(level + 1, 2 * i + 1, 2 * j, 2 * k)
                                                              - (field(level, i, j, k) - qs_i + qs_j + qs_k + qs_ij + qs_ik - qs_jk - qs_ijk);

            detail(index, level + 1, 2 * i, 2 * j + 1, 2 * k) = field(level + 1, 2 * i, 2 * j + 1, 2 * k)
                                                              - (field(level, i, j, k) + qs_i - qs_j + qs_k + qs_ij - qs_ik + qs_jk - qs_ijk);

            detail(index, level + 1, 2 * i + 1, 2 * j + 1, 2 * k) = field(level + 1, 2 * i + 1, 2 * j + 1, 2 * k)
                                                                  - (field(level, i, j, k) - qs_i - qs_j + qs_k - qs_ij + qs_ik + qs_jk
                                                                     + qs_ijk);

            detail(index, level + 1, 2 * i, 2 * j, 2 * k + 1) = field(level + 1, 2 * i, 2 * j, 2 * k + 1)
                                                              - (field(level, i, j, k) + qs_i + qs_j - qs_k - qs_ij + qs_ik + qs_jk - qs_ijk);

            detail(index, level + 1, 2 * i + 1, 2 * j, 2 * k + 1) = field(level + 1, 2 * i + 1, 2 * j, 2 * k + 1)
                                                                  - (field(level, i, j, k) - qs_i + qs_j - qs_k + qs_ij - qs_ik + qs_jk
                                                                     + qs_ijk);
            detail(index, level + 1, 2 * i, 2 * j + 1, 2 * k + 1) = field(level + 1, 2 * i, 2 * j + 1, 2 * k + 1)
                                                                  - (field(level, i, j, k) + qs_i - qs_j - qs_k + qs_ij + qs_ik - qs_jk
                                                                     + qs_ijk);

            detail(index, level + 1, 2 * i + 1, 2 * j + 1, 2 * k + 1) = field(level + 1, 2 * i + 1, 2 * j + 1, 2 * k + 1)
                                                                      - (field(level, i, j, k) - qs_i - qs_j - qs_k - qs_ij - qs_ik - qs_jk
                                                                         - qs_ijk);
        }

        template <std::size_t dim, class T1, class T2, std::size_t... Is>
        inline void compute_detail_impl(Dim<dim>, T1& detail, const T2& fields, std::index_sequence<Is...>) const
        {
            (compute_detail_impl(Dim<dim>(), Is, detail, std::get<Is>(fields)), ...);
        }

        template <std::size_t dim, class T1, class T2>
        inline void operator()(Dim<dim>, T1& detail, const T2& fields) const
        {
            compute_detail_impl(Dim<dim>(), detail, fields.elements(), std::make_index_sequence<std::tuple_size_v<typename T2::tuple_type>>{});
        }
    };

    template <class Field, class... T>
    inline auto compute_detail(Field& detail, const Field_tuple<T...>& fields)
    {
        return make_field_operator_function<compute_detail_on_tuple_op>(detail, fields);
    }

    /*******************************
     * compute max detail operator *
     *******************************/

    template <class TInterval>
    class compute_max_detail_op : public field_operator_base<TInterval>
    {
      public:

        INIT_OPERATOR(compute_max_detail_op)

        template <class T, class U>
        inline void operator()(Dim<1>, const U& detail, T& max_detail) const
        {
            auto ii       = 2 * i;
            ii.step       = 1;
            auto max_view = xt::view(max_detail, level + 1);

            max_view = xt::maximum(max_view, xt::amax(xt::abs(detail(level + 1, ii)), {0}));
        }

        template <class T, class U>
        inline void operator()(Dim<2>, const U& detail, T& max_detail) const
        {
            auto ii       = 2 * i;
            ii.step       = 1;
            auto max_view = xt::view(max_detail, level + 1);

            max_view = xt::maximum(
                max_view,
                xt::amax(xt::maximum(xt::abs(detail(level + 1, ii, 2 * j)), xt::abs(detail(level + 1, ii, 2 * j + 1))), {0}));
        }

        template <class T, class U>
        inline void operator()(Dim<3>, const U& detail, T& max_detail) const
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
    inline auto compute_max_detail(U&& detail, T&& max_detail)
    {
        return make_field_operator_function<compute_max_detail_op>(std::forward<U>(detail), std::forward<T>(max_detail));
    }

    /*******************************
     * compute max detail operator *
     *******************************/

    template <class TInterval>
    class compute_max_detail_op_ : public field_operator_base<TInterval>
    {
      public:

        INIT_OPERATOR(compute_max_detail_op_)

        template <class T, class U>
        inline void operator()(Dim<1>, const U& detail, T& max_detail) const
        {
            max_detail[level] = std::max(max_detail[level], xt::amax(xt::abs(detail(level, i)))[0]);
        }

        template <class T, class U>
        inline void operator()(Dim<2>, const U& detail, T& max_detail) const
        {
            max_detail[level] = std::max(max_detail[level], xt::amax(xt::abs(detail(level, i, j)))[0]);
        }

        template <class T, class U>
        inline void operator()(Dim<3>, const U& detail, T& max_detail) const
        {
            max_detail[level] = std::max(max_detail[level], xt::amax(xt::abs(detail(level, i, j, k)))[0]);
        }
    };

    template <class T, class U>
    inline auto compute_max_detail_(U&& detail, T&& max_detail)
    {
        return make_field_operator_function<compute_max_detail_op_>(std::forward<U>(detail), std::forward<T>(max_detail));
    }

    /***********************
     * to_coarsen operator *
     ***********************/

    template <class TInterval>
    class to_coarsen_op : public field_operator_base<TInterval>
    {
      public:

        INIT_OPERATOR(to_coarsen_op)

        template <class T, class U, class V>
        inline void operator()(Dim<1>, T& keep, const U& detail, double eps) const
        {
            auto mask = xt::abs(detail(level + 1, 2 * i)) < eps;
            // auto mask = (.5 *
            //              (xt::abs(detail(level + 1, 2 * i)) +
            //               xt::abs(detail(level + 1, 2 * i + 1))) /
            //              max_detail[level + 1]) < eps;

            for (coord_index_t ii = 0; ii < 2; ++ii)
            {
                xt::masked_view(keep(level + 1, 2 * i + ii), mask) = static_cast<int>(CellFlag::coarsen);
            }
        }

        template <class T, class U, class V>
        inline void operator()(Dim<2>, T& keep, const U& detail, double eps) const
        {
            auto mask = xt::abs(detail(level + 1, 2 * i, 2 * j)) < eps;

            // auto mask = (0.25 *
            //              (xt::abs(detail(level + 1, 2 * i, 2 * j)) +
            //               xt::abs(detail(level + 1, 2 * i + 1, 2 * j)) +
            //               xt::abs(detail(level + 1, 2 * i, 2 * j + 1)) +
            //               xt::abs(detail(level + 1, 2 * i + 1, 2 * j + 1))) /
            //              max_detail[level + 1]) < eps;

            for (coord_index_t jj = 0; jj < 2; ++jj)
            {
                for (coord_index_t ii = 0; ii < 2; ++ii)
                {
                    xt::masked_view(keep(level + 1, 2 * i + ii, 2 * j + jj), mask) = static_cast<int>(CellFlag::coarsen);
                }
            }
        }

        template <class T, class U, class V>
        inline void operator()(Dim<3>, T& keep, const U& detail, double eps) const
        {
            auto mask = xt::abs(detail(level + 1, 2 * i, 2 * j, 2 * k)) < eps;

            for (coord_index_t kk = 0; kk < 2; ++kk)
            {
                for (coord_index_t jj = 0; jj < 2; ++jj)
                {
                    for (coord_index_t ii = 0; ii < 2; ++ii)
                    {
                        xt::masked_view(keep(level + 1, 2 * i + ii, 2 * j + jj, 2 * k + kk), mask) = static_cast<int>(CellFlag::coarsen);
                    }
                }
            }
        }
    };

    template <class... CT>
    inline auto to_coarsen(CT&&... e)
    {
        return make_field_operator_function<to_coarsen_op>(std::forward<CT>(e)...);
    }

    /*************************
     * refine_ghost operator *
     *************************/

    template <class TInterval>
    class refine_ghost_op : public field_operator_base<TInterval>
    {
      public:

        INIT_OPERATOR(refine_ghost_op)

        template <class T>
        inline void operator()(Dim<1>, T& flag) const
        {
            auto mask                                 = flag(level + 1, i) & static_cast<int>(CellFlag::keep);
            xt::masked_view(flag(level, i / 2), mask) = static_cast<int>(CellFlag::refine);
        }

        template <class T>
        inline void operator()(Dim<2>, T& flag) const
        {
            auto mask                                        = flag(level + 1, i, j) & static_cast<int>(CellFlag::keep);
            xt::masked_view(flag(level, i / 2, j / 2), mask) = static_cast<int>(CellFlag::refine);
        }

        template <class T>
        inline void operator()(Dim<3>, T& flag) const
        {
            auto mask                                               = flag(level + 1, i, j, k) & static_cast<int>(CellFlag::keep);
            xt::masked_view(flag(level, i / 2, j / 2, k / 2), mask) = static_cast<int>(CellFlag::refine);
        }
    };

    template <class... CT>
    inline auto refine_ghost(CT&&... e)
    {
        return make_field_operator_function<refine_ghost_op>(std::forward<CT>(e)...);
    }

    /********************
     * enlarge operator *
     ********************/

    template <class TInterval>
    class enlarge_op : public field_operator_base<TInterval>
    {
      public:

        INIT_OPERATOR(enlarge_op)

        template <class T>
        inline void operator()(Dim<1>, T& cell_flag) const
        {
            auto keep_mask = cell_flag(level, i) & static_cast<int>(CellFlag::keep);

            for (int ii = -1; ii < 2; ++ii)
            {
                xt::masked_view(cell_flag(level, i + ii), keep_mask) |= static_cast<int>(CellFlag::enlarge);
            }
        }

        template <class T>
        inline void operator()(Dim<2>, T& cell_flag) const
        {
            auto keep_mask = cell_flag(level, i, j) & static_cast<int>(CellFlag::keep);

            for (int jj = -1; jj < 2; ++jj)
            {
                for (int ii = -1; ii < 2; ++ii)
                {
                    xt::masked_view(cell_flag(level, i + ii, j + jj), keep_mask) |= static_cast<int>(CellFlag::enlarge);
                }
            }
        }

        template <class T>
        inline void operator()(Dim<3>, T& cell_flag) const
        {
            auto keep_mask = cell_flag(level, i, j, k) & static_cast<int>(CellFlag::keep);

            for (int kk = -1; kk < 2; ++kk)
            {
                for (int jj = -1; jj < 2; ++jj)
                {
                    for (int ii = -1; ii < 2; ++ii)
                    {
                        xt::masked_view(cell_flag(level, i + ii, j + jj, k + kk), keep_mask) |= static_cast<int>(CellFlag::enlarge);
                    }
                }
            }
        }
    };

    template <class... CT>
    inline auto enlarge(CT&&... e)
    {
        return make_field_operator_function<enlarge_op>(std::forward<CT>(e)...);
    }

    /*******************************
     * keep_around_refine operator *
     *******************************/

    template <class TInterval>
    class keep_around_refine_op : public field_operator_base<TInterval>
    {
      public:

        INIT_OPERATOR(keep_around_refine_op)

        template <class T>
        inline void operator()(Dim<1>, T& cell_flag) const
        {
            auto refine_mask = cell_flag(level, i) & static_cast<int>(CellFlag::refine);

            for (int ii = -1; ii < 2; ++ii)
            {
                xt::masked_view(cell_flag(level, i + ii), refine_mask) |= static_cast<int>(CellFlag::keep);
            }
        }

        template <class T>
        inline void operator()(Dim<2>, T& cell_flag) const
        {
            auto refine_mask = cell_flag(level, i, j) & static_cast<int>(CellFlag::refine);

            for (int jj = -1; jj < 2; ++jj)
            {
                for (int ii = -1; ii < 2; ++ii)
                {
                    xt::masked_view(cell_flag(level, i + ii, j + jj), refine_mask) |= static_cast<int>(CellFlag::keep);
                }
            }
        }

        template <class T>
        inline void operator()(Dim<3>, T& cell_flag) const
        {
            auto refine_mask = cell_flag(level, i, j, k) & static_cast<int>(CellFlag::refine);

            for (int kk = -1; kk < 2; ++kk)
            {
                for (int jj = -1; jj < 2; ++jj)
                {
                    for (int ii = -1; ii < 2; ++ii)
                    {
                        xt::masked_view(cell_flag(level, i + ii, j + jj, k + kk), refine_mask) |= static_cast<int>(CellFlag::keep);
                    }
                }
            }
        }
    };

    template <class... CT>
    inline auto keep_around_refine(CT&&... e)
    {
        return make_field_operator_function<keep_around_refine_op>(std::forward<CT>(e)...);
    }

    /***********************
     * apply_expr operator *
     ***********************/

    template <class TInterval>
    class apply_expr_op : public field_operator_base<TInterval>
    {
      public:

        INIT_OPERATOR(apply_expr_op)

        template <class T, class E>
        inline void operator()(Dim<1>, T& field, const field_expression<E>& e) const
        {
            field(level, i) = e.derived_cast()(level, i);
        }

        template <class T, class E>
        inline void operator()(Dim<2>, T& field, const field_expression<E>& e) const
        {
            field(level, i, j) = e.derived_cast()(level, i, j);
        }

        template <class T, class E>
        inline void operator()(Dim<3>, T& field, const field_expression<E>& e) const
        {
            field(level, i, j, k) = e.derived_cast()(level, i, j, k);
        }
    };

    template <class... CT>
    inline auto apply_expr(CT&&... e)
    {
        return make_field_operator_function<apply_expr_op>(std::forward<CT>(e)...);
    }

    /*******************
     * extend operator *
     *******************/
    template <class TInterval>
    class extend_op : public field_operator_base<TInterval>
    {
      public:

        INIT_OPERATOR(extend_op)

        template <class T>
        inline void operator()(Dim<1>, T& tag) const
        {
            auto refine_mask = tag(level, i) & static_cast<int>(samurai::CellFlag::refine);

            const int added_cells = 1; // 1 by default

            for (int ii = -added_cells; ii < added_cells + 1; ++ii)
            {
                xt::masked_view(tag(level, i + ii), refine_mask) |= static_cast<int>(samurai::CellFlag::keep);
            }
        }

        template <class T>
        inline void operator()(Dim<2>, T& tag) const
        {
            auto refine_mask = tag(level, i, j) & static_cast<int>(samurai::CellFlag::refine);

            for (int jj = -1; jj < 2; ++jj)
            {
                for (int ii = -1; ii < 2; ++ii)
                {
                    xt::masked_view(tag(level, i + ii, j + jj), refine_mask) |= static_cast<int>(samurai::CellFlag::keep);
                }
            }
        }

        template <class T>
        inline void operator()(Dim<3>, T& tag) const
        {
            auto refine_mask = tag(level, i, j, k) & static_cast<int>(samurai::CellFlag::refine);

            for (int kk = -1; kk < 2; ++kk)
            {
                for (int jj = -1; jj < 2; ++jj)
                {
                    for (int ii = -1; ii < 2; ++ii)
                    {
                        xt::masked_view(tag(level, i + ii, j + jj, k + kk), refine_mask) |= static_cast<int>(samurai::CellFlag::keep);
                    }
                }
            }
        }
    };

    template <class... CT>
    inline auto extend(CT&&... e)
    {
        return make_field_operator_function<extend_op>(std::forward<CT>(e)...);
    }

    /****************************
     * make_graduation operator *
     ****************************/

    template <class TInterval>
    class make_graduation_op : public field_operator_base<TInterval>
    {
      public:

        INIT_OPERATOR(make_graduation_op)

        template <class T>
        inline void operator()(Dim<1>, T& tag) const
        {
            auto i_even = i.even_elements();
            if (i_even.is_valid())
            {
                auto mask = tag(level, i_even) & static_cast<int>(CellFlag::keep);
                xt::masked_view(tag(level - 1, i_even >> 1), mask) |= static_cast<int>(CellFlag::refine);
            }

            auto i_odd = i.odd_elements();
            if (i_odd.is_valid())
            {
                auto mask = tag(level, i_odd) & static_cast<int>(CellFlag::keep);
                xt::masked_view(tag(level - 1, i_odd >> 1), mask) |= static_cast<int>(CellFlag::refine);
            }
        }

        template <class T>
        inline void operator()(Dim<2>, T& tag) const
        {
            auto i_even = i.even_elements();
            if (i_even.is_valid())
            {
                auto mask = tag(level, i_even, j) & static_cast<int>(CellFlag::keep);
                xt::masked_view(tag(level - 1, i_even >> 1, j >> 1), mask) |= static_cast<int>(CellFlag::refine);
            }

            auto i_odd = i.odd_elements();
            if (i_odd.is_valid())
            {
                auto mask = tag(level, i_odd, j) & static_cast<int>(CellFlag::keep);
                xt::masked_view(tag(level - 1, i_odd >> 1, j >> 1), mask) |= static_cast<int>(CellFlag::refine);
            }
        }

        template <class T>
        inline void operator()(Dim<3>, T& tag) const
        {
            auto i_even = i.even_elements();
            if (i_even.is_valid())
            {
                auto mask = tag(level, i_even, j, k) & static_cast<int>(CellFlag::keep);
                xt::masked_view(tag(level - 1, i_even >> 1, j >> 1, k >> 1), mask) |= static_cast<int>(CellFlag::refine);
            }

            auto i_odd = i.odd_elements();
            if (i_odd.is_valid())
            {
                auto mask = tag(level, i_odd, j, k) & static_cast<int>(CellFlag::keep);
                xt::masked_view(tag(level - 1, i_odd >> 1, j >> 1, k >> 1), mask) |= static_cast<int>(CellFlag::refine);
            }
        }
    };

    template <class... CT>
    inline auto make_graduation(CT&&... e)
    {
        return make_field_operator_function<make_graduation_op>(std::forward<CT>(e)...);
    }
} // namespace samurai
