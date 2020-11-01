#pragma once

#include <mure/operators_base.hpp>

#define EPS_G 5.e-5
#define EPS_F 1e-1

namespace mure
{
    template<class TInterval>
    class compute_gradient_op : public field_operator_base<TInterval> {
        public:
        INIT_OPERATOR(compute_gradient_op)

        template<class T>
        inline void operator()(Dim<1>, const T &u, T &grad) const
        {
                double eps = 1e-3;
                auto max_left = xt::maximum(xt::abs(u(level, i)), xt::abs(u(level, i-1)));
                auto mask_left = max_left < eps;
                auto left = xt::eval(xt::abs(u(level, i) - u(level, i-1))/max_left);
                xt::masked_view(left, mask_left) = 0.;
                left = xt::maximum(xt::minimum(left, 1), 0);

                auto max_right = xt::maximum(xt::abs(u(level, i)), xt::abs(u(level, i+1)));
                auto mask_right = max_right < eps;
                auto right = xt::eval(xt::abs(u(level, i+1) - u(level, i))/max_right);
                xt::masked_view(right, mask_right) = 0.;
                right = xt::maximum(xt::minimum(right, 1), 0);

                grad(level, i, j) = xt::maximum(left, right);
        }

        template<class T>
        inline void operator()(Dim<2>, const T &u, T &grad) const
        {
                // auto grad_x = xt::abs(.5*(u(level, i+1, j  ) - u(level, i-1, j  ))/dx);
                // auto grad_y = xt::abs(.5*(u(level, i  , j+1) - u(level, i  , j-1))/dx);
                // grad(level, i, j) = grad_x + grad_y;

                double eps = 1e-3;
                auto max_left = xt::maximum(xt::abs(u(level, i, j)), xt::abs(u(level, i-1, j)));
                auto mask_left = max_left < eps;
                auto left = xt::eval(xt::abs(u(level, i, j) - u(level, i-1, j))/max_left);
                xt::masked_view(left, mask_left) = 0.;
                left = xt::maximum(xt::minimum(left, 1), 0);

                auto max_right = xt::maximum(xt::abs(u(level, i, j)), xt::abs(u(level, i+1, j)));
                auto mask_right = max_right < eps;
                auto right = xt::eval(xt::abs(u(level, i+1, j) - u(level, i, j))/max_right);
                xt::masked_view(right, mask_right) = 0.;
                right = xt::maximum(xt::minimum(right, 1), 0);

                auto max_down = xt::maximum(xt::abs(u(level, i, j-1)), xt::abs(u(level, i, j)));
                auto mask_down = max_down < eps;
                auto down = xt::eval(xt::abs(u(level, i, j) - u(level, i, j-1))/max_down);
                xt::masked_view(down, mask_down) = 0.;
                down = xt::maximum(xt::minimum(down, 1), 0);

                auto max_up = xt::maximum(xt::abs(u(level, i, j)), xt::abs(u(level, i, j+1)));
                auto mask_up = max_up < eps;
                auto up = xt::eval(xt::abs(u(level, i, j+1) - u(level, i, j))/max_up);
                xt::masked_view(up, mask_up) = 0.;
                up = xt::maximum(xt::minimum(up, 1), 0);
                grad(level, i, j) = xt::maximum(xt::maximum(left, right),
                                                xt::maximum(up, down));

                // auto max_right = xt::maximum(xt::maximum(xt::abs(u(level, i+1, j)), xt::abs(u(level, i, j))), eps);
                // auto right = xt::abs(u(level, i+1, j) - u(level, i, j))/max_right;
                // auto max_up = xt::maximum(xt::maximum(xt::abs(u(level, i, j+1)), xt::abs(u(level, i, j))), eps);
                // auto up = xt::abs(u(level, i, j+1) - u(level, i, j))/max_up;
                // auto max_down = xt::maximum(xt::maximum(xt::abs(u(level, i, j)), xt::abs(u(level, i, j-1))), eps);
                // auto down = xt::abs(u(level, i, j) - u(level, i, j-1))/max_down;
                // grad(level, i, j) = xt::maximum(xt::maximum(left, right),
                //                                 xt::maximum(up, down));

                // double eps = 1e-6;
                // auto max_left = xt::maximum(xt::maximum(xt::abs(u(level, i, j)), xt::abs(u(level, i-1, j))), eps);
                // auto left = xt::abs(u(level, i, j) - u(level, i-1, j))/max_left;
                // auto max_right = xt::maximum(xt::maximum(xt::abs(u(level, i+1, j)), xt::abs(u(level, i, j))), eps);
                // auto right = xt::abs(u(level, i+1, j) - u(level, i, j))/max_right;
                // auto max_up = xt::maximum(xt::maximum(xt::abs(u(level, i, j+1)), xt::abs(u(level, i, j))), eps);
                // auto up = xt::abs(u(level, i, j+1) - u(level, i, j))/max_up;
                // auto max_down = xt::maximum(xt::maximum(xt::abs(u(level, i, j)), xt::abs(u(level, i, j-1))), eps);
                // auto down = xt::abs(u(level, i, j) - u(level, i, j-1))/max_down;
                // grad(level, i, j) = xt::maximum(xt::maximum(left, right),
                //                                 xt::maximum(up, down));
        }
    };

    template<class... CT>
    inline auto compute_gradient(CT &&... e)
    {
        return make_field_operator_function<compute_gradient_op>(
            std::forward<CT>(e)...);
    }

    template<class TInterval>
    class to_coarsen_amr_op : public field_operator_base<TInterval> {
        public:
        INIT_OPERATOR(to_coarsen_amr_op)

        template<class T1, class T2>
        inline void operator()(Dim<1>, const T1& grad, T2 &tag, std::size_t min_level) const
        {
            auto mask = (grad(level, 2*i  ) < EPS_G) and
                        (grad(level, 2*i+1) < EPS_G);

            if (level > min_level)
            {
                xt::masked_view(tag(level, 2*i  ), mask) = static_cast<int>(mure::CellFlag::coarsen);
                xt::masked_view(tag(level, 2*i+1), mask) = static_cast<int>(mure::CellFlag::coarsen);
            }
        }

        template<class T1, class T2>
        inline void operator()(Dim<2>, const T1& grad, T2 &tag, std::size_t min_level) const
        {
            auto mask = (grad(level, 2*i  ,   2*j) < EPS_G) and
                        (grad(level, 2*i+1,   2*j) < EPS_G) and
                        (grad(level, 2*i  , 2*j+1) < EPS_G) and
                        (grad(level, 2*i+1, 2*j+1) < EPS_G);

            if (level > min_level)
            {
                xt::masked_view(tag(level, 2*i  ,   2*j), mask) = static_cast<int>(mure::CellFlag::coarsen);
                xt::masked_view(tag(level, 2*i+1,   2*j), mask) = static_cast<int>(mure::CellFlag::coarsen);
                xt::masked_view(tag(level, 2*i  , 2*j+1), mask) = static_cast<int>(mure::CellFlag::coarsen);
                xt::masked_view(tag(level, 2*i+1, 2*j+1), mask) = static_cast<int>(mure::CellFlag::coarsen);
            }
        }
    };

    template<class... CT>
    inline auto to_coarsen_amr(CT &&... e)
    {
        return make_field_operator_function<to_coarsen_amr_op>(
            std::forward<CT>(e)...);
    }

    template<class TInterval>
    class to_refine_amr_op : public field_operator_base<TInterval> {
        public:
        INIT_OPERATOR(to_refine_amr_op)

        template<class T1, class T2>
        inline void operator()(Dim<1>, const T1& grad, T2 &tag, std::size_t max_level) const
        {
            if (level < max_level)
            {
                auto mask1 = grad(level, 2*i) > EPS_F;
                xt::masked_view(tag(level, 2*i), mask1) = static_cast<int>(mure::CellFlag::refine);
                auto mask2 = grad(level, 2*i+1) > EPS_F;
                xt::masked_view(tag(level, 2*i+1), mask2) = static_cast<int>(mure::CellFlag::refine);
            }
        }

        template<class T1, class T2>
        inline void operator()(Dim<2>, const T1& grad, T2 &tag, std::size_t max_level) const
        {
            if (level < max_level)
            {
                auto mask1 = grad(level, 2*i, 2*j) > EPS_F;
                xt::masked_view(tag(level, 2*i, 2*j), mask1) = static_cast<int>(mure::CellFlag::refine);
                auto mask2 = grad(level, 2*i+1, 2*j) > EPS_F;
                xt::masked_view(tag(level, 2*i+1, 2*j), mask2) = static_cast<int>(mure::CellFlag::refine);
                auto mask3 = grad(level, 2*i, 2*j+1) > EPS_F;
                xt::masked_view(tag(level, 2*i, 2*j+1), mask3) = static_cast<int>(mure::CellFlag::refine);
                auto mask4 = grad(level, 2*i+1, 2*j+1) > EPS_F;
                xt::masked_view(tag(level, 2*i+1, 2*j+1), mask4) = static_cast<int>(mure::CellFlag::refine);
            }
        }
    };

    template<class... CT>
    inline auto to_refine_amr(CT &&... e)
    {
        return make_field_operator_function<to_refine_amr_op>(
            std::forward<CT>(e)...);
    }

    template<class TInterval>
    class extend_op : public field_operator_base<TInterval> {
        public:
        INIT_OPERATOR(extend_op)

        template<class T>
        inline void operator()(Dim<1>, T &tag) const
        {
            auto refine_mask =
                tag(level, i) & static_cast<int>(mure::CellFlag::refine);


            int added_cells = 2; // 1 by default

            for (int ii = -added_cells; ii < added_cells + 1; ++ii)
            {
                xt::masked_view(tag(level, i + ii), refine_mask) |= static_cast<int>(mure::CellFlag::keep);
            }
        }

        template<class T>
        inline void operator()(Dim<2>, T &tag) const
        {
            auto refine_mask =
                tag(level, i, j) & static_cast<int>(mure::CellFlag::refine);

            for (int jj = -1; jj < 2; ++jj)
            {
                for (int ii = -1; ii < 2; ++ii)
                {
                    xt::masked_view(tag(level, i + ii, j + jj), refine_mask) |= static_cast<int>(mure::CellFlag::keep);
                }
            }
        }
    };

    template<class... CT>
    inline auto extend(CT &&... e)
    {
        return make_field_operator_function<extend_op>(
            std::forward<CT>(e)...);
    }

    template<class TInterval>
    class make_graduation_op : public field_operator_base<TInterval> {
        public:
        INIT_OPERATOR(make_graduation_op)

        template<class T>
        inline void operator()(Dim<1>, T &tag) const
        {
            auto i_even = i.even_elements();
            if (i_even.is_valid())
            {
                auto mask = tag(level, i_even) & static_cast<int>(CellFlag::keep);
                xt::masked_view(tag(level-1, i_even>>1), mask) |= static_cast<int>(CellFlag::refine);
            }

            auto i_odd = i.odd_elements();
            if (i_odd.is_valid())
            {
                auto mask = tag(level, i_odd) & static_cast<int>(CellFlag::keep);
                xt::masked_view(tag(level-1, i_odd>>1), mask) |= static_cast<int>(CellFlag::refine);
            }
        }

        template<class T>
        inline void operator()(Dim<2>, T &tag) const
        {
            auto i_even = i.even_elements();
            if (i_even.is_valid())
            {
                auto mask = tag(level, i_even, j) & static_cast<int>(CellFlag::keep);
                xt::masked_view(tag(level-1, i_even>>1, j>>1), mask) |= static_cast<int>(CellFlag::refine);
            }

            auto i_odd = i.odd_elements();
            if (i_odd.is_valid())
            {
                auto mask = tag(level, i_odd, j) & static_cast<int>(CellFlag::keep);
                xt::masked_view(tag(level-1, i_odd>>1, j>>1), mask) |= static_cast<int>(CellFlag::refine);
            }
        }
    };

    template<class... CT>
    inline auto make_graduation(CT &&... e)
    {
        return make_field_operator_function<make_graduation_op>(
            std::forward<CT>(e)...);
    }

    template<class interval_t, class coord_index_t, class field_t>
    inline void compute_new_u_impl(
        const std::size_t level, const interval_t i,
        const xt::xtensor_fixed<coord_index_t, xt::xshape<0>> &index_yz,
        const field_t &field, field_t &new_field)
    {
        new_field(level+1,     2*i) = field(level, i);
        new_field(level+1, 2*i + 1) = field(level, i);
    }

    template<class interval_t, class coord_index_t, class field_t>
    inline void compute_new_u_impl(
        const std::size_t level, const interval_t i,
        const xt::xtensor_fixed<coord_index_t, xt::xshape<1>> &index_yz,
        const field_t &field, field_t &new_field)
    {
        auto j = index_yz[0];
        new_field(level+1, 2*i    ,   2*j) = field(level, i, j);
        new_field(level+1, 2*i + 1,   2*j) = field(level, i, j);
        new_field(level+1, 2*i    , 2*j+1) = field(level, i, j);
        new_field(level+1, 2*i + 1, 2*j+1) = field(level, i, j);

    }

    template<class interval_t, class coord_index_t, class field_t,
             std::size_t dim>
    inline void compute_new_u(
        const std::size_t level, const interval_t i,
        const xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> &index_yz,
        const field_t &field, field_t &new_field)
    {
        compute_new_u_impl(level, i, index_yz, field, new_field);
    }

    template<class TInterval>
    class amr_pred_op : public field_operator_base<TInterval> {
        public:
        INIT_OPERATOR(amr_pred_op)

        template<class T>
        inline void operator()(Dim<1>, T &field) const
        {
            auto even_i = i.even_elements();
            if (even_i.is_valid())
                field(level, even_i) = field(level-1, even_i >> 1);

            auto odd_i = i.odd_elements();
            if (odd_i.is_valid())
                field(level, odd_i) = field(level-1, odd_i >> 1);
        }

        template<class T>
        inline void operator()(Dim<2>, T &field) const
        {
            auto even_i = i.even_elements();
            if (even_i.is_valid())
                field(level, even_i, j) = field(level-1, even_i >> 1, j >> 1);

            auto odd_i = i.odd_elements();
            if (odd_i.is_valid())
                field(level, odd_i, j) = field(level-1, odd_i >> 1, j >> 1);
        }
    };

    template<class... CT>
    inline auto amr_pred(CT &&... e)
    {
        return make_field_operator_function<amr_pred_op>(
            std::forward<CT>(e)...);
    }

    template<class MRConfig>
    inline void amr_prediction(Field<MRConfig> &field)
    {
        constexpr auto max_refinement_level = MRConfig::max_refinement_level;

        auto mesh = field.mesh();
        for (std::size_t level = 1; level <= max_refinement_level; ++level)
        {

            if (!mesh[MeshType::cells][level].empty())
            {
                auto expr =
                    intersection(
                        difference(mesh[MeshType::all_cells][level],
                                    union_(mesh[MeshType::cells][level],
                                            mesh[MeshType::proj_cells][level])),
                        mesh.initial_mesh())
                        .on(level);

                expr.apply_op(amr_pred(field));
            }
        }
    }

}