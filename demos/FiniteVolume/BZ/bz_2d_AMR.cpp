// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <samurai/algorithm.hpp>
#include <samurai/cell_flag.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/mesh.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/operators.hpp>
#include <samurai/numeric/prediction.hpp>
#include <samurai/numeric/projection.hpp>
#include <samurai/static_algorithm.hpp>
#include <samurai/stencil_field.hpp>

#include "../../LBM/boundary_conditions.hpp"
#include "RockAndRadau/integration_stiff.h"

constexpr size_t dim = 2;

enum class SimpleID
{
    cells            = 0,
    cells_and_ghosts = 1,
    count            = 2,
    reference        = cells_and_ghosts
};

template <>
struct fmt::formatter<SimpleID> : formatter<string_view>
{
    // parse is inherited from formatter<string_view>.
    template <typename FormatContext>
    auto format(SimpleID c, FormatContext& ctx)
    {
        string_view name = "unknown";
        switch (c)
        {
            case SimpleID::cells:
                name = "cells";
                break;
            case SimpleID::cells_and_ghosts:
                name = "cells and ghosts";
                break;
        }
        return formatter<string_view>::format(name, ctx);
    }
};

template <std::size_t dim_>
struct AMRConfig
{
    static constexpr std::size_t dim                  = dim_;
    static constexpr std::size_t max_refinement_level = 20;
    // static constexpr std::size_t ghost_width = 1;
    static constexpr int ghost_width = 2;

    using interval_t = samurai::Interval<int>;
    using mesh_id_t  = SimpleID;
};

template <class Config>
class AMRMesh : public samurai::Mesh_base<AMRMesh<Config>, Config>
{
  public:

    using base_type                  = samurai::Mesh_base<AMRMesh<Config>, Config>;
    using config                     = typename base_type::config;
    static constexpr std::size_t dim = config::dim;

    using mesh_id_t = typename base_type::mesh_id_t;
    using cl_type   = typename base_type::cl_type;
    using lcl_type  = typename base_type::lcl_type;

    using ca_type = typename base_type::ca_type;

    inline AMRMesh(const cl_type& cl, std::size_t min_level, std::size_t max_level)
        : base_type(cl, min_level, max_level)
    {
    }

    inline AMRMesh(const samurai::Box<double, dim>& b, std::size_t start_level, std::size_t min_level, std::size_t max_level)
        : base_type(b, start_level, min_level, max_level)
    {
    }

    inline void update_sub_mesh_impl()
    {
        cl_type cl;
        for_each_interval(this->m_cells[mesh_id_t::cells],
                          [&](std::size_t level, const auto& interval, const auto& index_yz)
                          {
                              lcl_type& lcl = cl[level];
                              samurai::static_nested_loop<dim - 1, -config::ghost_width, config::ghost_width + 1>(
                                  [&](auto stencil)
                                  {
                                      auto index = xt::eval(index_yz + stencil);
                                      lcl[index].add_interval({interval.start - static_cast<int>(config::ghost_width),
                                                               interval.end + static_cast<int>(config::ghost_width)});
                                  });
                          });
        this->m_cells[mesh_id_t::cells_and_ghosts] = {cl, false};
    }
};

/// Timer used in tic & toc
auto tic_timer = std::chrono::high_resolution_clock::now();

/// Launching the timer
void tic()
{
    tic_timer = std::chrono::high_resolution_clock::now();
}

/// Stopping the timer and returning the duration in seconds
double toc()
{
    const auto toc_timer                          = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> time_span = toc_timer - tic_timer;
    return time_span.count();
}

/*
    ======= 2D BZ Test ======

    Equations :
     --
    |  db/dt - D_b * d^2 b/dxx = 1/eps * (b(1-b) + f(q-b)c/(q+b))
    |  dc/dt - D_c * d^2 c/dxx = b-c
     --
*/

template <class Mesh>
auto init_field(Mesh& mesh, const double f = 1.6, const double q = 2.e-3)
{
    using mesh_id_t = typename Mesh::mesh_id_t;
    /*
    field[0] : 'b' in the model
    field[1] : 'c' in the model
    */
    auto field = samurai::make_field<double, 2>("solution", mesh);

    samurai::for_each_cell(mesh[mesh_id_t::reference],
                           [&](auto cell)
                           {
                               double x     = cell.corner(0);
                               double y     = cell.corner(1);
                               double coeff = q * (f + 1) / (f - 1);

                               if (0.3 * x >= y && y >= 0 && x >= 0)
                               {
                                   field[cell][0] = 0.8;
                               }
                               else
                               {
                                   field[cell][0] = coeff;
                               }

                               if (x > 0)
                               {
                                   if (y >= 0)
                                   {
                                       field[cell][1] = coeff + (std::atan(y / x)) / (8 * M_PI * f);
                                   }
                                   else
                                   {
                                       field[cell][1] = coeff + (std::atan(y / x) + 2 * M_PI) / (8 * M_PI * f);
                                   }
                               }
                               else
                               {
                                   if (x < 0)
                                   {
                                       field[cell][1] = coeff + (std::atan(y / x) + M_PI) / (8 * M_PI * f);
                                   }
                                   else
                                   {
                                       if (y >= 0)
                                       {
                                           field[cell][1] = coeff + 1. / 16 / f;
                                       }
                                       else
                                       {
                                           field[cell][1] = coeff + 3. / 16 / f;
                                       }
                                   }
                               }
                           });

    return field;
}

void f_radau(const int* dimension_of_the_system, const double* time, const double* input, double* der_output, const double* rpar, const int* ipar)
{
    const double f       = 1.6;
    const double q       = 2.e-3;
    const double epsilon = 1.e-2;

    der_output[0] = 1.0 / epsilon * (input[0] - input[0] * input[0] + f * (q - input[0]) * input[1] / (q + input[0]));
    der_output[1] = input[0] - input[1];
}

void s_radau(int*, double*, double*, double*, double*, int*, int*, double*, int*, int*)
{
}

template <class Field>
void reaction(Field& field, const double t0, const double t1, const double f = 1.6, const double q = 2.e-3, const double epsilon = 1.e-2)
{
    const double tol = 1e-6;

    using mesh_id_t = typename Field::mesh_t::mesh_id_t;
    auto mesh       = field.mesh();

    samurai::for_each_cell(mesh[mesh_id_t::cells],
                           [&](auto cell)
                           {
                               // TODO: Be sure that sure that we can call with field
                               std::array<double, 2> values{
                                   {field[cell][0], field[cell][1]}
                               };
                               int info[8];

                               radau5_integration(t0, t1, 2, values.data(), f_radau, s_radau, tol, 2, info);

                               field[cell][0] = values[0];
                               field[cell][1] = values[1];
                           });
}

template <class TInterval>
class diffusion_op : public samurai::field_operator_base<TInterval>,
                     public samurai::finite_volume<diffusion_op<TInterval>>
{
  public:

    INIT_OPERATOR(diffusion_op)

    template <class T1, class T2>
    inline auto flux(T1&& ul, T2&& ur) const
    {
        return xt::eval((std::forward<T1>(ur) - std::forward<T2>(ul)) / dx());
    }

    template <class T1>
    inline auto left_flux(const T1& u) const
    {
        return flux(u(level, i - 1, j), u(level, i, j));
    }

    template <class T1>
    inline auto right_flux(const T1& u) const
    {
        return flux(u(level, i, j), u(level, i + 1, j));
    }

    template <class T1>
    inline auto down_flux(const T1& u) const
    {
        return flux(u(level, i, j - 1), u(level, i, j));
    }

    template <class T1>
    inline auto up_flux(const T1& u) const
    {
        return flux(u(level, i, j), u(level, i, j + 1));
    }
};

template <class... CT>
inline auto diffusion(CT&&... e)
{
    return samurai::make_field_operator_function<diffusion_op>(std::forward<CT>(e)...);
}

template <class Field, class Func>
void update_ghosts(Field& field, Func&& update_bc_for_level)
{
    auto mesh             = field.mesh();
    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    amr_projection(field);
    for (std::size_t level = min_level; level <= max_level; ++level)
    {
        update_bc_for_level(field, level);
    }
    amr_prediction(field, std::forward<Func>(update_bc_for_level));
}

template <class Field, class Func>
void RK4(Field& field, const double dt, std::size_t nbstep, Func&& bc, const double D_b = 2.5e-3, const double D_c = 1.5e-3)
{
    using mesh_id_t = typename Field::mesh_t::mesh_id_t;

    auto mesh             = field.mesh();
    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    double dt_rk = dt / nbstep; // Sub time steps
    xt::xtensor_fixed<double, xt::xshape<2>> nu{D_b, D_c};

    auto apply_nu = [&](auto& field)
    {
        samurai::for_each_interval(mesh[mesh_id_t::cells],
                                   [&](std::size_t level, const auto& i, const auto& index)
                                   {
                                       auto j = index[0];
                                       field(level, i, j) *= nu;
                                   });
    };

    Field k1("k1", mesh);
    Field k2("k2", mesh);
    Field k3("k3", mesh);
    Field k4("k4", mesh);

    for (std::size_t nite = 0; nite < nbstep; ++nite)
    {
        update_ghosts(field, bc);
        k1 = diffusion(field);
        apply_nu(k1);
        update_ghosts(k1, bc);

        k2 = diffusion(field + dt_rk / 2. * k1);
        apply_nu(k2);
        update_ghosts(k2, bc);

        k3 = diffusion(field + dt_rk / 2. * k2);
        apply_nu(k3);
        update_ghosts(k3, bc);

        k4 = diffusion(field + dt_rk * k3);
        apply_nu(k4);

        field = field + dt_rk / 6. * (k1 + 2. * k2 + 2. * k3 + k4);
    }
}

template <class Field>
inline void amr_projection(Field& field)
{
    auto mesh       = field.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;

    std::size_t min_level = mesh.min_level(), max_level = mesh.max_level();

    for (std::size_t level = max_level; level >= min_level; --level)
    {
        auto expr = samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells_and_ghosts][level - 1]).on(level - 1);

        expr.apply_op(samurai::projection(field));
    }
}

template <class Field, class Func>
inline void amr_prediction(Field& field, Func&& update_bc_for_level)
{
    auto mesh       = field.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;

    std::size_t min_level = mesh[mesh_id_t::cells].min_level(), max_level = mesh[mesh_id_t::cells].max_level();

    for (std::size_t level = min_level + 1; level <= max_level; ++level)
    {
        auto expr = samurai::intersection(mesh.domain(),
                                          samurai::difference(mesh[mesh_id_t::cells_and_ghosts][level], mesh.get_union()[level]))
                        .on(level);

        expr.apply_op(samurai::prediction<1, false>(field));
        update_bc_for_level(field, level);
    }
}

template <class TInterval>
class enlarge_AMR_op : public samurai::field_operator_base<TInterval>
{
  public:

    INIT_OPERATOR(enlarge_AMR_op)

    template <class T, class T0>
    inline void operator()(samurai::Dim<2>, T0& tmp_flag, T& cell_flag) const
    {
        auto keep_mask = cell_flag(level, i, j) & static_cast<int>(samurai::CellFlag::keep);

        for (int jj = -2; jj < 3; ++jj) // We add two neighbors for each direction
        {
            for (int ii = -2; ii < 3; ++ii)
            {
                xt::masked_view(tmp_flag(level, i + ii, j + jj), keep_mask) = static_cast<int>(samurai::CellFlag::keep);
            }
        }
    }
};

template <class... CT>
inline auto enlarge_AMR(CT&&... e)
{
    return samurai::make_field_operator_function<enlarge_AMR_op>(std::forward<CT>(e)...);
}

template <class Field, class Func, class Tag>
void AMR_criterion(Field& f, Func&& update_bc_for_level, Tag& tag, std::size_t ite)
{
    auto mesh             = f.mesh();
    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    tag.fill(static_cast<int>(samurai::CellFlag::keep)); // Important

    amr_projection(f);
    for (std::size_t level = min_level; level <= max_level; ++level)
    {
        update_bc_for_level(f, level);
    }
    amr_prediction(f, std::forward<Func>(update_bc_for_level));

    // The fact of considering at which iteration of the refinement process we
    // are lets one get rid of oscillations which generate an infinite loop in
    // the process
    for (std::size_t level = min_level; level <= max_level - ite; ++level)
    {
        double dx = samurai::cell_length(level);

        auto leaves = samurai::intersection(mesh[SimpleID::cells][level], mesh[SimpleID::cells][level]);

        leaves(
            [&](auto& interval, auto& index)
            {
                auto k = interval; // Logical index in x
                auto h = index[0]; // Logical index in y

                auto der_x = xt::eval(f(0, level, k + 1, h) - f(0, level, k - 1, h)) / (2. * dx); // Approximation of
                                                                                                  // db/dx
                auto der_y = xt::eval(f(0, level, k, h + 1) - f(0, level, k, h - 1)) / (2. * dx); // Approximation of
                                                                                                  // db/dy

                auto grad_abs      = xt::sqrt(xt::pow(der_x, 2.) + xt::pow(der_y, 2.)); // Computing Euclidian norm
                auto grad_abs_norm = grad_abs / xt::abs(f(0, level, k, h));             // Dividing by the field itself

                auto mask = grad_abs_norm > 0.1 / dx; // Criterion

                if (level == max_level)
                {
                    xt::masked_view(tag(level, k, h), mask)  = static_cast<int>(samurai::CellFlag::keep);
                    xt::masked_view(tag(level, k, h), !mask) = static_cast<int>(samurai::CellFlag::coarsen);
                }
                else
                {
                    if (level == min_level)
                    {
                        xt::masked_view(tag(level, k, h), mask)  = static_cast<int>(samurai::CellFlag::refine);
                        xt::masked_view(tag(level, k, h), !mask) = static_cast<int>(samurai::CellFlag::keep);
                        // tag(level, k) =
                        // static_cast<int>(samurai::CellFlag::keep);
                    }
                    else
                    {
                        xt::masked_view(tag(level, k, h), mask)  = static_cast<int>(samurai::CellFlag::refine);
                        xt::masked_view(tag(level, k, h), !mask) = static_cast<int>(samurai::CellFlag::coarsen);
                    }
                }
            });
    }
    // Here we copy the tag field because otherwise we modify the field
    // which is then used to decide where to enlarge.
    // This problem was solved in multiresolution by adding the choice of
    // the flag enlarge, but to do this we should change other parts of the code
    auto tag_tmp = samurai::make_field<int, 1>("tag", mesh);
    tag_tmp.fill(static_cast<int>(samurai::CellFlag::keep));

    for (std::size_t level = min_level; level <= max_level; ++level)
    {
        auto subset = samurai::intersection(mesh[SimpleID::cells][level], mesh[SimpleID::cells][level]);

        subset.apply_op(samurai::copy(tag_tmp, tag));
    }

    for (std::size_t level = min_level; level <= max_level - ite; ++level)
    {
        auto leaves = samurai::intersection(mesh[SimpleID::cells][level], mesh[SimpleID::cells][level]);
        leaves.apply_op(enlarge_AMR(tag_tmp, tag));
    }
    std::swap(tag.array(), tag_tmp.array());
}

template <class Field>
void make_graduation(Field& tag)
{
    auto mesh = tag.mesh();
    for (std::size_t level = mesh.max_level(); level >= 1; --level)
    {
        auto ghost_subset = samurai::intersection(mesh[SimpleID::cells][level], mesh[SimpleID::reference][level - 1]).on(level - 1);
        ghost_subset(
            [&](const auto& i, const auto& index)
            {
                auto j = index[0];
                tag(level - 1, i, j) |= static_cast<int>(samurai::CellFlag::keep);
            });

        auto subset_2 = intersection(mesh[SimpleID::cells][level], mesh[SimpleID::cells][level]);

        subset_2(
            [&](const auto& interval, const auto& index)
            {
                auto i                    = interval;
                auto j                    = index[0];
                xt::xtensor<bool, 1> mask = (tag(level, i, j) & static_cast<int>(samurai::CellFlag::refine));

                for (int jj = -1; jj < 2; ++jj)
                {
                    for (int ii = -1; ii < 2; ++ii)
                    {
                        xt::masked_view(tag(level, i + ii, j + jj), mask) |= static_cast<int>(samurai::CellFlag::keep);
                    }
                }
            });

        auto keep_subset = samurai::intersection(mesh[SimpleID::cells][level], mesh[SimpleID::cells][level]).on(level - 1);
        keep_subset(
            [&](const auto& interval, const auto& index)
            {
                auto i = interval;
                auto j = index[0];

                xt::xtensor<bool, 1> mask = (tag(level, 2 * i, 2 * j) & static_cast<int>(samurai::CellFlag::keep))
                                          | (tag(level, 2 * i + 1, 2 * j) & static_cast<int>(samurai::CellFlag::keep))
                                          | (tag(level, 2 * i, 2 * j + 1) & static_cast<int>(samurai::CellFlag::keep))
                                          | (tag(level, 2 * i + 1, 2 * j + 1) & static_cast<int>(samurai::CellFlag::keep));

                xt::masked_view(tag(level, 2 * i, 2 * j), mask) |= static_cast<int>(samurai::CellFlag::keep);
                xt::masked_view(tag(level, 2 * i + 1, 2 * j), mask) |= static_cast<int>(samurai::CellFlag::keep);
                xt::masked_view(tag(level, 2 * i, 2 * j + 1), mask) |= static_cast<int>(samurai::CellFlag::keep);
                xt::masked_view(tag(level, 2 * i + 1, 2 * j + 1), mask) |= static_cast<int>(samurai::CellFlag::keep);
            });

        xt::xtensor_fixed<int, xt::xshape<4, dim>> stencil{
            {1,  1 },
            {-1, -1},
            {-1, 1 },
            {1,  -1}
        };

        for (std::size_t i = 0; i < stencil.shape()[0]; ++i)
        {
            auto s = xt::view(stencil, i);
            auto subset = samurai::intersection(samurai::translate(mesh[SimpleID::cells][level], s), mesh[SimpleID::cells][level - 1]).on(level);

            subset(
                [&](const auto& interval, const auto& index)
                {
                    auto j_f = index[0];
                    auto i_f = interval.even_elements();

                    if (i_f.is_valid())
                    {
                        auto mask = tag(level, i_f - s[0], j_f - s[1]) & static_cast<int>(samurai::CellFlag::refine);
                        auto i_c  = i_f >> 1;
                        auto j_c  = j_f >> 1;
                        xt::masked_view(tag(level - 1, i_c, j_c), mask) |= static_cast<int>(samurai::CellFlag::refine);

                        mask = tag(level, i_f - s[0], j_f - s[1]) & static_cast<int>(samurai::CellFlag::keep);
                        xt::masked_view(tag(level - 1, i_c, j_c), mask) |= static_cast<int>(samurai::CellFlag::keep);
                    }

                    i_f = interval.odd_elements();
                    if (i_f.is_valid())
                    {
                        auto mask = tag(level, i_f - s[0], j_f - s[1]) & static_cast<int>(samurai::CellFlag::refine);
                        auto i_c  = i_f >> 1;
                        auto j_c  = j_f >> 1;
                        xt::masked_view(tag(level - 1, i_c, j_c), mask) |= static_cast<int>(samurai::CellFlag::refine);

                        mask = tag(level, i_f - s[0], j_f - s[1]) & static_cast<int>(samurai::CellFlag::keep);
                        xt::masked_view(tag(level - 1, i_c, j_c), mask) |= static_cast<int>(samurai::CellFlag::keep);
                    }
                });
        }
    }
}

template <class Field, class Tag>
bool update_mesh(Field& f, const Tag& tag)
{
    using mesh_t        = typename Field::mesh_t;
    using interval_t    = typename mesh_t::interval_t;
    using coord_index_t = typename interval_t::coord_index_t;
    using cl_type       = typename mesh_t::cl_type;

    auto mesh = f.mesh();

    cl_type cell_list;

    samurai::for_each_interval(mesh[SimpleID::cells],
                               [&](std::size_t level, const auto& interval, const auto& index_yz)
                               {
                                   for (int i = interval.start; i < interval.end; ++i)
                                   {
                                       if (tag[i + interval.index] & static_cast<int>(samurai::CellFlag::refine))
                                       {
                                           samurai::static_nested_loop<dim - 1, 0, 2>(
                                               [&](auto stencil)
                                               {
                                                   auto index = 2 * index_yz + stencil;
                                                   cell_list[level + 1][index].add_interval({2 * i, 2 * i + 2});
                                               });
                                       }
                                       else if (tag[i + interval.index] & static_cast<int>(samurai::CellFlag::keep))
                                       {
                                           cell_list[level][index_yz].add_point(i);
                                       }
                                       else
                                       {
                                           cell_list[level - 1][index_yz >> 1].add_point(i >> 1);
                                       }
                                   }
                               });

    mesh_t new_mesh(cell_list, mesh.min_level(), mesh.max_level());

    if (new_mesh == mesh)
    {
        return true;
    }

    Field new_f{f.name(), new_mesh};
    new_f.fill(0.);

    for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
    {
        auto subset = samurai::intersection(mesh[SimpleID::cells][level], new_mesh[SimpleID::cells][level]);

        subset.apply_op(samurai::copy(new_f, f));
    }

    for (std::size_t level = mesh.min_level() + 1; level <= mesh.max_level(); ++level)
    {
        auto subset = samurai::intersection(mesh[SimpleID::cells][level], new_mesh[SimpleID::cells][level - 1]).on(level - 1);
        subset.apply_op(projection(new_f, f));

        auto set_refine = intersection(new_mesh[SimpleID::cells][level], mesh[SimpleID::cells][level - 1]).on(level - 1);
        set_refine.apply_op(samurai::prediction<1, true>(new_f, f));
    }

    f.mesh_ptr()->swap(new_mesh);
    std::swap(f.array(), new_f.array());

    return false;
}

int main()
{
    using Config     = AMRConfig<dim>;
    using interval_t = typename Config::interval_t;

    std::size_t max_level = 8; // Maximum level of resolution
    std::size_t min_level = 1; // Minimum level of resolution
    samurai::Box<double, dim> box{
        {-1, -1},
        {1,  1 }
    }; // Domain [-1, 1]^2

    const double D_b     = 2.5e-3; // Diffusion coefficient 'b'
    const double D_c     = 1.5e-3; // Diffusion coefficient 'c'
    const double epsilon = 1.e-2;  // Stiffness parameter

    const double dx = 1. / (1 << max_level); // Space step
    const double Tf = 1.;

    AMRMesh<Config> mesh{box, max_level, min_level, max_level};

    auto field = init_field(mesh); // Initializing solution field

    auto update_bc_for_level = [](auto& field, std::size_t level)
    {
        update_bc_D2Q4_3_Euler_constant_extension(field, level);
    };

    double t           = 0.;
    std::size_t nb_ite = 0;
    std::size_t nsave  = 0;

    double dt           = 1.e-3;                                      // Time step (splitting time)
    double dt_diffusion = 0.25 * dx * dx / (2. * std::max(D_b, D_c)); // Diffusion time step

    samurai::save(std::string("bz_AMR_init_before"), mesh, field); // Saving

    while (t < Tf)
    {
        fmt::print(fmt::format("Iteration = {:4d}, t: {}\n", nb_ite, t));

        std::size_t idx = 0;

        while (true)
        {
            auto tag = samurai::make_field<int, 1>("tag", mesh);
            AMR_criterion(field, update_bc_for_level, tag, idx);
            make_graduation(tag);
            if (update_mesh(field, tag))
            {
                break;
            }
            idx++;
        }

        tic();
        reaction(field, t, t + .5 * dt);
        auto duration = toc();
        fmt::print(fmt::format("first reaction: {}\n", duration));

        tic();
        RK4(field, dt, std::ceil(dt / dt_diffusion), update_bc_for_level, D_b, D_c);
        duration = toc();
        fmt::print(fmt::format("diffusion: {}\n", duration));
        /*
        rock4_integration(double tini, double tend, int neq, double *u,
                    func_rock fcn, double tol, int *info)
        */
        tic();
        reaction(field, t + .5 * dt, t + dt);
        duration = toc();
        fmt::print(fmt::format("second reaction: {}\n", duration));

        if (nsave == 20)
        {
            samurai::save(fmt::format("bz_AMR_ite-{}", nb_ite), mesh, field); // Saving
            nsave = 0;
        }
        nsave++;
        t += dt;
        nb_ite++;
    }

    return 0;
}
