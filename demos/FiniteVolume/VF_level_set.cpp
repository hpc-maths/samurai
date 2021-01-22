// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <samurai/mesh.hpp>
#include <samurai/mr/cell_flag.hpp>
#include <samurai/mr/operators.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>

#include "stencil_field.hpp"

#include "../LBM/boundary_conditions.hpp"

#include <chrono>

constexpr size_t dim = 2;

enum class SimpleID
{
    cells = 0,
    cells_and_ghosts = 1,
    count = 2,
    reference = cells_and_ghosts
};

template <>
struct fmt::formatter<SimpleID>: formatter<string_view>
{
  // parse is inherited from formatter<string_view>.
  template <typename FormatContext>
  auto format(SimpleID c, FormatContext& ctx) {
    string_view name = "unknown";
    switch (c) {
    case SimpleID::cells:            name = "cells"; break;
    case SimpleID::cells_and_ghosts: name = "cells and ghosts"; break;
    }
    return formatter<string_view>::format(name, ctx);
  }
};

template <std::size_t dim_>
struct AMRConfig
{
    static constexpr std::size_t dim = dim_;
    static constexpr std::size_t max_refinement_level = 20;
    // static constexpr int ghost_width = 1;
    static constexpr int ghost_width = 2;

    using interval_t = samurai::Interval<int>;
    using mesh_id_t = SimpleID;
};

template <class Config>
class AMRMesh: public samurai::Mesh_base<AMRMesh<Config>, Config>
{
public:
    using base_type = samurai::Mesh_base<AMRMesh<Config>, Config>;
    using config = typename base_type::config;
    static constexpr std::size_t dim = config::dim;

    using mesh_id_t = typename base_type::mesh_id_t;
    using cl_type = typename base_type::cl_type;
    using lcl_type = typename base_type::lcl_type;

    using ca_type = typename base_type::ca_type;

    inline AMRMesh(const cl_type &cl, std::size_t min_level, std::size_t max_level)
    : base_type(cl, min_level, max_level)
    {}

    inline AMRMesh(const samurai::Box<double, dim>& b, std::size_t start_level, std::size_t min_level, std::size_t max_level)
    : base_type(b, start_level, min_level, max_level)
    {}

    inline void update_sub_mesh_impl()
    {
        cl_type cl;
        for_each_interval(this->m_cells[mesh_id_t::cells], [&](std::size_t level, const auto& interval, const auto& index_yz)
        {
            lcl_type& lcl = cl[level];
            samurai::static_nested_loop<dim - 1, -config::ghost_width, config::ghost_width + 1>([&](auto stencil)
            {
                auto index = xt::eval(index_yz + stencil);
                lcl[index].add_interval({interval.start - config::ghost_width,
                                         interval.end + config::ghost_width});
            });
        });
        this->m_cells[mesh_id_t::cells_and_ghosts] = {cl, false};
    }
};

template<class TInterval>
class projection_op_: public samurai::field_operator_base<TInterval>
{
public:
    INIT_OPERATOR(projection_op_)

    template<class T>
    inline void operator()(samurai::Dim<2>,T& new_field, const T& field) const
    {
        new_field(level, i, j) = .25 * (field(level + 1, 2 * i, 2 * j) +
                                        field(level + 1, 2 * i, 2 * j + 1) +
                                        field(level + 1, 2 * i + 1, 2 * j) +
                                        field(level + 1, 2 * i + 1, 2 * j + 1));
    }
};

template<class T>
inline auto projection(T&& new_field, T&& field)
{
    return samurai::make_field_operator_function<projection_op_>(std::forward<T>(new_field), std::forward<T>(field));
}


template <class Mesh>
auto init_level_set(Mesh & mesh)
{
    using mesh_id_t = typename Mesh::mesh_id_t;

    auto phi = samurai::make_field<double, 1>("phi", mesh);
    phi.fill(0);

    samurai::for_each_cell(mesh[mesh_id_t::cells], [&](auto &cell)
    {
        auto center = cell.center();


        double x = center[0];
        double y = center[1];
        double radius = .15;
        double x_center = 0.5, y_center = 0.75;

        phi[cell] = std::sqrt(std::pow(x - x_center, 2.) +
                              std::pow(y - y_center, 2.)) - radius;
        // phi[cell] = std::sqrt(std::pow(x - x_center, 2.) +
                            //   std::pow(y - y_center, 2.)) - radius > 0. ? 0. : 1.;
    });

    return phi;
}

template <class Mesh>
auto init_velocity(Mesh &mesh)
{

    using mesh_id_t = typename Mesh::mesh_id_t;

    auto u = samurai::make_field<double, 2>("u", mesh);
    u.fill(0);


    samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts], [&](auto &cell)
    {
        auto center = cell.center();

        double x = center[0];
        double y = center[1];

        u[cell][0] =    -std::pow(std::sin(M_PI*x), 2.) * std::sin(2.*M_PI*y);
        u[cell][1] =     std::pow(std::sin(M_PI*y), 2.) * std::sin(2.*M_PI*x);
        // u[cell][0] =   .3;
        // u[cell][1] =   .3;
    });

    return u;
}

template<class Field>
void make_graduation(Field & tag)
{
    auto mesh = tag.mesh();
    for (std::size_t level = mesh.max_level(); level >= 1; --level)
    {

        auto ghost_subset = samurai::intersection(mesh[SimpleID::cells][level],
                                                mesh[SimpleID::reference][level-1])
                        .on(level - 1);
        ghost_subset([&](const auto& i, const auto& index)
        {
            auto j = index[0];
            tag(level - 1, i, j) |= static_cast<int>(samurai::CellFlag::keep);
        });

        auto subset_2 = intersection(mesh[SimpleID::cells][level],
                                        mesh[SimpleID::cells][level]);

        subset_2([&](const auto& interval, const auto& index)
        {
            auto i = interval;
            auto j = index[0];
            xt::xtensor<bool, 1> mask = (tag(level, i, j) & static_cast<int>(samurai::CellFlag::refine));

            for(int jj = -1; jj < 2; ++jj)
            {
                for(int ii = -1; ii < 2; ++ii)
                {
                    xt::masked_view(tag(level, i + ii, j + jj), mask) |= static_cast<int>(samurai::CellFlag::keep);
                }
            }
        });

        auto keep_subset = samurai::intersection(mesh[SimpleID::cells][level],
                                                mesh[SimpleID::cells][level])
                        .on(level - 1);
        keep_subset([&](const auto& interval, const auto& index)
        {
            auto i = interval;
            auto j = index[0];

            xt::xtensor<bool, 1> mask = (tag(level,     2 * i,     2 * j) & static_cast<int>(samurai::CellFlag::keep))
                                        | (tag(level, 2 * i + 1,     2 * j) & static_cast<int>(samurai::CellFlag::keep))
                                        | (tag(level,     2 * i, 2 * j + 1) & static_cast<int>(samurai::CellFlag::keep))
                                        | (tag(level, 2 * i + 1, 2 * j + 1) & static_cast<int>(samurai::CellFlag::keep));

            xt::masked_view(tag(level,     2 * i,     2 * j), mask) |= static_cast<int>(samurai::CellFlag::keep);
            xt::masked_view(tag(level, 2 * i + 1,     2 * j), mask) |= static_cast<int>(samurai::CellFlag::keep);
            xt::masked_view(tag(level,     2 * i, 2 * j + 1), mask) |= static_cast<int>(samurai::CellFlag::keep);
            xt::masked_view(tag(level, 2 * i + 1, 2 * j + 1), mask) |= static_cast<int>(samurai::CellFlag::keep);

        });

        xt::xtensor_fixed<int, xt::xshape<4, dim>> stencil{{1, 1}, {-1, -1}, {-1, 1}, {1, -1}};
        // xt::xtensor_fixed<int, xt::xshape<4, dim>> stencil{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

        for(std::size_t i = 0; i < stencil.shape()[0]; ++i)
        {
            auto s = xt::view(stencil, i);
            auto subset = samurai::intersection(samurai::translate(mesh[SimpleID::cells][level], s),
                                            mesh[SimpleID::cells][level - 1])
                        .on(level);

            subset([&](const auto& interval, const auto& index)
            {
                auto j_f = index[0];
                auto i_f = interval.even_elements();

                if (i_f.is_valid())
                {
                    auto mask = tag(level, i_f  - s[0], j_f - s[1]) & static_cast<int>(samurai::CellFlag::refine);
                    auto i_c = i_f >> 1;
                    auto j_c = j_f >> 1;
                    xt::masked_view(tag(level - 1, i_c, j_c), mask) |= static_cast<int>(samurai::CellFlag::refine);

                    mask = tag(level, i_f  - s[0], j_f - s[1]) & static_cast<int>(samurai::CellFlag::keep);
                    xt::masked_view(tag(level - 1, i_c, j_c), mask) |= static_cast<int>(samurai::CellFlag::keep);
                }

                i_f = interval.odd_elements();
                if (i_f.is_valid())
                {
                    auto mask = tag(level, i_f  - s[0], j_f - s[1]) & static_cast<int>(samurai::CellFlag::refine);
                    auto i_c = i_f >> 1;
                    auto j_c = j_f >> 1;
                    xt::masked_view(tag(level - 1, i_c, j_c), mask) |= static_cast<int>(samurai::CellFlag::refine);

                    mask = tag(level, i_f  - s[0], j_f - s[1]) & static_cast<int>(samurai::CellFlag::keep);
                    xt::masked_view(tag(level - 1, i_c, j_c), mask) |= static_cast<int>(samurai::CellFlag::keep);
                }
            });
        }
    }
}

template<class Field, class Tag>
void AMR_criteria(const Field& f, Tag& tag)
{
    auto mesh = f.mesh();
    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    samurai::for_each_cell(mesh[SimpleID::cells], [&](auto cell)
    {

        double dx = 1./(1 << (max_level));

        if (std::abs(f[cell]) < 1.2 * 5 * std::sqrt(2.) * dx)
        {
            if (cell.level == max_level)
            {
                tag[cell] = static_cast<int>(samurai::CellFlag::keep);
            }
            else
            {
                tag[cell] = static_cast<int>(samurai::CellFlag::refine);
            }
        }
        else
        {
            if (cell.level == min_level)
            {
                tag[cell] = static_cast<int>(samurai::CellFlag::keep);
            }
            else
            {
                tag[cell] = static_cast<int>(samurai::CellFlag::coarsen);
            }
        }
    });
}

template<class Field, class Field_u, class Tag>
bool update_mesh(Field& f, Field_u& u, const Tag& tag)
{
    using mesh_t = typename Field::mesh_t;
    using interval_t = typename mesh_t::interval_t;
    using coord_index_t = typename interval_t::coord_index_t;
    using cl_type = typename mesh_t::cl_type;

    auto mesh = f.mesh();

    cl_type cell_list;

    samurai::for_each_interval(mesh[SimpleID::cells], [&](std::size_t level, const auto& interval, const auto& index_yz)
    {
        for (int i = interval.start; i < interval.end; ++i)
        {
            if (tag[i + interval.index] & static_cast<int>(samurai::CellFlag::refine))
            {
                samurai::static_nested_loop<dim - 1, 0, 2>([&](auto stencil)
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
                cell_list[level-1][index_yz>>1].add_point(i>>1);
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

    Field_u new_u{u.name(), new_mesh};
    new_u.fill(0.);

    for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
    {
        auto subset = samurai::intersection(mesh[SimpleID::cells][level],
                                         new_mesh[SimpleID::cells][level]);

        subset.apply_op(samurai::copy(new_f, f));
        subset.apply_op(samurai::copy(new_u, u));
    }

    samurai::for_each_interval(mesh[SimpleID::cells], [&](std::size_t level, const auto& interval, const auto& index_yz)
    {
        for (coord_index_t i = interval.start; i < interval.end; ++i)
        {
            if (tag[i + interval.index] & static_cast<int>(samurai::CellFlag::refine))
            {
                samurai::compute_prediction(level, interval_t{i, i + 1}, index_yz, f, new_f);
                samurai::compute_prediction(level, interval_t{i, i + 1}, index_yz, u, new_u);
            }
        }
    });

    for (std::size_t level = mesh.min_level() + 1; level <= mesh.max_level(); ++level)
    {
        auto subset = samurai::intersection(mesh[SimpleID::cells][level],
                                         new_mesh[SimpleID::cells][level - 1])
                     .on(level - 1);
        subset.apply_op(projection(new_f, f));
        subset.apply_op(projection(new_u, u));
    }

    f.mesh_ptr()->swap(new_mesh);
    std::swap(f.array(), new_f.array());
    std::swap(u.array(), new_u.array());

    return false;
}

template<class Field>
inline void amr_projection(Field &field)
{
    auto mesh = field.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;

    std::size_t min_level = mesh.min_level(), max_level = mesh.max_level();

    for (std::size_t level = max_level; level >= min_level; --level)
    {
        auto expr = samurai::intersection(mesh[mesh_id_t::cells][level],
                                       mesh[mesh_id_t::cells_and_ghosts][level - 1])
                   .on(level - 1);

        expr.apply_op(projection(field));
    }
}

template<class Field, class Func>
inline void amr_prediction(Field &field, Func&& update_bc_for_level)
{
    auto mesh = field.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;

    std::size_t min_level = mesh[mesh_id_t::cells].min_level(), max_level = mesh[mesh_id_t::cells].max_level();

    for (std::size_t level = min_level + 1; level <= max_level; ++level)
    {
        auto expr = samurai::intersection(mesh.domain(),
                                         samurai::difference(mesh[mesh_id_t::cells_and_ghosts][level],
                                                             mesh.get_union()[level]))
                   .on(level);

        expr.apply_op(prediction(field));
        update_bc_for_level(field, level);
    }
}

template <class Field, class Field_u, class Func>
void update_ghosts(Field& phi, Field_u& u, Func&& update_bc_for_level)
{
    auto mesh = phi.mesh();
    std::size_t min_level = mesh.min_level(), max_level = mesh.max_level();

    amr_projection(phi);
    amr_projection(u);

    for (std::size_t level = min_level; level <= max_level; ++level)
    {
        update_bc_for_level(phi, level);
        update_bc_for_level(u, level);
    }

    amr_prediction(phi, std::forward<Func>(update_bc_for_level));
    amr_prediction(u, std::forward<Func>(update_bc_for_level));
}

template <class Field, class Field_u>
void flux_correction(Field& phi_np1, const Field& phi_n, const Field_u& u, double dt)
{
    using mesh_t = typename Field::mesh_t;
    using mesh_id_t = typename mesh_t::mesh_id_t;
    using interval_t = typename mesh_t::interval_t;

    auto mesh = phi_np1.mesh();
    std::size_t min_level = mesh[mesh_id_t::cells].min_level();
    std::size_t max_level = mesh[mesh_id_t::cells].max_level();
    for (std::size_t level = min_level; level < max_level; ++level)
    {
        xt::xtensor_fixed<int, xt::xshape<dim>> stencil;

        stencil = {{-1, 0}};

        auto subset_right = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level+1], stencil),
                                               mesh[mesh_id_t::cells][level])
                           .on(level);

        subset_right([&](const auto& i, const auto& index)
        {
            auto j = index[0];
            double dx = 1./(1<<level);

            phi_np1(level, i, j) = phi_np1(level, i, j) + dt/dx * (samurai::upwind_variable_op<interval_t>(level, i, j).right_flux(u, phi_n, dt)
                                                            - .5*samurai::upwind_variable_op<interval_t>(level+1, 2*i+1, 2*j).right_flux(u, phi_n, dt)
                                                            - .5*samurai::upwind_variable_op<interval_t>(level+1, 2*i+1, 2*j+1).right_flux(u, phi_n, dt));
        });

        stencil = {{1, 0}};

        auto subset_left = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level+1], stencil),
                                        mesh[mesh_id_t::cells][level])
                            .on(level);

        subset_left([&](const auto& i, const auto& index)
        {
            auto j = index[0];
            double dx = 1./(1<<level);

            phi_np1(level, i, j) = phi_np1(level, i, j) - dt/dx * (samurai::upwind_variable_op<interval_t>(level, i, j).left_flux(u, phi_n, dt)
                                                            - .5 * samurai::upwind_variable_op<interval_t>(level+1, 2*i, 2*j).left_flux(u, phi_n, dt)
                                                            - .5 * samurai::upwind_variable_op<interval_t>(level+1, 2*i, 2*j+1).left_flux(u, phi_n, dt));
        });

        stencil = {{0, -1}};

        auto subset_up = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level+1], stencil),
                                    mesh[mesh_id_t::cells][level])
                            .on(level);

        subset_up([&](const auto& i, const auto& index)
        {
            auto j = index[0];
            double dx = 1./(1<<level);

            phi_np1(level, i, j) = phi_np1(level, i, j) + dt/dx * (samurai::upwind_variable_op<interval_t>(level, i, j).up_flux(u, phi_n, dt)
                                                            - .5 * samurai::upwind_variable_op<interval_t>(level+1, 2*i, 2*j+1).up_flux(u, phi_n, dt)
                                                            - .5 * samurai::upwind_variable_op<interval_t>(level+1, 2*i+1, 2*j+1).up_flux(u, phi_n, dt));
        });

        stencil = {{0, 1}};

        auto subset_down = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level+1], stencil),
                                    mesh[mesh_id_t::cells][level])
                            .on(level);

        subset_down([&](const auto& i, const auto& index)
        {
            auto j = index[0];
            double dx = 1./(1<<level);

            phi_np1(level, i, j) = phi_np1(level, i, j) - dt/dx * (samurai::upwind_variable_op<interval_t>(level, i, j).down_flux(u, phi_n, dt)
                                                            - .5 * samurai::upwind_variable_op<interval_t>(level+1, 2*i, 2*j).down_flux(u, phi_n, dt)
                                                            - .5 * samurai::upwind_variable_op<interval_t>(level+1, 2*i+1, 2*j).down_flux(u, phi_n, dt));
        });
    }
}

int main(int argc, char *argv[])
{
    using Config = AMRConfig<dim>;
    using interval_t = typename Config::interval_t;

    std::size_t min_level = 4;
    // std::size_t min_level = 8;

    std::size_t max_level = 8;
    samurai::Box<double, dim> box({0, 0}, {1, 1});
    AMRMesh<Config> mesh{box, max_level, min_level, max_level};

    // double Tf = 1.; // Final time
    double Tf = 3.14; // Final time


    // double dt = 1./std::sqrt(2.) * 1./(1<<max_level);
    double dt = 5./8/(1<<max_level);

    // We initialize the level set function
    // We initialize the velocity field

    auto phi = init_level_set(mesh);
    auto u   = init_velocity(mesh);

    auto update_bc_for_level = [](auto& field, std::size_t level)
    {
        update_bc_D2Q4_3_Euler_constant_extension(field, level);
    };

    std::size_t Ntot = Tf/dt;
    for (std::size_t nt=0; nt <= Ntot; ++nt)

    {
        std::cout<<std::endl<<"Iteration "<< nt <<std::endl;

        std::size_t ite = 0;
        while(true)
        {
            std::cout << "Mesh adaptation iteration " << ite++ << std::endl;
            auto tag = samurai::make_field<int, 1>("tag", mesh);
            AMR_criteria(phi, tag);
            make_graduation(tag);
            update_ghosts(phi, u, update_bc_for_level);

            if(update_mesh(phi, u, tag))
            {
                break;
            }
        }

        auto level_field = samurai::make_field<std::size_t, 1>("level", mesh);
        samurai::for_each_cell(mesh[SimpleID::cells], [&](auto &cell)
        {
            level_field[cell] = cell.level;
        });
        samurai::save(fmt::format("LevelSet_Advection_ite_{}", nt), mesh, phi, u, level_field);

        update_ghosts(phi, u, update_bc_for_level);
        auto phinp1 = samurai::make_field<double, 1>("phi", mesh);

        phinp1 = phi - dt * samurai::upwind_variable(u, phi, dt);
        flux_correction(phinp1, phi, u, dt);

        std::swap(phi.array(), phinp1.array());

        // Reinitialization of the level set
        std::size_t fict_iteration = 2; // Number of fictitious iterations
        double dt_fict = 0.01 * dt; // Fictitious Time step

        auto phi_0 = phi;
        for (std::size_t k = 0; k < fict_iteration; ++k)
        {
            // //Forward Euler - OK
            // update_ghosts(phi, u, update_bc_for_level);
            // phinp1 = phi - dt_fict * H_wrap(phi, phi_0, max_level);
            // std::swap(phi.array(), phinp1.array());


            // TVD-RK2
            update_ghosts(phi, u, update_bc_for_level);
            auto phihat = samurai::make_field<double, 1>("phi", mesh);
            phihat = phi - dt_fict * H_wrap(phi, phi_0, max_level);
            update_ghosts(phihat, u, update_bc_for_level); // Crucial !!!
            phinp1 = .5 * phi_0 + .5 * (phihat - dt_fict * H_wrap(phihat, phi_0, max_level));
            std::swap(phi.array(), phinp1.array());
        }
    }

    return 0;
}