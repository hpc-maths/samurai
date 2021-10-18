// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/algorithm.hpp>
#include <samurai/stencil_field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/subset/subset_op.hpp>

#include <xtensor/xfixed.hpp>

template <class Mesh>
auto init(Mesh& mesh)
{
    auto u = samurai::make_field<double, 1>("u", mesh);

    samurai::for_each_cell(mesh, [&](auto &cell) {
        auto center = cell.center();
        double radius = .2;
        double x_center = 0;
        if (std::abs(center[0] - x_center) <= radius)
        {
            u[cell] = 1;
        }
        else
        {
            u[cell] = 0;
        }
    });

    return u;
}

template<class Field>
void flux_correction(double dt, double a, const Field& u, Field& unp1)
{
    using mesh_t = typename Field::mesh_t;
    using mesh_id_t = typename mesh_t::mesh_id_t;
    using interval_t = typename mesh_t::interval_t;
    constexpr std::size_t dim = Field::dim;

    auto mesh = u.mesh();

    for (std::size_t level = mesh.min_level(); level < mesh.max_level(); ++level)
    {
        xt::xtensor_fixed<int, xt::xshape<dim>> stencil;

        stencil = {{-1}};

        auto subset_right = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level+1], stencil),
                                                  mesh[mesh_id_t::cells][level])
                           .on(level);

        subset_right([&](const auto& i, auto)
        {
            double dx = 1./(1<<level);

            unp1(level, i) = unp1(level, i) - dt/dx * (-samurai::upwind_op<interval_t>(level, i).right_flux(a, u)
                                                       +samurai::upwind_op<interval_t>(level+1, 2*i+1).right_flux(a, u));
        });

        stencil = {{1}};

        auto subset_left = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level+1], stencil),
                                                 mesh[mesh_id_t::cells][level])
                          .on(level);

        subset_left([&](const auto& i, auto)
        {
            double dx = 1./(1<<level);

            unp1(level, i) = unp1(level, i) - dt/dx * (samurai::upwind_op<interval_t>(level, i).left_flux(a, u)
                                                      -samurai::upwind_op<interval_t>(level+1, 2*i).left_flux(a, u));
        });
    }
}

template <class Field>
void dirichlet(std::size_t level, Field& u)
{
    using mesh_t = typename Field::mesh_t;
    using mesh_id_t = typename mesh_t::mesh_id_t;

    auto mesh = u.mesh();

    auto boundary = samurai::difference(mesh[mesh_id_t::reference][level],
                                        mesh.domain())
                   .on(level);
    boundary([&](const auto& i, auto)
    {
        u(level, i) = 0.;
    });
}

template <class Field>
void save(std::size_t nt, const Field& u)
{
    auto mesh = u.mesh();
    auto level_ = samurai::make_field<std::size_t, 1>("level", mesh);

    samurai::for_each_cell(mesh, [&](auto &cell)
    {
        level_[cell] = cell.level;
    });

    samurai::save(fmt::format("FV_advection_1d_ite_{}", nt), mesh, u, level_);
}

int main(int argc, char *argv[])
{
    constexpr size_t dim = 1;
    using Config = samurai::MRConfig<dim>;
    using interval_t = typename Config::interval_t;

    const double regularity = 1.; // Regularity guess for multiresolution
    const double epsilon_MR = 2.e-4; // Threshold used by multiresolution

    std::size_t min_level = 4, max_level = 10;
    samurai::Box<double, dim> box({-2}, {2});
    samurai::MRMesh<Config> mesh{box, min_level, max_level};

    double a = 1.;
    double dt = .95/(1<<max_level);

    auto u = init(mesh);
    auto unp1 = samurai::make_field<double, 1>("unp1", mesh);

    auto update_bc = [](auto& u, std::size_t level)
    {
        dirichlet(level, u);
    };

    auto MRadaptation = samurai::make_MRAdapt(u, update_bc);

    for (std::size_t nt=0; nt<500; ++nt)
    {
        std::cout << "iteration " << nt << "\n";

        if (max_level > min_level)
        {
            MRadaptation(epsilon_MR, regularity);
        }

        samurai::update_ghost_mr(u, update_bc);
        unp1.resize();
        unp1 = u - dt * samurai::upwind(a, u);
        flux_correction(dt, a, u, unp1);

        std::swap(u.array(), unp1.array());

        save(nt, u);
    }
    return 0;
}