// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// MR-adapted meshes refined around the surfaces of balls, and the copy of
// their sets into bench::SiteInputs. Needs the mesh, the fields, the boundary
// conditions and the MR adaptation: only generate_set_algebra_inputs and
// bench_set_library_ops include it.

#pragma once

#include <cmath>
#include <cstddef>
#include <map>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

#include <benchmark/benchmark.h>

#include <xtensor/generators/xbuilder.hpp>

#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/stencil.hpp>

#include "set_algebra_fixtures.hpp"
#include "site_inputs.hpp"

namespace bench
{
    template <std::size_t dim>
    auto mr_config(std::size_t max_level)
    {
        return samurai::mesh_config<dim>().min_level(2).max_level(max_level).max_stencil_size(2).disable_minimal_ghost_width();
    }

    template <std::size_t dim>
    using mr_mesh_t = decltype(samurai::mra::make_mesh(std::declval<samurai::Box<double, dim>>(), mr_config<dim>(4)));

    template <std::size_t dim>
    using mr_field_t = decltype(samurai::make_scalar_field<double>("u", std::declval<mr_mesh_t<dim>&>()));

    // The mesh adapted to this indicator is refined to the finest level along
    // every ball surface. `shift` moves the balls along the diagonal.
    template <std::size_t dim>
    auto balls_indicator(std::size_t nballs, double shift)
    {
        return [centers = ball_centers<dim>(nballs, 0.25, 0.75),
                radius  = 0.2 / std::pow(static_cast<double>(nballs), 1. / static_cast<double>(dim)),
                shift](const auto& x)
        {
            for (const auto& c : centers)
            {
                double r2 = 0;
                for (std::size_t d = 0; d < dim; ++d)
                {
                    const double dx = x[d] - c[d] - shift;
                    r2 += dx * dx;
                }
                if (r2 <= radius * radius)
                {
                    return 1.;
                }
            }
            return 0.;
        };
    }

    constexpr double mr_epsilon = 1e-3;

    // How far the fronts move in one time step.
    template <class Mesh>
    double step_shift(const Mesh& mesh)
    {
        return 4. * mesh.min_cell_length();
    }

    template <std::size_t dim>
    mr_mesh_t<dim> make_unit_mesh(std::size_t max_level)
    {
        return samurai::mra::make_mesh(samurai::Box<double, dim>(xt::zeros<double>({dim}), xt::ones<double>({dim})),
                                       mr_config<dim>(max_level));
    }

    template <std::size_t dim>
    struct AdaptedCase
    {
        mr_mesh_t<dim> mesh;
        mr_field_t<dim> u;

        AdaptedCase(std::size_t max_level, std::size_t nballs, bool moved)
            : mesh(make_unit_mesh<dim>(max_level))
            , u(samurai::make_scalar_field<double>("u", mesh, balls_indicator<dim>(nballs, moved ? step_shift(mesh) : 0.)))
        {
            samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);
            samurai::make_MRAdapt(u)(samurai::mra_config().epsilon(mr_epsilon));
        }

        AdaptedCase(const AdaptedCase&)            = delete;
        AdaptedCase& operator=(const AdaptedCase&) = delete;
    };

    // Built once per binary and shared by its benchmarks. `moved` gives the
    // mesh of the next time step.
    template <std::size_t dim>
    AdaptedCase<dim>& adapted_case(std::size_t max_level, std::size_t nballs, bool moved = false)
    {
        static std::map<std::tuple<std::size_t, std::size_t, bool>, std::unique_ptr<AdaptedCase<dim>>> cache;
        auto& entry = cache[{max_level, nballs, moved}];
        if (!entry)
        {
            entry = std::make_unique<AdaptedCase<dim>>(max_level, nballs, moved);
        }
        return *entry;
    }

    template <class Mesh>
    void report_mesh(benchmark::State& state, const Mesh& mesh)
    {
        report_mesh(state, mesh[Mesh::mesh_id_t::cells]);
    }

    template <std::size_t dim>
    SiteInputs<dim> make_site_inputs(const mr_mesh_t<dim>& mesh)
    {
        using mesh_t    = mr_mesh_t<dim>;
        using mesh_id_t = typename mesh_t::mesh_id_t;
        static_assert(std::is_same_v<typename mesh_t::ca_type, typename SiteInputs<dim>::ca_t>);
        static_assert(mesh_t::config_t::prediction_stencil_radius == prediction_stencil_radius);

        SiteInputs<dim> in;
        in.min_level        = mesh.min_level();
        in.max_level        = mesh.max_level();
        in.cells            = mesh[mesh_id_t::cells];
        in.cells_and_ghosts = mesh[mesh_id_t::cells_and_ghosts];
        in.proj_cells       = mesh[mesh_id_t::proj_cells];
        in.reference        = mesh[mesh_id_t::reference];
        in.union_           = mesh.get_union();
        in.domain_pyramid   = mesh.domain_pyramid();
        for (std::size_t level = 0; level <= mesh.max_level(); ++level)
        {
            in.subdomain_pyramid[level] = mesh.subdomain(level);
        }
        samurai::for_each_diagonal_direction<dim>(
            [&](const auto& direction)
            {
                in.corners.push_back(mesh.corner(direction));
            });
        return in;
    }
} // namespace bench
