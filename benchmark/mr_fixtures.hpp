// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// MR-adapted meshes refined around the surfaces of balls, and the copy of
// their sets into bench::SiteInputs. Needs
// the mesh, the fields, the boundary conditions and the MR adaptation: only
// generate_set_algebra_inputs and bench_set_library_ops include it.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
    // disable_minimal_ghost_width keeps the ghost width at stencil_radius: by
    // default the mesh raises the max stencil radius to 2 (see SET_ALGEBRA.md).
    template <std::size_t dim>
    auto mr_config(std::size_t max_level, int stencil_radius)
    {
        return samurai::mesh_config<dim>().min_level(2).max_level(max_level).max_stencil_radius(stencil_radius).disable_minimal_ghost_width();
    }

    template <std::size_t dim>
    using mr_mesh_t = decltype(samurai::mra::make_mesh(std::declval<samurai::Box<double, dim>>(), mr_config<dim>(4, 1)));

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

    // The mesh of c before adaptation, on the unit box.
    template <std::size_t dim>
    mr_mesh_t<dim> make_unit_mesh(const MeshCase& c)
    {
        const samurai::Box<double, dim> unit_box(xt::zeros<double>({dim}), xt::ones<double>({dim}));
        return samurai::mra::make_mesh(unit_box, mr_config<dim>(c.max_level, c.stencil_radius));
    }

    template <std::size_t dim>
    struct AdaptedCase
    {
        mr_mesh_t<dim> mesh;
        mr_field_t<dim> u;

        AdaptedCase(const MeshCase& c, bool moved)
            : mesh(make_unit_mesh<dim>(c))
            , u(samurai::make_scalar_field<double>("u", mesh, balls_indicator<dim>(c.nballs, moved ? step_shift(mesh) : 0.)))
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
    AdaptedCase<dim>& adapted_case(const MeshCase& c, bool moved = false)
    {
        static std::map<std::tuple<mesh_key_t, bool>, std::unique_ptr<AdaptedCase<dim>>> cache;
        auto& entry = cache[{c.key(), moved}];
        if (!entry)
        {
            entry = std::make_unique<AdaptedCase<dim>>(c, moved);
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
        in.min_level          = mesh.min_level();
        in.max_level          = mesh.max_level();
        in.graduation_width   = mesh.graduation_width();
        in.max_stencil_radius = mesh.max_stencil_radius();
        in.cells              = mesh[mesh_id_t::cells];
        in.cells_and_ghosts   = mesh[mesh_id_t::cells_and_ghosts];
        in.proj_cells         = mesh[mesh_id_t::proj_cells];
        in.reference          = mesh[mesh_id_t::reference];
        in.union_             = mesh.get_union();
        in.domain_pyramid     = mesh.domain_pyramid();
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

    // What Adapt::harten (mr/adapt.hpp) passes to make_graduation at the first
    // iteration of the adaptation of `adapted` to the fronts shifted by
    // `shift`: update_cell_array_from_tag(cells, tag), before graduation, in
    // in.ungraded_cells. A copy of harten up to that call, for the
    // configuration of AdaptedCase: sequential, no relative detail, boundary
    // not kept refined.
    template <std::size_t dim>
    void add_next_step(SiteInputs<dim>& in, const mr_mesh_t<dim>& adapted, std::size_t nballs, double shift)
    {
        using mesh_id_t = typename mr_mesh_t<dim>::mesh_id_t;

        auto mesh = adapted;
        auto u    = samurai::make_scalar_field<double>("u", mesh, balls_indicator<dim>(nballs, shift));
        samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);
        auto detail = samurai::make_scalar_field<double>("detail", mesh, 0.);
        auto tag    = samurai::make_scalar_field<std::uint8_t>("tag", mesh, std::uint8_t{0});

        const std::size_t min_level = mesh.min_level();
        const std::size_t max_level = mesh.max_level();
        const auto cfg              = samurai::mra_config().epsilon(mr_epsilon);

        samurai::for_each_cell(mesh[mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   tag[cell] = static_cast<std::uint8_t>(samurai::CellFlag::keep);
                               });
        samurai::update_ghost_mr(u);

        for (std::size_t level = (min_level > 0) ? min_level - 1 : 0; level < max_level; ++level)
        {
            samurai::ghosts_below_cells(mesh, level).apply_op(samurai::compute_detail(detail, u));
        }

        for (std::size_t level = std::max<std::size_t>(min_level, 1); level <= max_level; ++level)
        {
            const double eps_l = cfg.epsilon() / static_cast<double>(1 << (dim * (max_level - level)));
            auto subset_1      = samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::all_cells][level - 1]).on(level - 1);
            subset_1.apply_op(samurai::mr_criteria(detail, tag, eps_l, cfg.regularity() + dim));
            samurai::update_tag_subdomains(level, tag, true);
        }

        for (std::size_t level = max_level; level > 0; --level)
        {
            auto keep_subset = samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::all_cells][level - 1]).on(level - 1);
            samurai::update_tag_periodic(level, tag);
            samurai::update_tag_subdomains(level, tag);
            keep_subset.apply_op(samurai::maximum(tag));
        }

        in.ungraded_cells = samurai::update_cell_array_from_tag(mesh[mesh_id_t::cells], tag);
    }
} // namespace bench
