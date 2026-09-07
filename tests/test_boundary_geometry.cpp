// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// The boundary geometry (corners, inner and outer boundary layers of the domain,
// boundary condition regions) is precomputed at every level on the mesh and on
// the Bc. The consumers of the outer ghost update read those level cell arrays
// instead of coarsening the finest one with self(lca).on(level), and write their
// set expressions as intersections with the (thin) precomputed layers instead of
// differences with the domain. These tests pin the equivalence, cell by cell, of
// the precomputed sets and of the rewritten expressions with the former
// formulations, on an adapted mesh whose refinement reaches the domain boundary.

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include <samurai/bc.hpp>
#include <samurai/boundary.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/subset/node.hpp>

namespace samurai
{
    namespace
    {
        // Every cell of a set expression as (x, y[, z]), sorted. Comparing two sets
        // through this list is insensitive to the way the intervals are fragmented.
        template <class Set>
        auto cells_of(const Set& set)
        {
            constexpr std::size_t dim = std::decay_t<Set>::dim;
            std::vector<std::array<int, dim>> cells;
            set(
                [&](const auto& i, const auto& index)
                {
                    for (auto x = i.start; x < i.end; ++x)
                    {
                        std::array<int, dim> cell;
                        cell[0] = static_cast<int>(x);
                        for (std::size_t d = 1; d < dim; ++d)
                        {
                            cell[d] = static_cast<int>(index[d - 1]);
                        }
                        cells.push_back(cell);
                    }
                });
            std::sort(cells.begin(), cells.end());
            return cells;
        }

        template <std::size_t dim>
        auto make_adapted_mesh_and_field()
        {
            auto config = mesh_config<dim>().min_level(2).max_level(6).max_stencil_size(2).disable_minimal_ghost_width();
            auto mesh   = mra::make_mesh(Box<double, dim>{xt::zeros<double>({dim}), xt::ones<double>({dim})}, config);

            auto u = make_scalar_field<double>("u", mesh);
            // Two fronts spanning the whole domain: the refinement reaches the boundary
            // and the corners, so every boundary set is non trivial at several levels.
            for_each_cell(mesh,
                          [&](auto& cell)
                          {
                              auto c  = cell.center();
                              u[cell] = std::tanh(1000 * std::abs(c[0] - 1. / 3)) + std::tanh(1000 * std::abs(c[dim - 1] - 2. / 3)) - 2;
                          });
            make_bc<Dirichlet<1>>(u, 0.);
            make_MRAdapt(u)(mra_config().epsilon(1e-3));
            return std::make_pair(std::move(mesh), std::move(u));
        }

        template <std::size_t dim>
        void check_boundary_geometry()
        {
            auto [mesh, u]  = make_adapted_mesh_and_field<dim>();
            using mesh_id_t = typename decltype(mesh)::mesh_id_t;

            const std::size_t max_level = mesh.max_level();
            const int n_layers          = std::max(mesh.ghost_width(), mesh.max_stencil_radius());
            std::size_t checked         = 0;

            for (std::size_t level = 0; level <= max_level; ++level)
            {
                const auto& domain_l = mesh.domain(level);
                auto domain_on_level = self(mesh.domain()).on(level);
                EXPECT_EQ(cells_of(self(domain_l)), cells_of(domain_on_level)) << "domain pyramid, level " << level;

                for_each_cartesian_direction<dim>(
                    [&](const auto& direction)
                    {
                        // inner and outer boundary layers of the domain
                        EXPECT_EQ(cells_of(self(mesh.boundary_inner_layer(level, direction))),
                                  cells_of(difference(domain_on_level, translate(domain_on_level, -direction))))
                            << "inner layer, level " << level;
                        for (int k = 1; k <= n_layers; ++k)
                        {
                            EXPECT_EQ(cells_of(self(mesh.boundary_outer_layer(level, direction, k))),
                                      cells_of(difference(translate(domain_on_level, k * direction),
                                                          translate(domain_on_level, (k - 1) * direction))))
                                << "outer layer " << k << ", level " << level;
                        }

                        // domain_boundary: intersection with the inner layer == former difference
                        const auto& cells = mesh[mesh_id_t::cells][level];
                        EXPECT_EQ(cells_of(domain_boundary(mesh, level, direction)),
                                  cells_of(difference(cells, translate(domain_on_level, -direction))))
                            << "domain_boundary, level " << level;

                        // project_bc: the layer-th outer layer of the union, former and new form
                        const auto& inner = mesh.get_union()[level];
                        for (int k = 1; k <= n_layers; ++k)
                        {
                            auto former = intersection(
                                              difference(translate(inner, k * direction), translate(domain_on_level, (k - 1) * direction)),
                                              mesh[mesh_id_t::reference][level])
                                              .on(level);
                            auto current = intersection(translate(inner, k * direction),
                                                        mesh.boundary_outer_layer(level, direction, k),
                                                        mesh[mesh_id_t::reference][level]);
                            EXPECT_EQ(cells_of(current), cells_of(former)) << "project_bc layer " << k << ", level " << level;
                        }

                        // predict_bc: the union of the per-layer sets == the former materialised
                        // set, and the layers are disjoint (no cell is predicted twice)
                        if (level < max_level)
                        {
                            const int n_bc_ghosts = static_cast<int>(u.get_bc().front()->stencil_size()) / 2;
                            // named: a set expression only references the level cell arrays it is built from
                            auto former_lca   = domain_boundary_outer_layer(mesh, level, direction, n_bc_ghosts);
                            auto former       = intersection(former_lca, mesh[mesh_id_t::reference][level + 1]).on(level + 1);
                            auto former_cells = cells_of(former);

                            std::vector<std::array<int, dim>> current_cells;
                            auto inner_boundary = domain_boundary(mesh, level, direction);
                            for (int k = 1; k <= n_bc_ghosts; ++k)
                            {
                                auto layer_k = intersection(difference(translate(inner_boundary, k * direction), domain_l),
                                                            mesh[mesh_id_t::reference][level + 1])
                                                   .on(level + 1);
                                auto layer_cells = cells_of(layer_k);
                                current_cells.insert(current_cells.end(), layer_cells.begin(), layer_cells.end());
                            }
                            std::sort(current_cells.begin(), current_cells.end());
                            EXPECT_EQ(std::adjacent_find(current_cells.begin(), current_cells.end()), current_cells.end())
                                << "predict_bc: a ghost belongs to two layers, level " << level;
                            EXPECT_EQ(current_cells, former_cells) << "predict_bc, level " << level;
                        }

                        checked += 1;
                    });

                // corners: pyramid == coarsening of the finest corner
                for_each_diagonal_direction<dim>(
                    [&](const auto& direction)
                    {
                        EXPECT_EQ(cells_of(self(mesh.corner(direction, level))), cells_of(self(mesh.corner(direction)).on(level)))
                            << "corner, level " << level;
                    });

                // boundary condition regions: pyramid == former coarsened intersection
                const auto& bc     = *u.get_bc().front();
                const auto& region = bc.get_region();
                for (std::size_t d = 0; d < region.second.size(); ++d)
                {
                    const auto& cells = mesh[mesh_id_t::cells][level];
                    EXPECT_EQ(cells_of(intersection(cells, bc.region_at(d, level))), cells_of(intersection(cells, region.second[d]).on(level)))
                        << "bc region " << d << ", level " << level;
                }
            }
            EXPECT_EQ(checked, 2 * dim * (max_level + 1));
        }
    }

    TEST(boundary_geometry, equivalence_2d)
    {
        check_boundary_geometry<2>();
    }

    TEST(boundary_geometry, equivalence_3d)
    {
        check_boundary_geometry<3>();
    }
}
