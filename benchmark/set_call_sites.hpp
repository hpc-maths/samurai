// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// The call sites: the set expressions of the library functions samurai runs
// at every time step, copied from the source cited above each. Only the
// access to the operands differs: they are read from bench::SiteInputs
// instead of the mesh. Each site passes every set it evaluates to
// f(level, set).

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>

#include <xtensor/views/xview.hpp>

#include <samurai/level_cell_list.hpp>
#include <samurai/stencil.hpp>
#include <samurai/subset/node.hpp>

#include "site_inputs.hpp"

namespace bench::sites
{
    // MRMesh::update_sub_mesh_impl (mr/mesh.hpp): the ghosts the prediction
    // needs one and two levels below each cell, then the projection cells.
    template <std::size_t dim, class SubdomainAt, class F>
    void mesh_prediction_ghosts(const SiteInputs<dim>& in, SubdomainAt&& subdomain_at, F&& f)
    {
        constexpr int pred_r = prediction_stencil_radius;
        const auto& cells    = in.cells;
        const auto& cag      = in.cells_and_ghosts;
        for (std::size_t level = cells.min_level(); level <= cells.max_level(); ++level)
        {
            if (level < 2 || cells[level].empty())
            {
                continue;
            }
            f(level - 2, samurai::nestedExpand(samurai::self(cells[level]).on(level - 2), pred_r));
            f(level - 1,
              samurai::intersection(samurai::nestedExpand(samurai::self(cag[level]).on(level - 1), pred_r),
                                    samurai::nestedExpand(subdomain_at(level - 1), pred_r)));
        }
        for (std::size_t level = in.reference.min_level(); level <= in.reference.max_level(); ++level)
        {
            f(level, samurai::intersection(in.reference[level], in.union_[level]));
        }
    }

    // What the library does.
    template <std::size_t dim>
    auto projected_subdomain(const SiteInputs<dim>& in)
    {
        return [&in](std::size_t level)
        {
            return samurai::self(in.subdomain()).on(level);
        };
    }

    // Not what the library does: the pyramid the mesh already stores.
    template <std::size_t dim>
    auto cached_subdomain(const SiteInputs<dim>& in)
    {
        return [&in](std::size_t level) -> const auto&
        {
            return in.subdomain(level);
        };
    }

    // Mesh_base::construct_corners (mesh.hpp). Runs at every mesh
    // construction although its result only depends on the domain.
    template <std::size_t dim, class F>
    void corners(const SiteInputs<dim>& in, F&& f)
    {
        if constexpr (dim > 1)
        {
            using direction_t      = samurai::DirectionVector<dim>;
            const auto& domain_lca = in.domain();
            samurai::for_each_diagonal_direction<dim>(
                [&](const auto& direction)
                {
                    std::array<direction_t, dim> shifts;
                    std::size_t n = 0;
                    for (std::size_t i = 0; i < dim; ++i)
                    {
                        if (direction[i] != 0)
                        {
                            shifts[n].fill(0);
                            shifts[n][i] = -direction[i];
                            ++n;
                        }
                    }
                    if (n == 2)
                    {
                        f(domain_lca.level(),
                          samurai::difference(
                              domain_lca,
                              samurai::union_(samurai::translate(domain_lca, shifts[0]), samurai::translate(domain_lca, shifts[1]))));
                    }
                    else if constexpr (dim > 2)
                    {
                        f(domain_lca.level(),
                          samurai::difference(domain_lca,
                                              samurai::union_(samurai::translate(domain_lca, shifts[0]),
                                                              samurai::translate(domain_lca, shifts[1]),
                                                              samurai::translate(domain_lca, shifts[2]))));
                    }
                });
        }
    }

    // update_ghost_mr (algorithm/update_ghost_mr.hpp): projection of the
    // ghosts, bottom-up.
    template <std::size_t dim, class F>
    void ghost_projection(const SiteInputs<dim>& in, F&& f)
    {
        for (std::size_t level = in.max_level; level >= 1; --level)
        {
            f(level - 1, samurai::intersection(in.reference[level], in.proj_cells[level - 1]).on(level - 1));
        }
    }

    // update_ghost_mr: prediction of the ghosts, top-down.
    template <std::size_t dim, class F>
    void ghost_prediction(const SiteInputs<dim>& in, F&& f)
    {
        for (std::size_t level = 1; level <= in.max_level; ++level)
        {
            auto pred_ghosts = samurai::difference(in.all_cells()[level], samurai::union_(in.cells[level], in.proj_cells[level]));
            f(level, samurai::intersection(pred_ghosts, in.subdomain(level), in.all_cells()[level - 1]).on(level));
        }
    }

    // domain_boundary(mesh, level, direction) (boundary.hpp), which calls
    // boundary_layer with a layer width of 1.
    template <std::size_t dim, class Vector>
    auto domain_boundary(const SiteInputs<dim>& in, std::size_t level, const Vector& direction)
    {
        const std::size_t layer_width = 1;
        const auto& cells             = in.cells[level];
        return samurai::difference(cells, samurai::translate(samurai::self(in.domain()).on(level), -layer_width * direction));
    }

    // domain_boundary_outer_layer(mesh, level, direction, layer_width)
    // (boundary.hpp). The LevelCellList is built without the origin point and
    // scaling factor of the mesh: they play no part in the set algebra.
    template <std::size_t dim>
    auto
    domain_boundary_outer_layer(const SiteInputs<dim>& in, std::size_t level, const samurai::DirectionVector<dim>& direction, int layer_width)
    {
        using lca_t = typename SiteInputs<dim>::lca_t;
        using lcl_t = samurai::LevelCellList<dim>;

        auto domain = samurai::self(in.domain(level));

        lcl_t outer_boundary_lcl(level);

        auto inner_boundary = domain_boundary(in, level, direction);
        for (int layer = 1; layer <= layer_width; ++layer)
        {
            auto outer_layer = samurai::difference(samurai::translate(inner_boundary, layer * direction), domain);
            outer_layer(
                [&](const auto& i, const auto& index)
                {
                    outer_boundary_lcl[index].add_interval({i});
                });
        }
        return lca_t(outer_boundary_lcl);
    }

    // update_outer_ghosts (algorithm/update_outer_ghost.hpp) without the
    // boundary condition itself: project_corner_below, project_bc and
    // predict_bc, with one layer of boundary ghosts (Dirichlet). predict_bc
    // builds a LevelCellArray (domain_boundary_outer_layer) on every call.
    template <std::size_t dim, class DomainAt, class F>
    void outer_ghosts(const SiteInputs<dim>& in, DomainAt&& domain_at, F&& f)
    {
        constexpr int n_bc_ghosts      = 1;
        const std::size_t lowest_level = in.min_level > 0 ? in.min_level - 1 : 0;
        for (std::size_t level = in.max_level + 1; level-- > lowest_level;)
        {
            if constexpr (dim > 1)
            {
                if (level >= in.min_level)
                {
                    samurai::for_each_diagonal_direction<dim>(
                        [&](const auto& direction)
                        {
                            for (std::size_t delta_l = 1; delta_l <= 2 && delta_l <= level; ++delta_l)
                            {
                                auto fine_inner_corner = samurai::self(in.corner(direction)).on(level);
                                auto fine_outer_corner = samurai::intersection(samurai::translate(fine_inner_corner, direction),
                                                                               in.reference[level]);
                                f(level - delta_l,
                                  samurai::intersection(fine_outer_corner.on(level - delta_l), in.reference[level - delta_l]));
                            }
                        });
                }
            }
            samurai::for_each_cartesian_direction<dim>(
                [&](auto, const auto& direction)
                {
                    if (level < in.max_level)
                    {
                        auto outside_layer = samurai::difference(samurai::translate(in.union_[level], direction), domain_at(level));
                        f(level, samurai::intersection(outside_layer, in.reference[level]).on(level));
                    }
                    if (level < in.max_level && level >= in.min_level)
                    {
                        auto bc_ghosts = domain_boundary_outer_layer(in, level, direction, n_bc_ghosts);
                        f(level + 1, samurai::intersection(bc_ghosts, in.reference[level + 1]).on(level + 1));
                    }
                });
        }
    }

    // What the library does.
    template <std::size_t dim>
    auto projected_domain(const SiteInputs<dim>& in)
    {
        return [&in](std::size_t level)
        {
            return samurai::self(in.domain()).on(level);
        };
    }

    // Not what the library does: the pyramid the mesh already stores.
    template <std::size_t dim>
    auto cached_domain(const SiteInputs<dim>& in)
    {
        return [&in](std::size_t level) -> const auto&
        {
            return in.domain(level);
        };
    }

    // Adapt::harten (mr/adapt.hpp): detail computation.
    template <std::size_t dim, class F>
    void detail(const SiteInputs<dim>& in, F&& f)
    {
        const std::size_t min_level = in.min_level;
        for (std::size_t level = (min_level > 0) ? min_level - 1 : 0; level < in.max_level; ++level)
        {
            f(level, samurai::intersection(in.all_cells()[level], samurai::union_(in.cells[level + 1], in.cells[level + 2])).on(level));
        }
    }

    // Adapt::harten: tagging (mr_criteria) and tag finalization (maximum),
    // which evaluate the same set.
    template <std::size_t dim, class F>
    void tag(const SiteInputs<dim>& in, F&& f)
    {
        for (std::size_t level = std::max<std::size_t>(in.min_level, 1); level <= in.max_level; ++level)
        {
            f(level - 1, samurai::intersection(in.cells[level], in.all_cells()[level - 1]).on(level - 1));
        }
    }

    // for_each_interface (interface.hpp): faces between cells of the same
    // level, then between a cell and a finer one.
    template <std::size_t dim, class F>
    void interfaces(const SiteInputs<dim>& in, F&& f)
    {
        const auto& cells     = in.cells;
        const auto directions = samurai::positive_cartesian_directions<dim>();
        for (std::size_t d = 0; d < dim; ++d)
        {
            const samurai::DirectionVector<dim> direction = xt::view(directions, d);
            for (std::size_t level = cells.min_level(); level <= cells.max_level(); ++level)
            {
                f(level, samurai::intersection(cells[level], samurai::translate(cells[level], -direction)));
                if (level < cells.max_level())
                {
                    f(level + 1, samurai::intersection(cells[level], samurai::translate(cells[level + 1], -direction)).on(level + 1));
                    f(level + 1, samurai::intersection(cells[level], samurai::translate(cells[level + 1], direction)).on(level + 1));
                }
            }
        }
    }

    // update_fields (algorithm/update_fields.hpp): copy of the unchanged
    // cells, then coarsening and refinement between the old and new meshes.
    template <std::size_t dim, class F>
    void field_transfer(const SiteInputs<dim>& in, const SiteInputs<dim>& new_in, F&& f)
    {
        for (std::size_t level = in.min_level; level <= in.max_level; ++level)
        {
            f(level, samurai::intersection(in.reference[level], new_in.cells[level]));
        }
        for (std::size_t level = in.min_level + 1; level <= in.max_level; ++level)
        {
            f(level - 1, samurai::intersection(in.cells[level], new_in.cells[level - 1]).on(level - 1));
            f(level - 1, samurai::intersection(new_in.cells[level], in.cells[level - 1]).on(level - 1));
        }
    }
} // namespace bench::sites
