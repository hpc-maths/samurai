// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// The call sites: the set expressions of the library functions samurai runs
// at every time step, copied from the source cited above each and written
// against the backend B (set_algebra_backend.hpp) through A = Algebra<B>:
// samurai::f(...) becomes A::f(...) and s.on(l) becomes A::on(s, l). The
// operands are read from bench::SiteInputs instead of the mesh. Each site
// passes every set it evaluates to f(level, set).

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <type_traits>
#include <utility>
#include <vector>

#include <xtensor/views/xview.hpp>

#include <samurai/stencil.hpp>

#include "set_algebra_backend.hpp"
#include "site_inputs.hpp"

namespace bench::sites
{
    //////////////////////////////////////////////////////////////////////
    // Periodicity
    //////////////////////////////////////////////////////////////////////

    template <std::size_t d, std::size_t dim>
    void periodic_directions(const std::array<int, dim>& translation,
                             int delta,
                             const std::array<bool, dim>& is_periodic,
                             std::vector<samurai::DirectionVector<dim>>& directions,
                             samurai::DirectionVector<dim>& current)
    {
        // start_shift_neg (algorithm/graduation.hpp:32-35)
        const auto shifted = [delta](int value)
        {
            return delta >= 0 ? value >> delta : value << -delta;
        };
        const auto next = [&]()
        {
            if constexpr (d == dim - 1)
            {
                directions.push_back(current);
            }
            else
            {
                periodic_directions<d + 1>(translation, delta, is_periodic, directions, current);
            }
        };
        if (is_periodic[d])
        {
            current[d] = shifted(-translation[d]);
            next();
            current[d] = shifted(translation[d]);
            next();
        }
        current[d] = 0;
        next();
    }

    // detail::get_periodic_directions(translation, delta, is_periodic)
    // (algorithm/graduation.hpp:68-76): the translations by -translation[d],
    // +translation[d] (shifted right by delta) or 0 along each periodic axis
    // d, and 0 along the others, but the zero vector, in the order of the
    // library. With a translation of 1 along every axis and delta = 0, these
    // are the directions of detail::get_periodic_directions(is_periodic)
    // (mesh.hpp:45-79).
    template <std::size_t dim>
    std::vector<samurai::DirectionVector<dim>>
    periodic_directions(const std::array<int, dim>& translation, int delta, const std::array<bool, dim>& is_periodic)
    {
        samurai::DirectionVector<dim> current;
        current.fill(0);
        std::vector<samurai::DirectionVector<dim>> directions;
        periodic_directions<0>(translation, delta, is_periodic, directions, current);
        directions.pop_back();
        return directions;
    }

    template <std::size_t dim, class CA>
    bool is_periodic(const SiteInputs<dim, CA>& in)
    {
        return std::any_of(in.periodic.begin(),
                           in.periodic.end(),
                           [](bool p)
                           {
                               return p;
                           });
    }

    // The number of cells of the domain along each axis at max_level, the
    // translation of a periodic copy of the mesh (mr/mesh.hpp:291-298).
    template <std::size_t dim, class CA>
    std::array<int, dim> domain_extent(const SiteInputs<dim, CA>& in)
    {
        std::array<int, dim> extent;
        for (std::size_t d = 0; d < dim; ++d)
        {
            extent[d] = in.domain_max[d] - in.domain_min[d];
        }
        return extent;
    }

    // The directions of the periodic copies of the cells
    // MRMesh::update_sub_mesh_impl adds (mr/mesh.hpp:288-299), at max_level:
    // none when the mesh is not periodic.
    template <std::size_t dim, class CA>
    std::vector<samurai::DirectionVector<dim>> periodic_copies(const SiteInputs<dim, CA>& in)
    {
        if (!is_periodic(in))
        {
            return {};
        }
        return periodic_directions(domain_extent(in), 0, in.periodic);
    }

    // get_periodic_shift(mesh.domain(), level, d) (utils.hpp:489-502).
    template <std::size_t dim, class CA>
    samurai::DirectionVector<dim> periodic_shift(const SiteInputs<dim, CA>& in, std::size_t level, std::size_t d)
    {
        const std::size_t delta_l = in.max_level - level;
        samurai::DirectionVector<dim> shift;
        shift.fill(0);
        shift[d] = (in.domain_max[d] - in.domain_min[d]) >> delta_l;
        return shift;
    }

    // d >> delta_l, componentwise, as xtensor computes it.
    template <std::size_t dim>
    samurai::DirectionVector<dim> shift_right(const samurai::DirectionVector<dim>& d, int delta_l)
    {
        samurai::DirectionVector<dim> out;
        for (std::size_t k = 0; k < dim; ++k)
        {
            out[k] = d[k] >> delta_l;
        }
        return out;
    }

    // The meshes of the MPI neighbours of a rank: pointers into the ranks
    // of its partition, in the order of SiteInputs::neighbours. Empty for a
    // whole mesh.
    template <std::size_t dim, class CA>
    using neighbours_t = std::vector<const SiteInputs<dim, CA>*>;

    template <std::size_t dim, class CA>
    neighbours_t<dim, CA> neighbours_of(const std::vector<SiteInputs<dim, CA>>& ranks, std::size_t rank)
    {
        neighbours_t<dim, CA> neighbours;
        for (const std::size_t n : ranks[rank].neighbours)
        {
            neighbours.push_back(&ranks[n]);
        }
        return neighbours;
    }

    // Calls g(level) for the levels of `cells` that have cells, as
    // for_each_level (algorithm.hpp:52-62) does.
    template <class B, class Cells, class G>
    void for_each_nonempty_level(const Cells& cells, G&& g)
    {
        for (std::size_t level = cells.min_level(); level <= cells.max_level(); ++level)
        {
            if (!Algebra<B>::empty(cells[level]))
            {
                g(level);
            }
        }
    }

    // add_prediction_ghosts of MRMesh::update_sub_mesh_impl
    // (mr/mesh.hpp:334-382) for the cells of one level: the ghosts the
    // prediction needs one level below (passed first to g(level, set)), then
    // two levels below, for the cells and for each periodic copy of them
    // (directions, at max_level). cag gives the cells and ghosts of the level.
    template <class B, std::size_t dim, class CA, class Cag, class SubdomainAt, class G>
    void prediction_ghosts(const SiteInputs<dim, CA>& in,
                           const Cag& cag,
                           SubdomainAt&& subdomain_at,
                           const std::vector<samurai::DirectionVector<dim>>& directions,
                           std::size_t level,
                           G&& g)
    {
        using A              = Algebra<B>;
        constexpr int pred_r = prediction_stencil_radius;
        const auto add       = [&](const auto& subset_cells, const auto& subset_cells_and_ghosts)
        {
            g(level - 1, subset_cells_and_ghosts);
            if (level - 1 > 0)
            {
                g(level - 2, subset_cells);
            }
        };
        // own part
        add(A::expand(A::on(in.cells[level], level - 2), pred_r),
            A::intersection(A::expand(A::on(cag[level], level - 1), pred_r), A::expand(subdomain_at(level - 1), pred_r)));
        // periodic part
        const int delta_l = static_cast<int>(in.max_level - level);
        for (const auto& d : directions)
        {
            const auto t = shift_right(d, delta_l);
            add(A::intersection(A::expand(A::on(A::translate(in.cells[level], t), level - 2), pred_r), subdomain_at(level - 2)),
                A::intersection(A::expand(A::on(A::translate(cag[level], t), level - 1), pred_r), subdomain_at(level - 1)));
        }
    }

    // MRMesh::update_sub_mesh_impl (mr/mesh.hpp:357-382, 444): the ghosts the
    // prediction needs one and two levels below each cell, and below its
    // periodic copies, then the projection cells. The library visits the
    // levels that have cells, from min_level >= 2 here.
    template <class B, std::size_t dim, class CA, class SubdomainAt, class F>
    void mesh_prediction_ghosts(const SiteInputs<dim, CA>& in, SubdomainAt&& subdomain_at, F&& f)
    {
        using A               = Algebra<B>;
        const auto& cells     = in.cells;
        const auto directions = periodic_copies(in);
        for (std::size_t level = cells.min_level(); level <= cells.max_level(); ++level)
        {
            if (level < 2 || A::empty(cells[level]))
            {
                continue;
            }
            prediction_ghosts<B>(in, in.cells_and_ghosts, subdomain_at, directions, level, f);
        }
        for (std::size_t level = in.reference.min_level(); level <= in.reference.max_level(); ++level)
        {
            f(level, A::intersection(in.reference[level], in.union_[level]));
        }
    }

    // What the library does.
    template <class B, std::size_t dim, class CA>
    auto projected_subdomain(const SiteInputs<dim, CA>& in)
    {
        return [&in](std::size_t level)
        {
            return Algebra<B>::on(in.subdomain(), level);
        };
    }

    // Not what the library does: the pyramid the mesh already stores.
    template <std::size_t dim, class CA>
    auto cached_subdomain(const SiteInputs<dim, CA>& in)
    {
        return [&in](std::size_t level) -> const auto&
        {
            return in.subdomain(level);
        };
    }

    // Mesh_base::construct_corners (mesh.hpp). Runs at every mesh
    // construction although its result only depends on the domain.
    template <class B, std::size_t dim, class CA, class F>
    void corners(const SiteInputs<dim, CA>& in, F&& f)
    {
        using A = Algebra<B>;
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
                        f(in.max_level,
                          A::difference(domain_lca, A::union_(A::translate(domain_lca, shifts[0]), A::translate(domain_lca, shifts[1]))));
                    }
                    else if constexpr (dim > 2)
                    {
                        f(in.max_level,
                          A::difference(domain_lca,
                                        A::union_(A::translate(domain_lca, shifts[0]),
                                                  A::translate(domain_lca, shifts[1]),
                                                  A::translate(domain_lca, shifts[2]))));
                    }
                });
        }
    }

    // update_ghost_mr (algorithm/update_ghost_mr.hpp): projection of the
    // ghosts, bottom-up.
    template <class B, std::size_t dim, class CA, class F>
    void ghost_projection(const SiteInputs<dim, CA>& in, F&& f)
    {
        using A = Algebra<B>;
        for (std::size_t level = in.max_level; level >= 1; --level)
        {
            f(level - 1, A::on(A::intersection(in.reference[level], in.proj_cells[level - 1]), level - 1));
        }
    }

    // update_ghost_mr: prediction of the ghosts, top-down.
    template <class B, std::size_t dim, class CA, class F>
    void ghost_prediction(const SiteInputs<dim, CA>& in, F&& f)
    {
        using A = Algebra<B>;
        for (std::size_t level = 1; level <= in.max_level; ++level)
        {
            auto pred_ghosts = A::difference(in.all_cells()[level], A::union_(in.cells[level], in.proj_cells[level]));
            f(level, A::on(A::intersection(pred_ghosts, in.subdomain(level), in.all_cells()[level - 1]), level));
        }
    }

    // domain_boundary(mesh, level, direction) (boundary.hpp), which calls
    // boundary_layer with a layer width of 1.
    template <class B, std::size_t dim, class CA, class Vector>
    auto domain_boundary(const SiteInputs<dim, CA>& in, std::size_t level, const Vector& direction)
    {
        using A = Algebra<B>;
        // An int where boundary_layer takes a std::size_t: the negated
        // width is then -1, not an unsigned value that wraps back to -1.
        const int layer_width = 1;
        const auto& cells     = in.cells[level];
        return A::difference(cells, A::translate(A::on(in.domain(), level), -layer_width * direction));
    }

    // domain_boundary_outer_layer(mesh, level, direction, layer_width)
    // (boundary.hpp), which fills a LevelCellList: the builder of the
    // backend. Without the origin point and scaling factor of the mesh: they
    // play no part in the set algebra.
    template <class B, std::size_t dim, class CA>
    auto domain_boundary_outer_layer(const SiteInputs<dim, CA>& in,
                                     std::size_t level,
                                     const samurai::DirectionVector<dim>& direction,
                                     int layer_width)
    {
        using A = Algebra<B>;

        auto domain = A::lift(in.domain(level));

        auto outer_boundary = A::template builder<dim>(level);

        auto inner_boundary = domain_boundary<B>(in, level, direction);
        for (int layer = 1; layer <= layer_width; ++layer)
        {
            auto outer_layer = A::difference(A::translate(inner_boundary, layer * direction), domain);
            A::traverse(outer_layer,
                        [&](const auto& i, const auto& index)
                        {
                            outer_boundary.add_interval(i, index);
                        });
        }
        return outer_boundary.build();
    }

    // domain_boundary_outer_layer(mesh, level, layer_width) (boundary.hpp:51-75):
    // the outer layers of every Cartesian direction, in one LevelCellList.
    template <class B, std::size_t dim, class CA>
    auto domain_boundary_outer_layer(const SiteInputs<dim, CA>& in, std::size_t level, int layer_width)
    {
        using A             = Algebra<B>;
        auto domain         = A::lift(in.domain(level));
        auto outer_boundary = A::template builder<dim>(level);
        samurai::for_each_cartesian_direction<dim>(
            [&](const auto& direction)
            {
                auto inner_boundary = domain_boundary<B>(in, level, direction);
                for (int layer = 1; layer <= layer_width; ++layer)
                {
                    A::traverse(A::difference(A::translate(inner_boundary, layer * direction), domain),
                                [&](const auto& i, const auto& index)
                                {
                                    outer_boundary.add_interval(i, index);
                                });
                }
            });
        return outer_boundary.build();
    }

    // Whether one of the axes `direction` moves along is periodic: such a
    // direction has no boundary, and update_outer_ghosts and the boundary
    // conditions skip it (update_outer_ghost.hpp:357-368, 386, 433;
    // bc/apply_field_bc.hpp:102-111, 226-232, 698-707).
    template <std::size_t dim, class CA, class Direction>
    bool crosses_periodic_axis(const SiteInputs<dim, CA>& in, const Direction& direction)
    {
        for (std::size_t d = 0; d < dim; ++d)
        {
            if (direction[d] != 0 && in.periodic[d])
            {
                return true;
            }
        }
        return false;
    }

    // The ghosts of the Dirichlet condition of the adapted meshes, and
    // max_coarse_layer of update_outer_ghosts: one layer.
    inline constexpr int n_bc_ghosts = 1;

    // The projection ghosts project_bc(proj_level, direction, layer = 1,
    // field) fills (algorithm/update_outer_ghost.hpp:23-61): those outside
    // the domain, and on a domain that is not a box, those not filled by the
    // boundary condition of another direction, through a LevelCellArray
    // built at every call (domain_boundary_outer_layer). Passes the set to
    // g(proj_level, set).
    template <class B, std::size_t dim, class CA, class Domain, class G>
    void projection_bc_ghosts(const SiteInputs<dim, CA>& in,
                              std::size_t proj_level,
                              const Domain& domain,
                              const samurai::DirectionVector<dim>& direction,
                              G&& g)
    {
        using A                = Algebra<B>;
        constexpr int layer    = 1;
        const auto& inner      = in.union_[proj_level];
        auto outside_layer     = A::difference(A::translate(inner, layer * direction), A::translate(domain, (layer - 1) * direction));
        auto projection_ghosts = A::on(A::intersection(outside_layer, in.reference[proj_level]), proj_level);
        if (in.domain_is_box)
        {
            g(proj_level, projection_ghosts);
        }
        else
        {
            const auto bc_ghosts_in_other_directions = domain_boundary_outer_layer<B>(in, proj_level, n_bc_ghosts);
            g(proj_level, A::difference(projection_ghosts, bc_ghosts_in_other_directions));
        }
    }

    // update_outer_ghosts(level, field) (algorithm/update_outer_ghost.hpp:341)
    // without the boundary condition itself: project_corner_below,
    // project_bc and predict_bc, with one layer of boundary ghosts
    // (Dirichlet). predict_bc builds a LevelCellArray
    // (domain_boundary_outer_layer) on every call, and two on a domain that
    // is not a box. update_ghost_mr calls it for every level from max_level
    // down to 0 (update_ghost_mr.hpp:204-208): project_bc runs below the min
    // level too.
    template <class B, std::size_t dim, class CA, class DomainAt, class F>
    void outer_ghosts(const SiteInputs<dim, CA>& in, DomainAt&& domain_at, F&& f)
    {
        using A = Algebra<B>;
        for (std::size_t level = in.max_level + 1; level-- > 0;)
        {
            if constexpr (dim > 1)
            {
                if (level >= in.min_level)
                {
                    samurai::for_each_diagonal_direction<dim>(
                        [&](const auto& direction)
                        {
                            if (crosses_periodic_axis(in, direction))
                            {
                                return;
                            }
                            for (std::size_t delta_l = 1; delta_l <= 2 && delta_l <= level; ++delta_l)
                            {
                                auto fine_inner_corner = A::on(in.corner(direction), level);
                                auto fine_outer_corner = A::intersection(A::translate(fine_inner_corner, direction), in.reference[level]);
                                f(level - delta_l, A::intersection(A::on(fine_outer_corner, level - delta_l), in.reference[level - delta_l]));
                            }
                        });
                }
            }
            samurai::for_each_cartesian_direction<dim>(
                [&](std::size_t axis, const auto& direction)
                {
                    if (in.periodic[axis])
                    {
                        return;
                    }
                    if (level < in.max_level)
                    {
                        projection_bc_ghosts<B>(in, level, domain_at(level), direction, f);
                    }
                    if (level < in.max_level && level >= in.min_level)
                    {
                        // predict_bc(level + 1, direction, field) (update_outer_ghost.hpp:197-230)
                        const auto bc_ghosts           = domain_boundary_outer_layer<B>(in, level, direction, n_bc_ghosts);
                        auto outside_prediction_ghosts = A::on(A::intersection(bc_ghosts, in.reference[level + 1]), level + 1);
                        if (in.domain_is_box)
                        {
                            f(level + 1, outside_prediction_ghosts);
                        }
                        else
                        {
                            const auto bc_ghosts_in_other_directions = domain_boundary_outer_layer<B>(in, level + 1, n_bc_ghosts);
                            f(level + 1, A::difference(outside_prediction_ghosts, bc_ghosts_in_other_directions));
                        }
                    }
                });
        }
    }

    // What the library does.
    template <class B, std::size_t dim, class CA>
    auto projected_domain(const SiteInputs<dim, CA>& in)
    {
        return [&in](std::size_t level)
        {
            return Algebra<B>::on(in.domain(), level);
        };
    }

    // Not what the library does: the pyramid the mesh already stores.
    template <std::size_t dim, class CA>
    auto cached_domain(const SiteInputs<dim, CA>& in)
    {
        return [&in](std::size_t level) -> const auto&
        {
            return in.domain(level);
        };
    }

    // Adapt::harten (mr/adapt.hpp): detail computation. At the last level,
    // cells[level + 2] is cells[max_level + 1], as in the library: an empty
    // level past the mesh, which CellArray stores (it holds max_size + 1
    // levels), so the read is safe.
    template <class B, std::size_t dim, class CA, class F>
    void detail(const SiteInputs<dim, CA>& in, F&& f)
    {
        using A                     = Algebra<B>;
        const std::size_t min_level = in.min_level;
        for (std::size_t level = (min_level > 0) ? min_level - 1 : 0; level < in.max_level; ++level)
        {
            f(level, A::on(A::intersection(in.all_cells()[level], A::union_(in.cells[level + 1], in.cells[level + 2])), level));
        }
    }

    // Adapt::harten: tagging (mr_criteria) and tag finalization (maximum),
    // which evaluate the same set.
    template <class B, std::size_t dim, class CA, class F>
    void tag(const SiteInputs<dim, CA>& in, F&& f)
    {
        using A = Algebra<B>;
        for (std::size_t level = std::max<std::size_t>(in.min_level, 1); level <= in.max_level; ++level)
        {
            f(level - 1, A::on(A::intersection(in.cells[level], in.all_cells()[level - 1]), level - 1));
        }
    }

    // for_each_interior_interface (interface.hpp:340-361) in each positive
    // Cartesian direction, at each level: faces between cells of the same
    // level (lines 34-108), then between a cell and a finer one on either
    // side (lines 125-223 and 239-323), each also across the periodic
    // boundary when the mesh is periodic along the direction, and with the
    // cells of each MPI neighbour (lines 93-106, 207-221, 311-325).
    template <class B, std::size_t dim, class CA, class F>
    void interfaces(const SiteInputs<dim, CA>& in, const neighbours_t<dim, CA>& neighbours, F&& f)
    {
        using A               = Algebra<B>;
        const auto& cells     = in.cells;
        const auto directions = samurai::positive_cartesian_directions<dim>();
        for (std::size_t d = 0; d < dim; ++d)
        {
            const samurai::DirectionVector<dim> direction = xt::view(directions, d);
            for (std::size_t level = cells.min_level(); level <= cells.max_level(); ++level)
            {
                // same level
                const auto same_level = [&](const auto& c, const auto& shifted)
                {
                    f(level, A::intersection(c, shifted));
                };
                const auto& own = cells[level];
                // across the periodic boundary, with the cells `other`
                const auto periodic_same_level = [&](const auto& other)
                {
                    if (in.periodic[d])
                    {
                        const auto shift = periodic_shift(in, level, d);
                        same_level(own, A::translate(A::translate(other, shift), -direction));
                        same_level(A::translate(other, -shift), A::translate(own, -direction));
                    }
                };
                same_level(own, A::translate(own, -direction));
                periodic_same_level(own);
                for (const auto* neighbour : neighbours)
                {
                    const auto& other = neighbour->cells[level];
                    same_level(own, A::translate(other, -direction));
                    same_level(other, A::translate(own, -direction));
                    periodic_same_level(other);
                }
                if (level >= cells.max_level())
                {
                    continue;
                }
                const auto& fine = cells[level + 1];
                // level jumps, in the direction (sign = -1: the fine cells
                // translated by -direction) then in the opposite one
                for (const int sign : {-1, 1})
                {
                    const auto jump = [&](const auto& coarse_cells, const auto& fine_cells)
                    {
                        f(level + 1, A::on(A::intersection(coarse_cells, A::translate(fine_cells, sign * direction)), level + 1));
                    };
                    // across the periodic boundary, with the fine cells
                    // `other_fine` and the coarse cells `other_coarse`
                    const auto periodic_jumps = [&](const auto& other_fine, const auto& other_coarse)
                    {
                        if (in.periodic[d])
                        {
                            const auto fine_shift   = periodic_shift(in, level + 1, d);
                            const auto coarse_shift = periodic_shift(in, level, d);
                            jump(own, A::translate(other_fine, -sign * fine_shift));
                            jump(A::translate(other_coarse, sign * coarse_shift), fine);
                        }
                    };
                    jump(own, fine);
                    periodic_jumps(fine, own);
                    for (const auto* neighbour : neighbours)
                    {
                        jump(own, neighbour->cells[level + 1]);
                        jump(neighbour->cells[level], fine);
                        periodic_jumps(neighbour->cells[level + 1], neighbour->cells[level]);
                    }
                }
            }
        }
    }

    template <class B, std::size_t dim, class CA, class F>
    void interfaces(const SiteInputs<dim, CA>& in, F&& f)
    {
        interfaces<B>(in, neighbours_t<dim, CA>{}, std::forward<F>(f));
    }

    // update_fields (algorithm/update_fields.hpp): copy of the unchanged
    // cells, then coarsening and refinement between the old and new meshes.
    template <class B, std::size_t dim, class CA, class F>
    void field_transfer(const SiteInputs<dim, CA>& in, const SiteInputs<dim, CA>& new_in, F&& f)
    {
        using A = Algebra<B>;
        for (std::size_t level = in.min_level; level <= in.max_level; ++level)
        {
            f(level, A::intersection(in.reference[level], new_in.cells[level]));
        }
        for (std::size_t level = in.min_level + 1; level <= in.max_level; ++level)
        {
            f(level - 1, A::on(A::intersection(in.cells[level], new_in.cells[level - 1]), level - 1));
            f(level - 1, A::on(A::intersection(new_in.cells[level], in.cells[level - 1]), level - 1));
        }
    }

    //////////////////////////////////////////////////////////////////////
    // Sites that build level cell arrays
    //////////////////////////////////////////////////////////////////////

    // These sites pass every level cell array they keep to emit(level, lca),
    // see bench::run_build_site.

    template <std::size_t dim>
    using interval_t = typename samurai::LevelCellList<dim>::interval_t;

    template <std::size_t dim>
    using index_t = typename samurai::LevelCellList<dim>::index_yz_t;

    template <class B, std::size_t dim>
    using lca_t = typename B::template lca_t<dim>;

    // The ghost width of the mesh configuration (mesh_config::parse_args).
    template <std::size_t dim, class CA>
    int ghost_width(const SiteInputs<dim, CA>& in)
    {
        return std::max(in.max_stencil_radius, prediction_stencil_radius);
    }

    // The (y, z) of an interval of a traversal, whatever its type in the
    // backend.
    template <std::size_t dim, class Index>
    index_t<dim> to_index(const Index& index)
    {
        index_t<dim> out;
        for (std::size_t k = 0; k + 1 < dim; ++k)
        {
            out[k] = index[k];
        }
        return out;
    }

    // The callback of a traversal that adds each interval to `builder`, as
    // the library adds the intervals of a set to a LevelCellList.
    template <std::size_t dim, class Builder>
    auto add_to(Builder& builder)
    {
        return [&builder](const auto& i, const auto& index)
        {
            builder.add_interval(interval_t<dim>{i.start, i.end}, to_index<dim>(index));
        };
    }

    // The counterpart of samurai's CellList for the backend B: one builder
    // per level, from 0 to max_level + 1.
    template <class B, std::size_t dim>
    class BuilderList
    {
      public:

        using builder_t = decltype(B::template builder<dim>(0));

        explicit BuilderList(std::size_t max_level)
        {
            m_levels.reserve(max_level + 2);
            for (std::size_t level = 0; level <= max_level + 1; ++level)
            {
                m_levels.push_back(B::template builder<dim>(level));
            }
        }

        builder_t& operator[](std::size_t level)
        {
            return m_levels[level];
        }

        // CellArray{cell_list, false}: the level cell array of every level.
        std::vector<lca_t<B, dim>> build() const
        {
            std::vector<lca_t<B, dim>> levels;
            levels.reserve(m_levels.size());
            for (const auto& builder : m_levels)
            {
                levels.push_back(builder.build());
            }
            return levels;
        }

      private:

        std::vector<builder_t> m_levels;
    };

    // Emits levels[level] for every level.
    template <class Levels, class Emit>
    void emit_levels(const Levels& levels, Emit&& emit)
    {
        for (std::size_t level = 0; level < levels.size(); ++level)
        {
            emit(level, levels[level]);
        }
    }

    // Adds to cell_list[level] the intervals of `cells`, widened by `width`
    // along x and copied on the rows at most `width` away: the scheme ghosts
    // of MRMesh::update_sub_mesh_impl (mr/mesh.hpp:247-258) and of
    // amr::Mesh::update_sub_mesh_impl (amr/mesh.hpp:201-212). The library
    // loops over the intervals of the CellArray; a backend can only traverse
    // them.
    template <class B, std::size_t dim, class CellsCA>
    void add_scheme_ghosts(BuilderList<B, dim>& cell_list, const CellsCA& cells, int width)
    {
        for (std::size_t level = cells.min_level(); level <= cells.max_level(); ++level)
        {
            auto& lcl = cell_list[level];
            Algebra<B>::traverse(cells[level],
                                 [&](const auto& i, const auto& index_yz)
                                 {
                                     const auto yz = to_index<dim>(index_yz);
                                     samurai::static_nested_loop<dim - 1>(
                                         -width,
                                         width + 1,
                                         [&](const auto& stencil)
                                         {
                                             lcl.add_interval(interval_t<dim>{i.start - width, i.end + width}, index_t<dim>(yz + stencil));
                                         });
                                 });
        }
    }

    // The steps of mesh_sub_mesh, which add their cells to its cell list.
    // subdomain_at(level) is the subdomain at `level`.

    // The scheme ghosts the cells of the MPI neighbours give to the
    // subdomain (mr/mesh.hpp:262-281).
    template <class B, std::size_t dim, class CA, class SubdomainAt>
    void add_neighbour_scheme_ghosts(BuilderList<B, dim>& cell_list,
                                     const SiteInputs<dim, CA>& in,
                                     const neighbours_t<dim, CA>& neighbours,
                                     const SubdomainAt& subdomain_at)
    {
        using A          = Algebra<B>;
        const int radius = in.max_stencil_radius;
        for (const auto* neighbour : neighbours)
        {
            for_each_nonempty_level<B>(
                neighbour->cells,
                [&](std::size_t level)
                {
                    A::traverse(A::intersection(A::expand(neighbour->cells[level], radius), A::expand(subdomain_at(level), radius)),
                                add_to<dim>(cell_list[level]));
                });
        }
    }

    // The periodic copies of the cells, own then of the neighbours, near the
    // subdomain (mr/mesh.hpp:283-334).
    template <class B, std::size_t dim, class CA, class Directions, class SubdomainAt>
    void add_periodic_cells(BuilderList<B, dim>& cell_list,
                            const SiteInputs<dim, CA>& in,
                            const neighbours_t<dim, CA>& neighbours,
                            const Directions& directions,
                            const SubdomainAt& subdomain_at)
    {
        using A                     = Algebra<B>;
        const std::size_t max_level = in.max_level;
        const int radius            = in.max_stencil_radius;
        const auto add_copies       = [&](const auto& subset, std::size_t level)
        {
            const int delta_l = static_cast<int>(max_level - level);
            for (const auto& d : directions)
            {
                A::traverse(A::intersection(A::expand(A::translate(subset, shift_right(d, delta_l)), radius),
                                            A::expand(subdomain_at(level), radius)),
                            add_to<dim>(cell_list[level]));
            }
        };
        for_each_nonempty_level<B>(in.cells,
                                   [&](std::size_t level)
                                   {
                                       add_copies(in.cells[level], level);
                                   });
        for (const auto* neighbour : neighbours)
        {
            for_each_nonempty_level<B>(neighbour->cells,
                                       [&](std::size_t level)
                                       {
                                           add_copies(neighbour->cells[level], level);
                                       });
        }
    }

    // The prediction ghosts of the cells of the MPI neighbours and of their
    // periodic copies (mr/mesh.hpp:388-420). add_to_level(level, set) adds
    // the set to the cell list at `level`.
    template <class B, std::size_t dim, class CA, class Directions, class SubdomainAt, class AddToLevel>
    void add_neighbour_prediction_ghosts(const SiteInputs<dim, CA>& in,
                                         const neighbours_t<dim, CA>& neighbours,
                                         const Directions& directions,
                                         const SubdomainAt& subdomain_at,
                                         const AddToLevel& add_to_level)
    {
        using A              = Algebra<B>;
        constexpr int pred_r = prediction_stencil_radius;
        for (const auto* neighbour : neighbours)
        {
            for_each_nonempty_level<B>(
                neighbour->cells,
                [&](std::size_t level)
                {
                    const auto add = [&](const auto& subset_cells, const auto& subset_cells_and_ghosts)
                    {
                        add_to_level(level - 1, subset_cells_and_ghosts);
                        if (level - 1 > 0)
                        {
                            add_to_level(level - 2, subset_cells);
                        }
                    };
                    const auto& n_cells = neighbour->cells[level];
                    const auto& n_cag   = neighbour->cells_and_ghosts[level];
                    add(A::intersection(A::expand(A::on(n_cells, level - 2), pred_r), subdomain_at(level - 2)),
                        A::intersection(A::expand(A::on(n_cag, level - 1), pred_r), subdomain_at(level - 1)));
                    const int delta_l = static_cast<int>(in.max_level - level);
                    for (const auto& d : directions)
                    {
                        const auto t = shift_right(d, delta_l);
                        add(A::intersection(A::expand(A::on(A::translate(n_cells, t), level - 2), pred_r), subdomain_at(level - 2)),
                            A::intersection(A::expand(A::on(A::translate(n_cag, t), level - 1), pred_r), subdomain_at(level - 1)));
                    }
                });
        }
    }

    // The reference cells refined level by level where the union covers
    // them, and the projection cells (mr/mesh.hpp:440-459).
    template <class B, std::size_t dim, class CA, class Reference>
    void
    refine_reference(BuilderList<B, dim>& cell_list, const SiteInputs<dim, CA>& in, Reference& reference, std::vector<lca_t<B, dim>>& proj_cells)
    {
        using A = Algebra<B>;
        for (std::size_t level = 0; level <= in.max_level; ++level)
        {
            if (A::empty(reference[level]))
            {
                continue;
            }
            auto& lcl     = cell_list[level + 1];
            auto lcl_proj = A::template builder<dim>(level);
            A::traverse(A::intersection(reference[level], in.union_[level]),
                        [&](const auto& i, const auto& index_yz)
                        {
                            const auto yz = to_index<dim>(index_yz);
                            samurai::static_nested_loop<dim - 1, 0, 2>(
                                [&](const auto& stencil)
                                {
                                    lcl.add_interval(interval_t<dim>{i.start << 1, i.end << 1}, index_t<dim>((yz << 1) + stencil));
                                });
                            lcl_proj.add_interval(interval_t<dim>{i.start, i.end}, yz);
                        });
            reference[level + 1] = lcl.build();
            proj_cells[level]    = lcl_proj.build();
        }
    }

    // MRMesh::update_sub_mesh_impl (mr/mesh.hpp:229-461), at every mesh
    // construction: the cells and ghosts (scheme ghosts added row by row to a
    // cell list, in no order), the scheme ghosts the cells of the MPI
    // neighbours give to the subdomain, the periodic copies of the cells, own
    // then of the neighbours, the prediction ghosts (prediction_ghosts, added
    // to the same cell list), then those of the cells of the neighbours and
    // of their periodic copies, then the reference cells, refined level by
    // level where the union covers them, and the projection cells: one step
    // each, in the order of the library. Emits cells_and_ghosts, then
    // reference, then proj_cells, each from level 0 to max_level + 1. In a
    // partition, the neighbours have their own cells_and_ghosts: the scheme
    // ghosts of their cells, which update_meshid_neighbour sends (line 389).
    template <class B, std::size_t dim, class CA, class Emit>
    void mesh_sub_mesh(const SiteInputs<dim, CA>& in, const neighbours_t<dim, CA>& neighbours, Emit&& emit)
    {
        using A = Algebra<B>;

        BuilderList<B, dim> cell_list(in.max_level);
        add_scheme_ghosts(cell_list, in.cells, in.max_stencil_radius);
        const auto cells_and_ghosts = cell_list.build();

        const auto subdomain_at = [&](std::size_t level)
        {
            return A::on(in.subdomain(), level);
        };
        add_neighbour_scheme_ghosts(cell_list, in, neighbours, subdomain_at);
        const auto directions = periodic_copies(in);
        add_periodic_cells(cell_list, in, neighbours, directions, subdomain_at);

        std::vector<lca_t<B, dim>> proj_cells;
        for (std::size_t level = 0; level <= in.max_level + 1; ++level)
        {
            proj_cells.push_back(A::template builder<dim>(level).build());
        }

        if (in.max_level == in.min_level)
        {
            const auto reference = cell_list.build();
            emit_levels(cells_and_ghosts, emit);
            emit_levels(reference, emit);
            emit_levels(proj_cells, emit);
            return;
        }

        const auto add_to_level = [&](std::size_t ghost_level, const auto& set)
        {
            A::traverse(set, add_to<dim>(cell_list[ghost_level]));
        };
        for_each_nonempty_level<B>(in.cells,
                                   [&](std::size_t level)
                                   {
                                       prediction_ghosts<B>(in, cells_and_ghosts, subdomain_at, directions, level, add_to_level);
                                   });
        add_neighbour_prediction_ghosts<B>(in, neighbours, directions, subdomain_at, add_to_level);

        auto reference = cell_list.build();
        refine_reference(cell_list, in, reference, proj_cells);
        // the library builds the reference cells once more from the cell
        // list (mr/mesh.hpp:458), which holds the same cells
        reference = cell_list.build();

        emit_levels(cells_and_ghosts, emit);
        emit_levels(reference, emit);
        emit_levels(proj_cells, emit);
    }

    // The sub-mesh of a whole mesh, sequential.
    template <class B, std::size_t dim, class CA, class Emit>
    void mesh_sub_mesh(const SiteInputs<dim, CA>& in, Emit&& emit)
    {
        mesh_sub_mesh<B>(in, neighbours_t<dim, CA>{}, std::forward<Emit>(emit));
    }

    // The scheme ghosts of the cells, the cells_and_ghosts of
    // MRMesh::update_sub_mesh_impl (mr/mesh.hpp:247-254), for each level from
    // 0 to max_level + 1.
    template <class B, std::size_t dim, class CA>
    std::vector<lca_t<B, dim>> scheme_ghosts(const SiteInputs<dim, CA>& in)
    {
        BuilderList<B, dim> cell_list(in.max_level);
        add_scheme_ghosts(cell_list, in.cells, in.max_stencil_radius);
        return cell_list.build();
    }

    // Mesh_base::construct_union (mesh.hpp:1274), at every mesh
    // construction: union_[level - 1] from union_[level], top-down. Emits
    // union_ from max_level down to 0.
    template <class B, std::size_t dim, class CA, class Emit>
    void construct_union(const SiteInputs<dim, CA>& in, Emit&& emit)
    {
        using A                   = Algebra<B>;
        const std::size_t max_lvl = in.max_level;

        std::vector<lca_t<B, dim>> union_levels; // union_[level] is union_levels[max_lvl - level]
        union_levels.reserve(max_lvl + 1);
        union_levels.push_back(A::template builder<dim>(max_lvl).build());
        emit(max_lvl, union_levels.back());
        for (std::size_t level = max_lvl; level >= 1; --level)
        {
            auto lcl = A::template builder<dim>(level - 1);
            A::traverse(A::on(A::union_(in.cells[level], union_levels.back()), level - 1), add_to<dim>(lcl));
            union_levels.push_back(lcl.build());
            emit(level - 1, union_levels.back());
        }
    }

    // Mesh_base::build_pyramid (mesh.hpp:1250) of the domain: run by
    // the constructor from a box and, with MPI, by construct_subdomain at
    // every mesh construction. The library appends the intervals of each
    // projection to a LevelCellArray (add_interval_back), which is what
    // materialize does. Emits the pyramid from level 0 to max_level.
    template <class B, std::size_t dim, class CA, class Emit>
    void build_pyramid(const SiteInputs<dim, CA>& in, Emit&& emit)
    {
        using A               = Algebra<B>;
        const auto& reference = in.domain(); // at max_level
        for (std::size_t level = 0; level <= in.max_level; ++level)
        {
            if (level == in.max_level)
            {
                const auto copy = reference;
                emit(level, copy);
                continue;
            }
            emit(level, A::materialize(A::on(reference, level)));
        }
    }

    // update_cell_array_from_tag (algorithm/graduation.hpp:866-872), at every
    // iteration of an adaptation: the new cells from the old ones and the
    // sets the tags gave (SiteInputs::tag_*). The library appends the
    // intervals to the new CellArray (add_interval_back), which is what
    // materialize does. Emits the new cells from min_level to max_level.
    template <class B, std::size_t dim, class CA, class Emit>
    void cell_array_from_tag(const SiteInputs<dim, CA>& in, Emit&& emit)
    {
        using A = Algebra<B>;
        for (std::size_t level = in.min_level; level <= in.max_level; ++level)
        {
            emit(level,
                 A::materialize(A::difference(A::union_(in.cells[level], in.tag_add_m[level], in.tag_add_p[level]),
                                              A::union_(in.tag_remove_m[level], in.tag_remove_p[level]))));
        }
    }

    // amr::Mesh::update_sub_mesh_impl (amr/mesh.hpp:197), at every
    // construction of an AMR mesh, here on the cells of the MR mesh: scheme
    // ghosts, projection cells (each level built from the one below),
    // prediction cells, their ghosts one level down, then all cells. Emits
    // cells_and_ghosts, proj_cells, pred_cells and all_cells, each from level
    // 0 to max_level + 1.
    template <class B, std::size_t dim, class CA, class Emit>
    void amr_sub_mesh(const SiteInputs<dim, CA>& in, Emit&& emit)
    {
        using A              = Algebra<B>;
        constexpr int pred_r = prediction_stencil_radius;
        const auto& cells    = in.cells;
        const auto& union_   = in.union_;

        BuilderList<B, dim> cl(in.max_level);
        add_scheme_ghosts(cl, cells, ghost_width(in));
        auto cells_and_ghosts = cl.build();

        const std::size_t max_level = cells.max_level();
        const std::size_t min_level = cells.min_level();

        std::vector<lca_t<B, dim>> proj_cells;
        std::vector<lca_t<B, dim>> pred_cells;
        for (std::size_t level = 0; level <= in.max_level + 1; ++level)
        {
            proj_cells.push_back(A::template builder<dim>(level).build());
            pred_cells.push_back(A::template builder<dim>(level).build());
        }

        for (std::size_t level = min_level + 1; level <= max_level; ++level)
        {
            auto lcl = A::template builder<dim>(level);
            A::traverse(A::on(A::difference(A::union_(A::intersection(cells_and_ghosts[level - 1], union_[level - 1]), proj_cells[level - 1]),
                                            cells[level - 1]),
                              level),
                        add_to<dim>(lcl));
            proj_cells[level] = lcl.build();
        }

        for (std::size_t level = min_level + 1; level <= max_level; ++level)
        {
            auto lcl = A::template builder<dim>(level);
            A::traverse(
                A::intersection(A::difference(cells_and_ghosts[level], A::union_(union_[level], cells[level])), A::on(in.domain(), level)),
                add_to<dim>(lcl));
            pred_cells[level] = lcl.build();
        }

        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            auto& lcl = cl[level - 1];
            A::traverse(A::on(A::intersection(pred_cells[level], pred_cells[level]), level - 1),
                        [&](const auto& i, const auto& index_yz)
                        {
                            const auto yz = to_index<dim>(index_yz);
                            samurai::static_nested_loop<dim - 1, -pred_r, pred_r + 1>(
                                [&](const auto& stencil)
                                {
                                    lcl.add_interval(interval_t<dim>{i.start - pred_r, i.end + pred_r}, index_t<dim>(yz + stencil));
                                });
                        });
        }
        cells_and_ghosts = cl.build();

        std::vector<lca_t<B, dim>> all_cells;
        for (std::size_t level = 0; level <= in.max_level + 1; ++level)
        {
            auto lcl = A::template builder<dim>(level);
            if (level >= min_level && level <= max_level)
            {
                A::traverse(A::union_(cells_and_ghosts[level], proj_cells[level]), add_to<dim>(lcl));
            }
            all_cells.push_back(lcl.build());
        }

        emit_levels(cells_and_ghosts, emit);
        emit_levels(proj_cells, emit);
        emit_levels(pred_cells, emit);
        emit_levels(all_cells, emit);
    }

    //////////////////////////////////////////////////////////////////////
    // Periodic ghosts, at every ghost update
    //////////////////////////////////////////////////////////////////////

    // LevelCellArray(level, box) of the cells [min_corner, max_corner), as
    // the library builds a box; through the builder of a backend with level
    // cell arrays of its own.
    template <class B, std::size_t dim>
    auto box_lca(std::size_t level, const std::array<int, dim>& min_corner, const std::array<int, dim>& max_corner)
    {
        using lca_type = typename B::template lca_t<dim>;
        if constexpr (std::is_same_v<lca_type, samurai::LevelCellArray<dim>>)
        {
            xt::xtensor_fixed<int, xt::xshape<dim>> lo;
            xt::xtensor_fixed<int, xt::xshape<dim>> hi;
            for (std::size_t d = 0; d < dim; ++d)
            {
                lo[d] = min_corner[d];
                hi[d] = max_corner[d];
            }
            return samurai::LevelCellArray<dim>(level, samurai::Box<int, dim>(lo, hi));
        }
        else
        {
            auto builder = Algebra<B>::template builder<dim>(level);
            if constexpr (dim == 1)
            {
                builder.add_interval(interval_t<dim>{min_corner[0], max_corner[0]}, index_t<dim>{});
            }
            else
            {
                std::array<int, dim - 1> lo;
                std::array<int, dim - 1> hi;
                for (std::size_t d = 1; d < dim; ++d)
                {
                    lo[d - 1] = min_corner[d];
                    hi[d - 1] = max_corner[d];
                }
                index_t<dim> index;
                const auto rows = [&](const auto& self, std::size_t d) -> void
                {
                    if (d == dim - 1)
                    {
                        builder.add_interval(interval_t<dim>{min_corner[0], max_corner[0]}, index);
                        return;
                    }
                    for (int y = lo[d]; y < hi[d]; ++y)
                    {
                        index[d] = y;
                        self(self, d + 1);
                    }
                };
                rows(rows, 0);
            }
            return builder.build();
        }
    }

    // iterate_over_periodic_ghosts(level, field, ...)
    // (algorithm/update_periodic.hpp:34-180): along each periodic axis, four
    // boxes of ghost_width layers at both ends of the domain, built as
    // LevelCellArrays at every call, the two sets of the ghosts filled by
    // copy from the other end, then the two sets sent to each MPI neighbour
    // and the two received from each (lines 124-171). update_tag_periodic
    // (lines 210-416) evaluates the same sets, the MPI ones twice.
    template <class B, std::size_t dim, class CA, class F>
    void periodic_ghosts_at(const SiteInputs<dim, CA>& in, const neighbours_t<dim, CA>& neighbours, std::size_t level, F&& f)
    {
        using A                   = Algebra<B>;
        const auto& mesh_ref      = in.reference;
        const int gw              = ghost_width(in);
        const std::size_t delta_l = in.max_level - level;
        const auto& min_indices   = in.domain_min;
        const auto& max_indices   = in.domain_max;

        std::array<int, dim> min_corner;
        std::array<int, dim> max_corner;
        samurai::DirectionVector<dim> shift;
        for (std::size_t d = 0; d < dim; ++d)
        {
            min_corner[d] = (min_indices[d] >> delta_l) - gw;
            max_corner[d] = (max_indices[d] >> delta_l) + gw;
            shift[d]      = 0;
        }
        for (std::size_t d = 0; d < dim; ++d)
        {
            if (!in.periodic[d])
            {
                continue;
            }
            shift[d] = (max_indices[d] - min_indices[d]) >> delta_l;

            min_corner[d]        = (min_indices[d] >> delta_l) - gw;
            max_corner[d]        = (min_indices[d] >> delta_l);
            const auto lca_min_m = box_lca<B>(level, min_corner, max_corner);

            min_corner[d]        = (max_indices[d] >> delta_l) - gw;
            max_corner[d]        = (max_indices[d] >> delta_l);
            const auto lca_max_m = box_lca<B>(level, min_corner, max_corner);

            min_corner[d]        = (min_indices[d] >> delta_l);
            max_corner[d]        = (min_indices[d] >> delta_l) + gw;
            const auto lca_min_p = box_lca<B>(level, min_corner, max_corner);

            min_corner[d]        = (max_indices[d] >> delta_l);
            max_corner[d]        = (max_indices[d] >> delta_l) + gw;
            const auto lca_max_p = box_lca<B>(level, min_corner, max_corner);

            f(level,
              A::intersection(A::translate(A::intersection(mesh_ref[level], lca_min_p), shift), A::intersection(mesh_ref[level], lca_max_p)));
            f(level,
              A::intersection(A::translate(A::intersection(mesh_ref[level], lca_max_m), -shift), A::intersection(mesh_ref[level], lca_min_m)));

            for (const auto* neighbour : neighbours)
            {
                const auto& neighbour_ref = neighbour->reference[level];
                f(level,
                  A::intersection(A::translate(A::intersection(mesh_ref[level], lca_min_p), shift),
                                  A::intersection(neighbour_ref, lca_max_p)));
                f(level,
                  A::intersection(A::translate(A::intersection(mesh_ref[level], lca_max_m), -shift),
                                  A::intersection(neighbour_ref, lca_min_m)));
            }
            for (const auto* neighbour : neighbours)
            {
                const auto& neighbour_ref = neighbour->reference[level];
                f(level,
                  A::intersection(A::translate(A::intersection(neighbour_ref, lca_min_p), shift),
                                  A::intersection(mesh_ref[level], lca_max_p)));
                f(level,
                  A::intersection(A::translate(A::intersection(neighbour_ref, lca_max_m), -shift),
                                  A::intersection(mesh_ref[level], lca_min_m)));
            }

            shift[d]      = 0;
            min_corner[d] = (min_indices[d] >> delta_l) - gw;
            max_corner[d] = (max_indices[d] >> delta_l) + gw;
        }
    }

    // The periodic ghosts of one update_ghost_mr: update_ghost_periodic at
    // each level twice from max_level down to 0, then once from 1 up to
    // max_level (update_ghost_mr.hpp:207, 215, 232). Nothing to do on a mesh
    // that is not periodic.
    template <class B, std::size_t dim, class CA, class F>
    void periodic_ghosts(const SiteInputs<dim, CA>& in, const neighbours_t<dim, CA>& neighbours, F&& f)
    {
        for (std::size_t level = in.max_level + 1; level-- > 0;)
        {
            periodic_ghosts_at<B>(in, neighbours, level, f);
            periodic_ghosts_at<B>(in, neighbours, level, f);
        }
        for (std::size_t level = 1; level <= in.max_level; ++level)
        {
            periodic_ghosts_at<B>(in, neighbours, level, f);
        }
    }

    template <class B, std::size_t dim, class CA, class F>
    void periodic_ghosts(const SiteInputs<dim, CA>& in, F&& f)
    {
        periodic_ghosts<B>(in, neighbours_t<dim, CA>{}, std::forward<F>(f));
    }

    //////////////////////////////////////////////////////////////////////
    // Boundary conditions, at every ghost update
    //////////////////////////////////////////////////////////////////////

    // The regions of a boundary condition on the whole boundary
    // (Everywhere::get_region, bc/bc.hpp:327), built once, when the
    // condition is created, as LevelCellArrays: here those of the Cartesian
    // directions, in the order of for_each_cartesian_direction.
    template <class B, std::size_t dim, class CA>
    std::vector<lca_t<B, dim>> bc_regions(const SiteInputs<dim, CA>& in)
    {
        using A = Algebra<B>;
        std::vector<lca_t<B, dim>> regions;
        samurai::for_each_cartesian_direction<dim>(
            [&](const auto& direction)
            {
                regions.push_back(A::materialize(A::difference(in.domain(), A::translate(in.domain(), -direction))));
            });
        return regions;
    }

    // apply_bc_impl (bc/apply_field_bc.hpp:121-124), which update_outer_ghosts
    // (algorithm/update_outer_ghost.hpp:412) calls for each Cartesian
    // direction at every level from min_level to max_level, with one
    // condition (Dirichlet) on the whole boundary. The region is at the
    // finest level: the intersection refines cells[level] up to it, and
    // coarsens the result back, at every call.
    template <class B, std::size_t dim, class CA, class Regions, class F>
    void bc_cells(const SiteInputs<dim, CA>& in, const Regions& regions, F&& f)
    {
        using A = Algebra<B>;
        for (std::size_t level = in.max_level + 1; level-- > in.min_level;)
        {
            std::size_t d = 0;
            samurai::for_each_cartesian_direction<dim>(
                [&](std::size_t axis, const auto&)
                {
                    if (!in.periodic[axis])
                    {
                        f(level, A::on(A::intersection(in.cells[level], regions[d]), level));
                    }
                    ++d;
                });
        }
    }

    // PolynomialExtrapolation<Field, 2>::max_stencil_size_implemented_PE and
    // max_ghost_layers_implemented_PE (bc/polynomial_extrapolation.hpp:14-24).
    inline constexpr int max_stencil_size_PE = 6;
    inline constexpr int max_ghost_layers_PE = max_stencil_size_PE - 1;

    // update_outer_corners_by_polynomial_extrapolation(level, direction,
    // field) (bc/apply_field_bc.hpp:517), which update_outer_ghosts calls
    // for each diagonal direction at every level from min_level to
    // max_level: step 1 through apply_extrapolation_bc_cells (line 295) or
    // mirror_corner_ghost_layer (line 483), step 2 traverses source_at_k
    // once per corner ghost it fills. apply_field_bc_diagonal evaluates the
    // corner_cells of step 2 for the conditions that fill the diagonal
    // directions, which no finite volume condition does.
    template <class B, std::size_t dim, class CA, class F>
    void bc_corners(const SiteInputs<dim, CA>& in, F&& f)
    {
        using A = Algebra<B>;
        if constexpr (dim > 1)
        {
            using direction_t = samurai::DirectionVector<dim>;
            const int gw      = ghost_width(in);
            for (std::size_t level = in.max_level + 1; level-- > in.min_level;)
            {
                samurai::for_each_diagonal_direction<dim>(
                    [&](const auto& diagonal)
                    {
                        if (crosses_periodic_axis(in, diagonal))
                        {
                            return;
                        }
                        const direction_t direction = diagonal;
                        const auto& corner_lca      = in.corner(direction);
                        const auto& cells           = in.cells[level];
                        const auto& reference       = in.reference[level];

                        for (int ghost_layer = 1; ghost_layer <= gw; ++ghost_layer)
                        {
                            auto corner = A::on(corner_lca, level);
                            if (2 * ghost_layer <= max_stencil_size_PE)
                            {
                                if (ghost_layer == 1)
                                {
                                    f(level, A::on(A::intersection(cells, corner), level));
                                }
                                else
                                {
                                    f(level, A::on(A::intersection(A::translate(reference, -ghost_layer * direction), cells, corner), level));
                                }
                            }
                            else
                            {
                                const direction_t to_ghost = (2 * ghost_layer - 1) * direction;
                                auto corner_at_level       = A::on(corner_lca, level);
                                f(level,
                                  A::on(A::intersection(A::translate(corner_at_level, (1 - ghost_layer) * direction),
                                                        cells,
                                                        A::translate(reference, -to_ghost)),
                                        level));
                            }
                        }

                        std::size_t num_nonzero = 0;
                        for (std::size_t d = 0; d < dim; ++d)
                        {
                            num_nonzero += (direction[d] != 0) ? std::size_t{1} : std::size_t{0};
                        }
                        std::size_t num_combos = 1;
                        for (std::size_t p = 1; p < num_nonzero; ++p)
                        {
                            num_combos *= static_cast<std::size_t>(gw);
                        }

                        auto corner_at_level = A::on(corner_lca, level);
                        auto corner_cells    = A::on(A::intersection(corner_at_level, cells), level);
                        for (int k = 1; k <= gw; ++k)
                        {
                            auto source_at_k = A::translate(corner_cells, k * direction);
                            for (std::size_t combo = 0; combo < num_combos; ++combo)
                            {
                                bool is_source  = true;
                                std::size_t tmp = combo;
                                for (std::size_t p = 1; p < num_nonzero; ++p)
                                {
                                    const int g_p = static_cast<int>(tmp % static_cast<std::size_t>(gw));
                                    tmp /= static_cast<std::size_t>(gw);
                                    if (g_p - (k - 1) != 0)
                                    {
                                        is_source = false;
                                    }
                                }
                                if (!is_source)
                                {
                                    f(level, source_at_k);
                                }
                            }
                        }
                    });
            }
        }
    }

    // translated_outer_neighbours<layers> (bc/apply_field_bc.hpp:327-353).
    template <std::size_t layers, class B, class Reference, class Direction, std::size_t... Is>
    auto translated_outer_neighbours(const Reference& reference, const Direction& direction, std::index_sequence<Is...>)
    {
        using A = Algebra<B>;
        if constexpr (sizeof...(Is) == 1)
        {
            return A::translate(reference, -static_cast<int>(layers) * direction);
        }
        else
        {
            return A::intersection(A::translate(reference, -static_cast<int>(layers - Is) * direction)...);
        }
    }

    // Calls f(std::integral_constant<std::size_t, layer>) for the runtime
    // layer in [1, max_ghost_layers_PE], as dispatch_static does.
    template <class F, std::size_t... Ls>
    void dispatch_layer(int layer, F&& f, std::index_sequence<Ls...>)
    {
        ((layer == static_cast<int>(Ls + 1) ? (f(std::integral_constant<std::size_t, Ls + 1>{}), 0) : 0), ...);
    }

    // The domain at `level` less its translate by -direction, the boundary
    // ghosts of the direction that
    // update_further_ghosts_by_polynomial_extrapolation passes to
    // apply_extrapolation_bc_ghosts (bc/apply_field_bc.hpp:784-785).
    template <class B, std::size_t dim, class CA, class Direction>
    auto boundary_ghosts(const SiteInputs<dim, CA>& in, std::size_t level, const Direction& direction)
    {
        using A      = Algebra<B>;
        auto domain2 = A::on(in.domain(), level);
        return A::difference(domain2, A::translate(domain2, -direction));
    }

    // The ghosts apply_extrapolation_bc_ghosts (bc/apply_field_bc.hpp:387-391)
    // fills at `level` from the translated outer neighbours: three nested
    // .on(level).
    template <class B, std::size_t dim, class CA, class Outer, class Ghosts>
    auto extrapolation_bc_ghosts(const SiteInputs<dim, CA>& in,
                                 std::size_t level,
                                 const Outer& translated_outer_nghbr,
                                 const Ghosts& inner_ghosts_location)
    {
        using A                               = Algebra<B>;
        auto potential_inner_cells_and_ghosts = A::on(A::intersection(translated_outer_nghbr, inner_ghosts_location), level);
        auto inner_cells_and_ghosts           = A::on(A::intersection(potential_inner_cells_and_ghosts, in.union_[level]), level);
        return A::on(A::difference(inner_cells_and_ghosts, in.cells[level]), level);
    }

    // update_further_ghosts_by_polynomial_extrapolation(level, direction,
    // field) (bc/apply_field_bc.hpp:715), which update_outer_ghosts calls
    // for each Cartesian direction at every level from min_level to
    // max_level: the ghost layers past those of the condition (Dirichlet:
    // one), through apply_extrapolation_bc_cells (line 295) and
    // apply_extrapolation_bc_ghosts (lines 389-392: three nested .on(level)).
    // Nothing to do when the ghost width is 1.
    template <class B, std::size_t dim, class CA, class F>
    void bc_further_ghosts(const SiteInputs<dim, CA>& in, F&& f)
    {
        using A                                            = Algebra<B>;
        constexpr int ghost_layers_filled_by_bc            = 1;
        constexpr int ghost_layers_filled_by_projection_bc = 1;
        const int gw                                       = ghost_width(in);
        for (std::size_t level = in.max_level + 1; level-- > in.min_level;)
        {
            samurai::for_each_cartesian_direction<dim>(
                [&](std::size_t axis, const auto& direction)
                {
                    if (in.periodic[axis])
                    {
                        return;
                    }
                    const auto& cells     = in.cells[level];
                    const auto& reference = in.reference[level];
                    for (int ghost_layer = ghost_layers_filled_by_bc + 1; ghost_layer <= gw; ++ghost_layer)
                    {
                        auto boundary_cells = domain_boundary<B>(in, level, direction);
                        f(level, A::on(A::intersection(A::translate(reference, -ghost_layer * direction), cells, boundary_cells), level));
                    }
                    for (int ghost_layer = ghost_layers_filled_by_projection_bc + 1; ghost_layer <= gw; ++ghost_layer)
                    {
                        dispatch_layer(
                            ghost_layer,
                            [&](auto layer_ic)
                            {
                                constexpr std::size_t layer = decltype(layer_ic)::value;
                                auto translated_outer_nghbr = translated_outer_neighbours<layer, B>(reference,
                                                                                                    direction,
                                                                                                    std::make_index_sequence<layer>{});
                                f(level,
                                  extrapolation_bc_ghosts<B>(in, level, translated_outer_nghbr, boundary_ghosts<B>(in, level, direction)));
                            },
                            std::make_index_sequence<max_ghost_layers_PE>{});
                    }
                });
        }
    }

    //////////////////////////////////////////////////////////////////////
    // Sets built cell by cell
    //////////////////////////////////////////////////////////////////////

    // The LevelCellArray of the one cell (x, index) at `level`. The library
    // appends the cell to a LevelCellArray it clears after use
    // (add_point_back, clear); a backend has only its builder.
    template <class B, std::size_t dim, class Index>
    auto one_cell(std::size_t level, int x, const Index& index)
    {
        auto builder = Algebra<B>::template builder<dim>(level);
        builder.add_interval(interval_t<dim>{x, x + 1}, index);
        return builder.build();
    }

    // g(one_cell(level, x, index)) for every cell (x, index) of the set at
    // `level`, in order.
    template <class B, std::size_t dim, class Set, class G>
    void for_each_one_cell(const Set& set, std::size_t level, G&& g)
    {
        Algebra<B>::traverse(set,
                             [&](const auto& i, const auto& index)
                             {
                                 const auto yz = to_index<dim>(index);
                                 for (auto ii = i.start; ii < i.end; ++ii)
                                 {
                                     g(one_cell<B, dim>(level, ii, yz));
                                 }
                             });
    }

    // project_bc(proj_level, direction, layer, field) and project_bc(subset,
    // ...) (algorithm/update_outer_ghost.hpp:22-153), with one layer of
    // boundary ghosts: for every projection ghost, the set of its children
    // one level up, or two when there are none. One expression per ghost
    // cell, at every ghost update, for every level from max_level - 1 down to
    // 0 (see outer_ghosts). The projection ghosts are the set of
    // BM_Site_OuterGhosts: traversed here but not counted.
    template <class B, std::size_t dim, class CA, class F>
    void per_cell_children(const SiteInputs<dim, CA>& in, F&& f)
    {
        using A = Algebra<B>;
        for (std::size_t proj_level = in.max_level; proj_level-- > 0;)
        {
            samurai::for_each_cartesian_direction<dim>(
                [&](std::size_t axis, const auto& direction)
                {
                    if (in.periodic[axis])
                    {
                        return;
                    }
                    auto domain = A::on(in.domain(), proj_level);
                    projection_bc_ghosts<B>(
                        in,
                        proj_level,
                        domain,
                        direction,
                        [&](std::size_t, const auto& projection_ghosts)
                        {
                            for_each_one_cell<B, dim>(
                                projection_ghosts,
                                proj_level,
                                [&](const auto& proj_ghost_lca)
                                {
                                    for (auto children_level = proj_level + 1; children_level <= proj_level + 2; ++children_level)
                                    {
                                        if (f(children_level,
                                              A::intersection(A::on(proj_ghost_lca, children_level), in.reference[children_level]))
                                            > 0)
                                        {
                                            break;
                                        }
                                    }
                                });
                        });
                });
        }
    }

    // The ghosts compute_cell_ownership (petsc/compute_cell_ownership.hpp)
    // gives an owner cell by cell, on one MPI rank: the projection ghosts
    // (line 182) and the prediction ghosts (line 236) that have none yet,
    // per level. The library finds them by reading the owner of each cell;
    // here they are computed with the set algebra, outside the timed loop.
    // A cell is owned when it is a cell, a boundary ghost (line 108), a
    // projection ghost with an owned child or a prediction ghost with an
    // owned parent.
    template <class B, std::size_t dim>
    struct OwnershipGhosts
    {
        std::vector<lca_t<B, dim>> projection;
        std::vector<lca_t<B, dim>> prediction;
    };

    template <class B, std::size_t dim, class CA>
    OwnershipGhosts<B, dim> ownership_ghosts(const SiteInputs<dim, CA>& in)
    {
        using A               = Algebra<B>;
        const auto& reference = in.reference;
        const std::size_t min = reference.min_level();
        const std::size_t max = reference.max_level();

        OwnershipGhosts<B, dim> ghosts;
        std::vector<lca_t<B, dim>> owned;
        for (std::size_t level = 0; level <= max + 1; ++level)
        {
            ghosts.projection.push_back(A::template builder<dim>(level).build());
            ghosts.prediction.push_back(A::template builder<dim>(level).build());
            owned.push_back(A::template builder<dim>(level).build());
        }
        for (std::size_t level = min; level <= max; ++level)
        {
            const auto outer_layer = domain_boundary_outer_layer<B>(in, level, ghost_width(in));
            owned[level]           = A::materialize(A::union_(in.cells[level], A::intersection(outer_layer, reference[level])));
        }
        for (std::size_t level = max - 1; level >= min; --level)
        {
            ghosts.projection[level] = A::materialize(
                A::difference(A::on(A::intersection(reference[level], reference[level + 1]), level), owned[level]));
            owned[level] = A::materialize(A::union_(owned[level], A::intersection(ghosts.projection[level], A::on(owned[level + 1], level))));
            if (level == 0)
            {
                break;
            }
        }
        for (std::size_t level = min + 1; level <= max; ++level)
        {
            ghosts.prediction[level] = A::materialize(
                A::difference(A::on(A::intersection(reference[level], reference[level - 1]), level), owned[level]));
            owned[level] = A::materialize(A::union_(owned[level], A::intersection(ghosts.prediction[level], A::on(owned[level - 1], level))));
        }
        return ghosts;
    }

    // compute_cell_ownership (petsc/compute_cell_ownership.hpp:166-247), on
    // one MPI rank: for every projection ghost without an owner, the set of
    // its children (line 182); for every prediction ghost without an owner,
    // the set of its parent (line 236). One expression per ghost cell. Runs
    // once per mesh, when a PETSc solver assembles a system. The outer sets
    // (lines 171 and 225) are counted; `ghosts` stands for the owners the
    // library reads.
    template <class B, std::size_t dim, class CA, class Ghosts, class F>
    void ownership_per_cell(const SiteInputs<dim, CA>& in, const Ghosts& ghosts, F&& f)
    {
        using A               = Algebra<B>;
        const auto& reference = in.reference;
        const std::size_t min = reference.min_level();
        const std::size_t max = reference.max_level();
        for (std::size_t level = max - 1; level >= min; --level)
        {
            f(level, A::on(A::intersection(reference[level], reference[level + 1]), level));
            for_each_one_cell<B, dim>(ghosts.projection[level],
                                      level,
                                      [&](const auto& proj_ghost_lca)
                                      {
                                          f(level + 1, A::intersection(A::on(proj_ghost_lca, level + 1), reference[level + 1]));
                                      });
            if (level == 0)
            {
                break;
            }
        }
        for (std::size_t level = min + 1; level <= max; ++level)
        {
            f(level, A::on(A::intersection(reference[level], reference[level - 1]), level));
            for_each_one_cell<B, dim>(ghosts.prediction[level],
                                      level,
                                      [&](const auto& pred_ghost_lca)
                                      {
                                          f(level - 1, A::intersection(A::on(pred_ghost_lca, level - 1), reference[level - 1]));
                                      });
        }
    }

    //////////////////////////////////////////////////////////////////////
    // Partitions: what each MPI rank runs, sequentially
    //////////////////////////////////////////////////////////////////////

    // These sites take the meshes of all the ranks of a partition
    // (SiteInputs::candidates and neighbours give the other ranks) and run
    // what one rank runs; the benchmarks run them for every rank in turn.

    // Whether a level cell array is empty, as the library asks it of a
    // LevelCellArray; a backend with level cell arrays of its own asks its
    // algebra.
    template <class B, class LCA>
    bool lca_empty(const LCA& lca)
    {
        if constexpr (requires { lca.empty(); })
        {
            return lca.empty();
        }
        else
        {
            return Algebra<B>::empty(lca);
        }
    }

    // Mesh_base::ghost_physical_reach (mesh.hpp:637-647) of a rank in cells
    // of max_level, the level of the subdomains find_neighbourhood compares:
    // 2 R + 4 P + 4 cells of the coarsest level of its cells, R the max
    // stencil radius and P the prediction stencil radius (0 when min_level
    // is max_level); 0 when the rank has no cell.
    template <std::size_t dim, class CA>
    int ghost_reach(const SiteInputs<dim, CA>& in)
    {
        const auto& cells = in.cells;
        if (cells.min_level() > cells.max_level())
        {
            return 0;
        }
        const int prediction_size = (in.min_level != in.max_level) ? prediction_stencil_radius : 0;
        const int reach           = 2 * in.max_stencil_radius + 4 * prediction_size + 4;
        return reach << (in.max_level - cells.min_level());
    }

    // Mesh_base::find_neighbourhood (mesh.hpp:1309-1413) on rank `rank`, its
    // phase 3: for each candidate the bounding boxes kept
    // (SiteInputs::candidates), whether its subdomain meets the subdomain of
    // the rank expanded by the ghost reach of the pair, directly or through
    // a periodic boundary. Passes each set to f(level, set), which returns a
    // positive number when the set is not empty. Returns the neighbours it
    // finds. The exchange of the bounding boxes and of the subdomains is not
    // measured.
    template <class B, std::size_t dim, class CA, class F>
    std::vector<std::size_t> find_neighbourhood(const std::vector<SiteInputs<dim, CA>>& ranks, std::size_t rank, F&& f)
    {
        using A        = Algebra<B>;
        const auto& in = ranks[rank];
        std::array<int, dim> ones;
        ones.fill(1);
        const auto directions = periodic_directions(ones, 0, in.periodic);
        const auto extent     = domain_extent(in);
        const int my_reach    = ghost_reach(in);

        std::vector<std::size_t> neighbours;
        for (const std::size_t candidate : in.candidates)
        {
            const auto& other = ranks[candidate];
            const int width   = std::max({1, my_reach, ghost_reach(other)});
            if (f(in.max_level, A::intersection(A::expand(in.subdomain(), width), other.subdomain())) > 0)
            {
                neighbours.push_back(candidate);
                continue;
            }
            for (const auto& direction : directions)
            {
                samurai::DirectionVector<dim> shift;
                for (std::size_t d = 0; d < dim; ++d)
                {
                    shift[d] = direction[d] * extent[d];
                }
                if (f(in.max_level, A::intersection(A::expand(in.subdomain(), width), A::translate(other.subdomain(), shift))) > 0)
                {
                    neighbours.push_back(candidate);
                    break;
                }
            }
        }
        return neighbours;
    }

    // Mesh_base::construct_subdomain (mesh.hpp:1203-1239) on more than one
    // rank, at every mesh construction: the cells of the rank refined to
    // max_level, row by row, into a LevelCellList, then the build_pyramid
    // (mesh.hpp:1250-1271) of the LevelCellArray it gives. Emits the
    // subdomain from level 0 to max_level.
    template <class B, std::size_t dim, class CA, class Emit>
    void construct_subdomain(const SiteInputs<dim, CA>& in, Emit&& emit)
    {
        using A                     = Algebra<B>;
        const std::size_t max_level = in.max_level;
        const auto& cells           = in.cells;

        auto lcl = A::template builder<dim>(max_level);
        for (std::size_t level = cells.min_level(); level <= cells.max_level(); ++level)
        {
            A::traverse(cells[level],
                        [&](const auto& i, const auto& index_yz)
                        {
                            const std::size_t shift = max_level - level;
                            const interval_t<dim> to_add{i.start << shift, i.end << shift};
                            const auto shift_index = to_index<dim>(index_yz) << shift;
                            samurai::static_nested_loop<dim - 1>(0,
                                                                 1 << shift,
                                                                 1,
                                                                 [&](const auto& stencil)
                                                                 {
                                                                     lcl.add_interval(to_add, index_t<dim>(shift_index + stencil));
                                                                 });
                        });
        }
        const auto reference = lcl.build();
        for (std::size_t level = 0; level <= max_level; ++level)
        {
            if (level == max_level)
            {
                const auto copy = reference;
                emit(level, copy);
                continue;
            }
            emit(level, A::materialize(A::on(reference, level)));
        }
    }

    // outer_subdomain_corner<to_send>(level, field, neighbour)
    // (algorithm/update_subdomain.hpp:20-82): along each axis that is not
    // periodic, the ghosts outside the domain, layer by layer, that the
    // subdomain of the sender owns and the receiver has. The library pushes
    // them to an ArrayOfIntervalAndPoint, sorts it and appends it to a
    // LevelCellArray; here they go to the builder of the backend.
    template <bool to_send, class B, std::size_t dim, class CA>
    auto outer_subdomain_corner(const SiteInputs<dim, CA>& in, const SiteInputs<dim, CA>& neighbour, std::size_t level)
    {
        using A           = Algebra<B>;
        const int gw      = ghost_width(in);
        auto builder      = A::template builder<dim>(level);
        const auto& mesh1 = to_send ? in : neighbour;
        const auto& mesh2 = to_send ? neighbour : in;
        samurai::for_each_cartesian_direction<dim>(
            [&](std::size_t axis, const auto& bdry_direction)
            {
                if (in.periodic[axis])
                {
                    return;
                }
                auto domain = A::lift(in.domain(level));
                for (int layer = 1; layer <= gw; ++layer)
                {
                    auto layer_band   = A::difference(A::translate(domain, layer * bdry_direction),
                                                    A::translate(domain, (layer - 1) * bdry_direction));
                    auto owned_ghosts = A::intersection(mesh1.reference[level],
                                                        layer_band,
                                                        A::translate(mesh1.subdomain(level), layer * bdry_direction));
                    A::traverse(A::intersection(owned_ghosts, mesh2.reference[level]), add_to<dim>(builder));
                }
            });
        return builder.build();
    }

    // exchange_subdomains_merged(level, field) (algorithm/update_ghost_mr.hpp:125-187)
    // for one field: for each neighbour where both ranks have cells at the
    // level, the cells of the interface the rank sends and the outer corners
    // it owns (pack_subdomain_send, lines 63-87), then those it receives
    // (unpack_subdomain_recv, lines 93-118). The library reads the corners
    // with for_each_interval; here they are passed to f.
    template <class B, std::size_t dim, class CA, class F>
    void exchange_subdomains(const SiteInputs<dim, CA>& in, const neighbours_t<dim, CA>& neighbours, std::size_t level, F&& f)
    {
        using A           = Algebra<B>;
        const auto active = [&](const SiteInputs<dim, CA>& neighbour)
        {
            return !lca_empty<B>(in.reference[level]) && !lca_empty<B>(neighbour.reference[level]);
        };
        // the receives are posted first, under the same test
        for (const auto* neighbour : neighbours)
        {
            benchmark::DoNotOptimize(active(*neighbour));
        }
        for (const auto* neighbour : neighbours)
        {
            if (!active(*neighbour))
            {
                continue;
            }
            f(level, A::intersection(in.reference[level], neighbour->reference[level], in.subdomain(level)));
            f(level, outer_subdomain_corner<true, B>(in, *neighbour, level));
        }
        for (const auto* neighbour : neighbours)
        {
            if (!active(*neighbour))
            {
                continue;
            }
            f(level, A::intersection(neighbour->reference[level], in.reference[level], neighbour->subdomain(level)));
            f(level, outer_subdomain_corner<false, B>(in, *neighbour, level));
        }
    }

    // The subdomain exchanges of one update_ghost_mr: twice per level from
    // max_level down to 0, then once per level from 1 up to max_level
    // (update_ghost_mr.hpp:206, 214, 231). update_tag_subdomains
    // (algorithm/update_subdomain.hpp:84-147) evaluates the two interfaces,
    // without the corners, at each level of an adaptation.
    template <class B, std::size_t dim, class CA, class F>
    void subdomain_exchange(const SiteInputs<dim, CA>& in, const neighbours_t<dim, CA>& neighbours, F&& f)
    {
        for (std::size_t level = in.max_level + 1; level-- > 0;)
        {
            exchange_subdomains<B>(in, neighbours, level, f);
            exchange_subdomains<B>(in, neighbours, level, f);
        }
        for (std::size_t level = 1; level <= in.max_level; ++level)
        {
            exchange_subdomains<B>(in, neighbours, level, f);
        }
    }

    // Diffusion::give_to_neighbour (load_balancing/strategies/diffusion.hpp:383-529):
    // the rank cedes to a neighbour the coarse cells of its interface, one
    // ring after the other, until it has given `remaining` cells. Adjacency
    // is computed at min_level: the cells of the rank and of the neighbour
    // projected to it (project_to_level, lines 722-735), the ring of the
    // owned coarse cells next to the front (lines 413-427), and for each
    // coarse cell of the ring a one-cell set intersected with the cells of
    // every level, once to weigh it and once to cede it (lines 448-503);
    // then the front and the owned cells are rebuilt (set_union and
    // set_difference, lines 696-719). Every cell weighs 1, and all the cells
    // of the rank are its own: the library reads both from fields, cell by
    // cell, which is not measured. Where the library builds a CellArray (the
    // one-cell set, the ring, the projections), here it is a level cell array
    // of min_level. Emits the final front, then the cells the rank keeps.
    template <class B, std::size_t dim, class CA, class Emit>
    void diffusion_peel(const SiteInputs<dim, CA>& in, const SiteInputs<dim, CA>& neighbour, double remaining, Emit&& emit)
    {
        using A               = Algebra<B>;
        const std::size_t ref = in.min_level;

        const auto project_to_level = [&](const auto& cells)
        {
            auto builder = A::template builder<dim>(ref);
            for (std::size_t level = cells.min_level(); level <= cells.max_level(); ++level)
            {
                A::traverse(A::on(cells[level], ref), add_to<dim>(builder));
            }
            return builder.build();
        };
        const auto materialize_at_ref = [&](const auto& set)
        {
            auto builder = A::template builder<dim>(ref);
            A::traverse(set, add_to<dim>(builder));
            return builder.build();
        };
        // the cells of every level in a coarse cell
        const auto weigh = [&](const auto& cc)
        {
            double weight = 0.;
            for (std::size_t level = in.min_level; level <= in.max_level; ++level)
            {
                A::traverse(A::on(A::intersection(cc, in.cells[level]), level),
                            [&](const auto& i, const auto&)
                            {
                                weight += static_cast<double>(i.end - i.start);
                            });
            }
            return weight;
        };

        auto owned            = project_to_level(in.cells);
        auto boundary         = project_to_level(neighbour.cells);
        const auto directions = samurai::cartesian_directions<dim>();

        while (remaining > 0.)
        {
            auto ring_builder = A::template builder<dim>(ref);
            for (std::size_t k = 0; k < directions.shape()[0]; ++k)
            {
                const samurai::DirectionVector<dim> d = xt::view(directions, k);
                A::traverse(A::intersection(owned, A::translate(boundary, d)), add_to<dim>(ring_builder));
            }
            const auto ring = ring_builder.build();
            if (A::empty(ring))
            {
                break;
            }

            auto ceded_builder = A::template builder<dim>(ref);
            bool stop          = false;
            A::traverse(ring,
                        [&](const auto& interval, const auto& index)
                        {
                            const auto yz = to_index<dim>(index);
                            for (auto i = interval.start; i < interval.end && !stop; ++i)
                            {
                                const auto cc          = one_cell<B, dim>(ref, i, yz);
                                const double cc_weight = weigh(cc);
                                if (cc_weight > remaining && cc_weight > 0.)
                                {
                                    stop = true;
                                    break;
                                }
                                benchmark::DoNotOptimize(weigh(cc)); // the commit visits the same cells
                                remaining -= cc_weight;
                                ceded_builder.add_interval(interval_t<dim>{i, i + 1}, yz);
                                if (remaining <= 0.)
                                {
                                    stop = true;
                                }
                            }
                        });
            const auto ceded = ceded_builder.build();
            boundary         = materialize_at_ref(A::union_(boundary, ceded));
            owned            = materialize_at_ref(A::difference(owned, ceded));
            if (stop)
            {
                break;
            }
        }
        emit(ref, boundary);
        emit(ref, owned);
    }

    // The load diffusion_peel cedes to each neighbour: a tenth of the cells
    // of the rank. The library takes it from the fluxes of a diffusion on the
    // loads of the ranks, which the equal partitions of the benchmarks would
    // make zero.
    template <class B, std::size_t dim, class CA>
    double peel_load(const SiteInputs<dim, CA>& in)
    {
        double cells = 0.;
        for (std::size_t level = in.cells.min_level(); level <= in.cells.max_level(); ++level)
        {
            Algebra<B>::traverse(in.cells[level],
                                 [&](const auto& i, const auto&)
                                 {
                                     cells += static_cast<double>(i.end - i.start);
                                 });
        }
        return cells / 10.;
    }
} // namespace bench::sites
