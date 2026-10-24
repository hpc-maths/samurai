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
#include <utility>
#include <vector>

#include <xtensor/views/xview.hpp>

#include <samurai/stencil.hpp>

#include "set_algebra_backend.hpp"
#include "site_inputs.hpp"

namespace bench::sites
{
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
    // two levels below. cag gives the cells and ghosts of the level.
    template <class B, std::size_t dim, class CA, class Cag, class SubdomainAt, class G>
    void prediction_ghosts(const SiteInputs<dim, CA>& in, const Cag& cag, SubdomainAt&& subdomain_at, std::size_t level, G&& g)
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
        add(A::expand(A::on(in.cells[level], level - 2), pred_r),
            A::intersection(A::expand(A::on(cag[level], level - 1), pred_r), A::expand(subdomain_at(level - 1), pred_r)));
    }

    // MRMesh::update_sub_mesh_impl (mr/mesh.hpp:357-382, 444): the ghosts the
    // prediction needs one and two levels below each cell, then the
    // projection cells. The library visits the levels that have cells, from
    // min_level >= 2 here.
    template <class B, std::size_t dim, class CA, class SubdomainAt, class F>
    void mesh_prediction_ghosts(const SiteInputs<dim, CA>& in, SubdomainAt&& subdomain_at, F&& f)
    {
        using A           = Algebra<B>;
        const auto& cells = in.cells;
        for (std::size_t level = cells.min_level(); level <= cells.max_level(); ++level)
        {
            if (level < 2 || A::empty(cells[level]))
            {
                continue;
            }
            prediction_ghosts<B>(in, in.cells_and_ghosts, subdomain_at, level, f);
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

    // The ghosts of the Dirichlet condition of the adapted meshes, and
    // max_coarse_layer of update_outer_ghosts: one layer.
    inline constexpr int n_bc_ghosts = 1;

    // The projection ghosts project_bc(proj_level, direction, layer = 1,
    // field) fills (algorithm/update_outer_ghost.hpp:23-61): those outside
    // the domain. Passes the set to g(proj_level, set).
    template <class B, std::size_t dim, class CA, class Domain, class G>
    void projection_bc_ghosts(const SiteInputs<dim, CA>& in,
                              std::size_t proj_level,
                              const Domain& domain,
                              const samurai::DirectionVector<dim>& direction,
                              G&& g)
    {
        using A             = Algebra<B>;
        constexpr int layer = 1;
        const auto& inner   = in.union_[proj_level];
        auto outside_layer  = A::difference(A::translate(inner, layer * direction), A::translate(domain, (layer - 1) * direction));
        g(proj_level, A::on(A::intersection(outside_layer, in.reference[proj_level]), proj_level));
    }

    // update_outer_ghosts(level, field) (algorithm/update_outer_ghost.hpp:341)
    // without the boundary condition itself: project_corner_below,
    // project_bc and predict_bc, with one layer of boundary ghosts
    // (Dirichlet). predict_bc builds a LevelCellArray
    // (domain_boundary_outer_layer) on every call. update_ghost_mr calls it
    // for every level from max_level down to 0 (update_ghost_mr.hpp:204-208):
    // project_bc runs below the min level too.
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
                [&](const auto& direction)
                {
                    if (level < in.max_level)
                    {
                        projection_bc_ghosts<B>(in, level, domain_at(level), direction, f);
                    }
                    if (level < in.max_level && level >= in.min_level)
                    {
                        // predict_bc(level + 1, direction, field) (update_outer_ghost.hpp:197-230)
                        const auto bc_ghosts = domain_boundary_outer_layer<B>(in, level, direction, n_bc_ghosts);
                        f(level + 1, A::on(A::intersection(bc_ghosts, in.reference[level + 1]), level + 1));
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
    // side (lines 125-223 and 239-323).
    template <class B, std::size_t dim, class CA, class F>
    void interfaces(const SiteInputs<dim, CA>& in, F&& f)
    {
        using A               = Algebra<B>;
        const auto& cells     = in.cells;
        const auto directions = samurai::positive_cartesian_directions<dim>();
        for (std::size_t d = 0; d < dim; ++d)
        {
            const samurai::DirectionVector<dim> direction = xt::view(directions, d);
            for (std::size_t level = cells.min_level(); level <= cells.max_level(); ++level)
            {
                const auto& own = cells[level];
                // same level
                f(level, A::intersection(own, A::translate(own, -direction)));
                if (level >= cells.max_level())
                {
                    continue;
                }
                const auto& fine = cells[level + 1];
                // level jumps, in the direction (sign = -1: the fine cells
                // translated by -direction) then in the opposite one
                for (const int sign : {-1, 1})
                {
                    f(level + 1, A::on(A::intersection(own, A::translate(fine, sign * direction)), level + 1));
                }
            }
        }
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
    // cell list, in no order), the prediction ghosts (prediction_ghosts, added
    // to the same cell list), then the reference cells, refined level by
    // level where the union covers them, and the projection cells: one step
    // each, in the order of the library. Emits cells_and_ghosts, then
    // reference, then proj_cells, each from level 0 to max_level + 1.
    template <class B, std::size_t dim, class CA, class Emit>
    void mesh_sub_mesh(const SiteInputs<dim, CA>& in, Emit&& emit)
    {
        using A = Algebra<B>;

        BuilderList<B, dim> cell_list(in.max_level);
        add_scheme_ghosts(cell_list, in.cells, in.max_stencil_radius);
        const auto cells_and_ghosts = cell_list.build();

        const auto subdomain_at = [&](std::size_t level)
        {
            return A::on(in.subdomain(), level);
        };

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
                                       prediction_ghosts<B>(in, cells_and_ghosts, subdomain_at, level, add_to_level);
                                   });

        auto reference = cell_list.build();
        refine_reference(cell_list, in, reference, proj_cells);
        // the library builds the reference cells once more from the cell
        // list (mr/mesh.hpp:458), which holds the same cells
        reference = cell_list.build();

        emit_levels(cells_and_ghosts, emit);
        emit_levels(reference, emit);
        emit_levels(proj_cells, emit);
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
            for (const auto& region : regions)
            {
                f(level, A::on(A::intersection(in.cells[level], region), level));
            }
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
                [&](const auto& direction)
                {
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
                [&](const auto& direction)
                {
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
} // namespace bench::sites
