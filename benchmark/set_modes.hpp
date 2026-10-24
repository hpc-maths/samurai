// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Ways of consuming a set and operators the call sites of set_call_sites.hpp
// do not cover, measured on the sets of the adapted meshes and written
// against a backend B, as the sites are. Each one comes with the variant it
// is meant to be compared with, which yields the same cells: the library
// writes the first, the second is the other way to write it. Each passes
// every set it evaluates (or materializes) as the sites do.

#pragma once

#include <array>
#include <cstddef>

#include <samurai/stencil.hpp>

#include "set_algebra_backend.hpp"
#include "set_call_sites.hpp"
#include "site_inputs.hpp"

namespace bench::modes
{
    // f(level, set(level)) for every level from `first` to max_level.
    template <std::size_t dim, class CA, class F, class Set>
    void emit_per_level(const SiteInputs<dim, CA>& in, std::size_t first, F& f, Set&& set)
    {
        for (std::size_t level = first; level <= in.max_level; ++level)
        {
            f(level, set(level));
        }
    }

    // intersection(reference, union_) at `level`: the projection cells of
    // MRMesh::update_sub_mesh_impl (mr/mesh.hpp:445).
    template <class B, std::size_t dim, class CA>
    auto proj_cells(const SiteInputs<dim, CA>& in, std::size_t level)
    {
        return Algebra<B>::intersection(in.reference[level], in.union_[level]);
    }

    // The projection cells of each level (proj_cells) through three
    // consumers. traverse_proj_cells only reads them;
    // materialize_proj_cells appends them to a LevelCellArray (to_lca(),
    // LevelCellArray(set)); build_proj_cells adds them, in order, to the
    // builder of the backend (a LevelCellList for samurai), then converts it.
    template <class B, std::size_t dim, class CA, class F>
    void traverse_proj_cells(const SiteInputs<dim, CA>& in, F&& f)
    {
        emit_per_level(in,
                       0,
                       f,
                       [&](std::size_t level)
                       {
                           return proj_cells<B>(in, level);
                       });
    }

    template <class B, std::size_t dim, class CA, class Emit>
    void materialize_proj_cells(const SiteInputs<dim, CA>& in, Emit&& emit)
    {
        emit_per_level(in,
                       0,
                       emit,
                       [&](std::size_t level)
                       {
                           return Algebra<B>::materialize(proj_cells<B>(in, level));
                       });
    }

    template <class B, std::size_t dim, class CA, class Emit>
    void build_proj_cells(const SiteInputs<dim, CA>& in, Emit&& emit)
    {
        using A = Algebra<B>;
        for (std::size_t level = 0; level <= in.max_level; ++level)
        {
            auto builder = A::template builder<dim>(level);
            A::traverse(proj_cells<B>(in, level), sites::add_to<dim>(builder));
            emit(level, builder.build());
        }
    }

    // B::empty, which stops at the first interval. The library asks it in
    // Mesh_base::find_neighbourhood (mesh.hpp:1395, MPI). The projection
    // cells are not empty at most levels: the answer comes early. A cell is
    // never under the union of its level, so cells ∩ union_ is empty: the
    // answer needs the whole traversal.
    template <class B, std::size_t dim, class CA, class F>
    void empty_hit(const SiteInputs<dim, CA>& in, F&& f)
    {
        traverse_proj_cells<B>(in, f);
    }

    template <class B, std::size_t dim, class CA, class F>
    void empty_miss(const SiteInputs<dim, CA>& in, F&& f)
    {
        emit_per_level(in,
                       0,
                       f,
                       [&](std::size_t level)
                       {
                           return Algebra<B>::intersection(in.cells[level], in.union_[level]);
                       });
    }

    // contract(set, width), which samurai writes as an intersection of
    // 2 dim + 1 operands (subset/contraction.hpp:124-181): the boundary
    // cells of every level, domain_boundary(mesh, level) (boundary.hpp:48).
    // The masked variant contracts along x only.
    template <class B, std::size_t dim, class CA, class F>
    void contract(const SiteInputs<dim, CA>& in, F&& f)
    {
        using A = Algebra<B>;
        emit_per_level(in,
                       in.min_level,
                       f,
                       [&](std::size_t level)
                       {
                           return A::difference(in.cells[level], A::contract(in.domain(level), std::size_t{1}));
                       });
    }

    template <class B, std::size_t dim, class CA, class F>
    void contract_masked(const SiteInputs<dim, CA>& in, F&& f)
    {
        using A = Algebra<B>;
        std::array<bool, dim> mask{};
        mask[0] = true;
        emit_per_level(in,
                       in.min_level,
                       f,
                       [&](std::size_t level)
                       {
                           return A::difference(in.cells[level], A::contract(in.domain(level), std::size_t{1}, mask));
                       });
    }

    // The three forms of the expansion of the cells of each level by one
    // cell, which give the same cells: one width (nestedExpand, what the
    // library calls), one width per axis, one width along the masked axes,
    // all axes here. The library calls only the first.
    template <class B, std::size_t dim, class CA, class F>
    void expand_uniform(const SiteInputs<dim, CA>& in, F&& f)
    {
        emit_per_level(in,
                       in.min_level,
                       f,
                       [&](std::size_t level)
                       {
                           return Algebra<B>::expand(in.cells[level], 1);
                       });
    }

    template <class B, std::size_t dim, class CA, class F>
    void expand_axes(const SiteInputs<dim, CA>& in, F&& f)
    {
        std::array<int, dim> widths;
        widths.fill(1);
        emit_per_level(in,
                       in.min_level,
                       f,
                       [&](std::size_t level)
                       {
                           return Algebra<B>::expand(in.cells[level], widths);
                       });
    }

    template <class B, std::size_t dim, class CA, class F>
    void expand_masked(const SiteInputs<dim, CA>& in, F&& f)
    {
        std::array<bool, dim> mask;
        mask.fill(true);
        emit_per_level(in,
                       in.min_level,
                       f,
                       [&](std::size_t level)
                       {
                           return Algebra<B>::expand(in.cells[level], 1, mask);
                       });
    }

    // .on(level) at the level of the result, with an operand at a finer
    // level: apply_bc_impl (bc/apply_field_bc.hpp:121) intersects
    // cells[level] with a region at the finest level. on_mixed_levels uses
    // the domain at the finest level, on_same_level the domain the mesh
    // stores at that level.
    template <class B, std::size_t dim, class CA, class F>
    void on_mixed_levels(const SiteInputs<dim, CA>& in, F&& f)
    {
        using A = Algebra<B>;
        emit_per_level(in,
                       in.min_level,
                       f,
                       [&](std::size_t level)
                       {
                           return A::on(A::intersection(in.cells[level], in.domain()), level);
                       });
    }

    template <class B, std::size_t dim, class CA, class F>
    void on_same_level(const SiteInputs<dim, CA>& in, F&& f)
    {
        using A = Algebra<B>;
        emit_per_level(in,
                       in.min_level,
                       f,
                       [&](std::size_t level)
                       {
                           return A::on(A::intersection(in.cells[level], in.domain(level)), level);
                       });
    }

    // A chain of .on(level) at the level of its operands, as
    // apply_extrapolation_bc_ghosts writes it (bc/apply_field_bc.hpp:389-392),
    // here with one ghost layer so that it runs on every mesh, against the
    // same set as one expression.
    template <class B, std::size_t dim, class CA, class F>
    void nested_on(const SiteInputs<dim, CA>& in, F&& f)
    {
        using A = Algebra<B>;
        for (std::size_t level = in.min_level; level <= in.max_level; ++level)
        {
            samurai::for_each_cartesian_direction<dim>(
                [&](const auto& direction)
                {
                    f(level,
                      sites::extrapolation_bc_ghosts<B>(in,
                                                        level,
                                                        A::translate(in.reference[level], -direction),
                                                        sites::boundary_ghosts<B>(in, level, direction)));
                });
        }
    }

    template <class B, std::size_t dim, class CA, class F>
    void flat_on(const SiteInputs<dim, CA>& in, F&& f)
    {
        using A = Algebra<B>;
        for (std::size_t level = in.min_level; level <= in.max_level; ++level)
        {
            samurai::for_each_cartesian_direction<dim>(
                [&](const auto& direction)
                {
                    auto boundary_ghosts = sites::boundary_ghosts<B>(in, level, direction);
                    f(level,
                      A::difference(A::intersection(A::translate(in.reference[level], -direction), boundary_ghosts, in.union_[level]),
                                    in.cells[level]));
                });
        }
    }

    // intersection(X, X).on(level - 1), as amr::Mesh::update_sub_mesh_impl
    // writes it (amr/mesh.hpp:258), against self(X).on(level - 1), on the
    // reference cells.
    template <class B, std::size_t dim, class CA, class F>
    void intersection_self_on(const SiteInputs<dim, CA>& in, F&& f)
    {
        using A = Algebra<B>;
        for (std::size_t level = in.min_level; level <= in.max_level; ++level)
        {
            f(level - 1, A::on(A::intersection(in.reference[level], in.reference[level]), level - 1));
        }
    }

    template <class B, std::size_t dim, class CA, class F>
    void self_on(const SiteInputs<dim, CA>& in, F&& f)
    {
        using A = Algebra<B>;
        for (std::size_t level = in.min_level; level <= in.max_level; ++level)
        {
            f(level - 1, A::on(in.reference[level], level - 1));
        }
    }
} // namespace bench::modes
