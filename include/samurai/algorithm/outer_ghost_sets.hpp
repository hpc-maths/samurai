// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cstddef>
#include <vector>

#include "../algorithm.hpp"
#include "../bc/extrapolation_sets.hpp"
#include "../boundary.hpp"
#include "../stencil.hpp"
#include "update_periodic.hpp" // has_ghost_tables

namespace samurai::detail
{
    // Calls f with the tables of the ghosts outside the (box) domain at `level`
    // when the mesh holds them, and returns what f returns: true when the
    // tables did the work.
    template <class Mesh, class Func>
    bool with_outer_ghost_tables(const Mesh& mesh, std::size_t level, Func&& f)
    {
        if constexpr (has_ghost_tables<Mesh>)
        {
            const auto& outer = mesh.ghost_tables().outer;
            return level < outer.size() && f(outer[level]);
        }
        else
        {
            return false;
        }
    }

    // Position of `direction` in for_each_cartesian_direction.
    template <std::size_t dim>
    std::size_t cartesian_direction_index(const DirectionVector<dim>& direction)
    {
        std::size_t index = 0;
        std::size_t found = 0;
        for_each_cartesian_direction<dim>(
            [&](auto, const auto& d)
            {
                if (d == direction)
                {
                    found = index;
                }
                ++index;
            });
        return found;
    }

    //////////////////////////////////////////////////////////////////////
    // Field projection of the B.C. (project_bc)
    //////////////////////////////////////////////////////////////////////

    // Calls f with the ghosts at `proj_level` of the layer `layer` outside a box
    // domain in `direction`, facing the union: the ghosts project_bc fills.
    template <class Mesh, class Func>
    void with_bc_projection_ghosts(const Mesh& mesh, std::size_t proj_level, const DirectionVector<Mesh::dim>& direction, int layer, Func&& f)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        const auto& domain = mesh.domain(proj_level);
        const auto& inner  = mesh.get_union()[proj_level];

        // only the layer `layer`: the closer ones are removed with the domain translated by (layer - 1) * direction
        auto outside_layer     = difference(translate(inner, layer * direction), translate(domain, (layer - 1) * direction));
        auto projection_ghosts = intersection(outside_layer, mesh[mesh_id_t::reference][proj_level]).on(proj_level);
        f(projection_ghosts);
    }

    // For each ghost of `projection_ghosts`: visitor.ghost(ii, index), then
    // visitor.child(children_level, ii_child, index_child) for each of its
    // children in the reference, at proj_level + 1 or, when it has none there, at
    // proj_level + 2, then visitor.end(n_children).
    template <class Mesh, class Subset, class Visitor>
    void visit_bc_projection(const Mesh& mesh, Subset& projection_ghosts, std::size_t proj_level, Visitor& visitor)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        using lca_t     = typename Mesh::lca_type;

        lca_t ghost_lca(proj_level, mesh.origin_point(), mesh.scaling_factor());
        projection_ghosts(
            [&](const auto& i, const auto& index)
            {
                for (auto ii = i.start; ii < i.end; ++ii)
                {
                    visitor.ghost(ii, index);
                    ghost_lca.add_point_back(ii, index); // only the current ghost
                    int n_children = 0;
                    for (auto children_level = proj_level + 1; children_level <= proj_level + 2; ++children_level)
                    {
                        auto children = intersection(self(ghost_lca).on(children_level), mesh[mesh_id_t::reference][children_level]);
                        children(
                            [&](const auto& i_child, const auto& index_child)
                            {
                                for (auto ii_child = i_child.start; ii_child < i_child.end; ++ii_child)
                                {
                                    visitor.child(children_level, ii_child, index_child);
                                    ++n_children;
                                }
                            });
                        if (n_children > 0)
                        {
                            break;
                        }
                    }
                    visitor.end(n_children);
                    ghost_lca.clear();
                }
            });
    }

    //////////////////////////////////////////////////////////////////////
    // Prediction of the B.C. (predict_bc)
    //////////////////////////////////////////////////////////////////////

    // Calls f with the ghosts at `pred_level` below the layer `layer` of B.C.
    // ghosts at pred_level - 1 outside a box domain in `direction`. The layers
    // 1 to n together are the ghosts predict_bc fills with n B.C. ghosts.
    template <class Mesh, class Func>
    void with_bc_prediction_layer(const Mesh& mesh, std::size_t pred_level, const DirectionVector<Mesh::dim>& direction, int layer, Func&& f)
    {
        using mesh_id_t   = typename Mesh::mesh_id_t;
        const auto upto   = domain_boundary_outer_layer(mesh, pred_level - 1, direction, layer);
        const auto before = domain_boundary_outer_layer(mesh, pred_level - 1, direction, layer - 1);
        auto ghosts       = intersection(difference(upto, before), mesh[mesh_id_t::reference][pred_level]).on(pred_level);
        f(ghosts);
    }

    //////////////////////////////////////////////////////////////////////
    // Copy of the corner ghosts below (project_corner_below)
    //////////////////////////////////////////////////////////////////////

    // For each corner ghost at level - 1 and level - 2 below the corner ghosts
    // of `level` in `direction`: visitor.copy(proj_level, ii, index, i_child,
    // index_child) when its corner-most child at `level` exists.
    template <class Mesh, class Visitor>
    void visit_corner_copies(const Mesh& mesh, std::size_t level, const DirectionVector<Mesh::dim>& direction, Visitor& visitor)
    {
        using mesh_id_t                  = typename Mesh::mesh_id_t;
        using coord_t                    = typename Mesh::lca_type::all_coord_type;
        static constexpr std::size_t dim = Mesh::dim;

        if (level == 0)
        {
            return;
        }

        for (std::size_t delta_l = 1; delta_l <= 2; ++delta_l) // lower level (1 or 2)
        {
            auto proj_level = level - delta_l;

            // The source is the corner ghost at `level`, whether the corner has cells at `level`
            // (ghost filled by polynomial extrapolation) or not (ghost filled by the projection
            // from level+1, which has already been done since the levels are processed from fine
            // to coarse). This cascade fills the corner ghosts at every level below the corner
            // cells, even when max_level - min_level > 2.
            auto fine_inner_corner = self(mesh.corner(direction)).on(level);
            auto fine_outer_corner = intersection(translate(fine_inner_corner, direction), mesh[mesh_id_t::reference][level]);
            auto projection_ghost  = intersection(fine_outer_corner.on(proj_level), mesh[mesh_id_t::reference][proj_level]);

            // Each coarse corner ghost is copied from its corner-most child at `level`, the cell at
            // the reconstructed fixed-parity position (ii << delta_l) + offset. projection_ghost keeps
            // a coarse ghost as soon as one of its children lies in the reference array, but on an
            // adapted mesh whose refinement reaches the domain boundary the corner-most child may be
            // absent while another child is present. Reading the absent child would index the cell
            // array with the not-found offset (out of bounds). The child is therefore looked up per
            // cell and copied only when it exists; where it exists it is the single relevant child, so
            // the value matches the former unconditional read.
            const auto& ref_level = mesh[mesh_id_t::reference][level];
            projection_ghost(
                [&](const auto& i, const auto& index)
                {
                    using index_t = std::decay_t<decltype(index)>;

                    index_t index_child = index << delta_l;
                    for (std::size_t d = 0; d < dim - 1; ++d)
                    {
                        index_child[d] += direction[d + 1] == -1 ? ((1 << delta_l) - 1) : 0;
                    }

                    coord_t child_coord;
                    auto i_child = (i.start << delta_l) + (direction[0] == -1 ? ((1 << delta_l) - 1) : 0);
                    for (auto ii = i.start; ii < i.end; ++ii, i_child += (1 << delta_l))
                    {
                        child_coord[0] = i_child;
                        for (std::size_t d = 1; d < dim; ++d)
                        {
                            child_coord[d] = index_child[d - 1];
                        }

                        if (find(ref_level, child_coord) >= 0)
                        {
                            visitor.copy(proj_level, ii, index, i_child, index_child);
                        }
                    }
                });
            if (proj_level == 0)
            {
                break;
            }
        }
    }

    //////////////////////////////////////////////////////////////////////
    // Tables
    //////////////////////////////////////////////////////////////////////

    // dest[k] = mean of child[start[k] .. start[k + 1]), 0 when there is none.
    struct BCProjectionTable
    {
        std::vector<std::size_t> dest;
        std::vector<std::size_t> start{0};
        std::vector<std::size_t> child;
    };

    // dest[k] = src[k], cell by cell.
    struct CopyTable
    {
        std::vector<std::size_t> dest;
        std::vector<std::size_t> src;
    };

    // update_outer_corners_by_polynomial_extrapolation in one diagonal direction.
    struct CornerExtrapolation
    {
        bool recorded = false;
        ExtrapolationSequence sequence;
    };

    // The ghosts outside a box domain at one level that update_outer_ghosts
    // fills from other ghosts or by polynomial extrapolation.
    struct OuterGhostTables
    {
        std::vector<std::vector<BCProjectionTable>> projection;                  // [direction][layer - 1]: project_bc to this level
        std::vector<std::vector<CopyTable>> prediction;                          // [direction][layer - 1]: predict_bc at this level
        CopyTable corners;                                                       // project_outer_corners_below from this level
        std::vector<LazyExtrapolationLayers> extrapolation_on_cells;             // [direction], each layer recorded on first use
        std::vector<std::vector<ExtrapolationSequence>> extrapolation_on_ghosts; // [direction][layer - 1], from the layer 2
        std::vector<CornerExtrapolation> corner_extrapolation;                   // [direction_code(diagonal direction)]
    };

    template <class Mesh>
    struct BCProjectionRecorder
    {
        BCProjectionTable& table;
        std::size_t level;
        const Mesh& mesh;

        void ghost(auto ii, const auto& index)
        {
            table.dest.push_back(static_cast<std::size_t>(memory_offset(mesh, {level, ii, index})));
        }

        void child(std::size_t children_level, auto ii_child, const auto& index_child)
        {
            table.child.push_back(static_cast<std::size_t>(memory_offset(mesh, {children_level, ii_child, index_child})));
        }

        void end(int)
        {
            table.start.push_back(table.child.size());
        }
    };

    template <class Mesh>
    struct CornerCopyRecorder
    {
        CopyTable& table;
        std::size_t level;
        const Mesh& mesh;

        void copy(std::size_t proj_level, auto ii, const auto& index, auto i_child, const auto& index_child)
        {
            table.dest.push_back(static_cast<std::size_t>(memory_offset(mesh, {proj_level, ii, index})));
            table.src.push_back(static_cast<std::size_t>(memory_offset(mesh, {level, i_child, index_child})));
        }
    };

    template <class Mesh>
    OuterGhostTables make_outer_ghost_tables(std::size_t level, const Mesh& mesh)
    {
        constexpr std::size_t dim = Mesh::dim;
        OuterGhostTables tables;
        auto offset = [&](std::size_t l, auto x, const auto& index)
        {
            return static_cast<std::size_t>(memory_offset(mesh, {l, x, index}));
        };

        // the layers update_outer_ghosts asks for when the B.C. uses the whole stencil
        const int n_layers           = mesh.max_stencil_radius();
        const int n_projected_layers = (n_layers + 1) / 2;

        // update_further_ghosts_by_polynomial_extrapolation, when the ghost width is one it can fill
        const int ghost_width           = static_cast<int>(mesh.ghost_width());
        const bool record_extrapolation = level >= mesh.min_level() && level <= mesh.max_level()
                                       && ghost_width <= static_cast<int>(max_ghost_layers_implemented_PE);

        for_each_cartesian_direction<dim>(
            [&](auto d, const auto& direction)
            {
                auto& projection = tables.projection.emplace_back();
                auto& prediction = tables.prediction.emplace_back();
                auto& on_cells   = tables.extrapolation_on_cells.emplace_back();
                auto& on_ghosts  = tables.extrapolation_on_ghosts.emplace_back();
                if (mesh.is_periodic(d))
                {
                    return;
                }
                if (record_extrapolation)
                {
                    on_cells = LazyExtrapolationLayers(static_cast<std::size_t>(ghost_width));
                    on_ghosts.resize(static_cast<std::size_t>(ghost_width));
                    for (int layer = 2; layer <= ghost_width; ++layer)
                    {
                        ExtrapolationRecorder from_ghosts{on_ghosts[static_cast<std::size_t>(layer - 1)]};
                        visit_extrapolation_on_ghosts(mesh, level, direction, layer, layer, from_ghosts);
                    }
                }
                // the levels where update_outer_ghosts calls project_bc and predict_bc
                if (level < mesh.max_level() && level + 1 >= mesh.min_level())
                {
                    for (int layer = 1; layer <= n_projected_layers; ++layer)
                    {
                        BCProjectionRecorder<Mesh> recorder{projection.emplace_back(), level, mesh};
                        with_bc_projection_ghosts(mesh,
                                                  level,
                                                  direction,
                                                  layer,
                                                  [&](auto& ghosts)
                                                  {
                                                      visit_bc_projection(mesh, ghosts, level, recorder);
                                                  });
                    }
                }
                if (level > mesh.min_level() && level <= mesh.max_level())
                {
                    for (int layer = 1; layer <= n_layers; ++layer)
                    {
                        auto& table = prediction.emplace_back();
                        with_bc_prediction_layer(mesh,
                                                 level,
                                                 direction,
                                                 layer,
                                                 [&](auto& ghosts)
                                                 {
                                                     ghosts(
                                                         [&](const auto& i, const auto& index)
                                                         {
                                                             for (auto ii = i.start; ii < i.end; ++ii)
                                                             {
                                                                 table.dest.push_back(offset(level, ii, index));
                                                                 table.src.push_back(offset(level - 1, ii >> 1, index >> 1));
                                                             }
                                                         });
                                                 });
                    }
                }
            });

        if constexpr (dim > 1)
        {
            if (level < mesh.min_level() || level > mesh.max_level())
            {
                return tables;
            }
            CornerCopyRecorder<Mesh> recorder{tables.corners, level, mesh};
            std::size_t n_directions = 1;
            for (std::size_t d = 0; d < dim; ++d)
            {
                n_directions *= 3;
            }
            tables.corner_extrapolation.resize(n_directions);
            for_each_diagonal_direction<dim>(
                [&](const auto& direction)
                {
                    for (std::size_t d = 0; d < dim; ++d)
                    {
                        if (direction[d] != 0 && mesh.is_periodic(d))
                        {
                            return;
                        }
                    }
                    auto& corner    = tables.corner_extrapolation[direction_code(direction)];
                    corner.recorded = true;
                    ExtrapolationRecorder extrapolation{corner.sequence};
                    visit_corner_extrapolation(mesh, level, direction, extrapolation);
                    visit_corner_copies(mesh, level, direction, recorder);
                });
        }
        return tables;
    }

    // Same arithmetic as project_bc: the sum of the children from 0, divided by their number.
    template <class Field>
    void run_bc_projection(Field& field, const BCProjectionTable& table)
    {
        using value_t            = typename Field::value_type;
        constexpr std::size_t nc = Field::n_comp;
        auto* data               = field.data();
        for (std::size_t k = 0; k < table.dest.size(); ++k)
        {
            const auto n = static_cast<int>(table.start[k + 1] - table.start[k]);
            for (std::size_t c = 0; c < nc; ++c)
            {
                value_t sum = 0;
                for (std::size_t j = table.start[k]; j < table.start[k + 1]; ++j)
                {
                    sum += data[table.child[j] * nc + c];
                }
                if (n > 0)
                {
                    sum /= n;
                }
                data[table.dest[k] * nc + c] = sum;
            }
        }
    }

    template <class Field>
    void run_copies(Field& field, const CopyTable& table)
    {
        constexpr std::size_t nc = Field::n_comp;
        auto* data               = field.data();
        for (std::size_t k = 0; k < table.dest.size(); ++k)
        {
            for (std::size_t c = 0; c < nc; ++c)
            {
                data[table.dest[k] * nc + c] = data[table.src[k] * nc + c];
            }
        }
    }
}
