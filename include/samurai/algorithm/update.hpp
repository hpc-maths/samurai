// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include "../subset/subset_op.hpp"
#include "../mr/operators.hpp"
#include "utils.hpp"
#include "../numeric/projection.hpp"
#include "../numeric/prediction.hpp"

namespace samurai
{
    template<class Field, class... Fields, class Func>
    void update_ghost(Func&& update_bc_for_level, Field& field, Fields&... fields)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;
        constexpr std::size_t pred_order = Field::mesh_t::config::prediction_order;

        auto mesh = field.mesh();
        std::size_t min_level = mesh.min_level();
        std::size_t max_level = mesh.max_level();

        for (std::size_t level = max_level; level >= 1; --level)
            {
            auto set_at_levelm1 = intersection(mesh[mesh_id_t::proj_cells][level],
                                               mesh[mesh_id_t::reference][level-1])
                                 .on(level - 1);
            set_at_levelm1.apply_op(variadic_projection(field, fields...));
        }

        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            auto set_at_level = intersection(mesh[mesh_id_t::pred_cells][level],
                                             mesh[mesh_id_t::reference][level-1])
                               .on(level);
            update_bc_for_level(level-1, field, fields...);
            set_at_level.apply_op(variadic_prediction<pred_order, false>(field, fields...));
            update_bc_for_level(level, field, fields...);
        }
    }

    template<class Field, class Func>
    void update_ghost_mro(Field& field, Func&& update_bc_for_level)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;
        constexpr std::size_t pred_order = Field::mesh_t::config::prediction_order;
        auto mesh = field.mesh();

        std::size_t min_level = mesh.min_level();
        std::size_t max_level = mesh.max_level();

        for (std::size_t level = max_level; level >= 1; --level)
            {
            auto set_at_levelm1 = intersection(mesh[mesh_id_t::reference][level],
                                               mesh[mesh_id_t::proj_cells][level-1])
                                 .on(level - 1);
            set_at_levelm1.apply_op(projection(field));
        }

        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            // We eliminate the overleaves from the computation since they
            // are done separately
            // auto expr = difference(intersection(difference(mesh[mesh_id_t::all_cells][level],
            //                                                union_(mesh[mesh_id_t::cells][level],
            //                                                       mesh[mesh_id_t::proj_cells][level])),
            //                                     mesh.domain()),
            //                        difference(mesh[mesh_id_t::overleaves][level],
            //                                   union_(mesh[mesh_id_t::union_cells][level],
            //                                          mesh[mesh_id_t::cells_and_ghosts][level])))
            //             .on(level);

            auto expr = intersection(difference(mesh[mesh_id_t::all_cells][level],
                                                           union_(mesh[mesh_id_t::cells][level],
                                                                  mesh[mesh_id_t::proj_cells][level])),
                                                mesh.domain())
                        .on(level);
            update_bc_for_level(field, level-1);
            expr.apply_op(prediction<pred_order, false>(field));
            update_bc_for_level(field, level);
        }
    }

    template<class Field, class Func>
    void update_ghost_mr(Field& field, Func&& update_bc_for_level)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;

        auto mesh = field.mesh();
        std::size_t min_level = mesh.min_level();
        std::size_t max_level = mesh.max_level();

        for (std::size_t level = max_level; level >= 1; --level)
            {
            auto set_at_levelm1 = intersection(mesh[mesh_id_t::reference][level],
                                               mesh[mesh_id_t::proj_cells][level-1])
                                 .on(level - 1);
            set_at_levelm1.apply_op(projection(field));
        }

        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            auto expr = intersection(difference(mesh[mesh_id_t::all_cells][level],
                                                union_(mesh[mesh_id_t::cells][level],
                                                       mesh[mesh_id_t::proj_cells][level])),
                                     mesh.domain())
                        .on(level);

            update_bc_for_level(field, level-1);
            expr.apply_op(prediction<1, false>(field));
            update_bc_for_level(field, level);
        }
    }

    template<class Field, class Func>
    void update_overleaves_mr(Field& field, Func&& update_bc_for_level)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;

        auto mesh = field.mesh();
        std::size_t min_level = mesh.min_level();
        std::size_t max_level = mesh.max_level();

        for (std::size_t level = min_level + 1; level <= max_level; ++level)
        {
            // These are the overleaves which are nothing else
            // because when this procedure is called all the rest
            // should be already with the right value.
            auto overleaves_to_predict = difference(difference(mesh[mesh_id_t::overleaves][level],
                                                               mesh[mesh_id_t::cells_and_ghosts][level]),
                                                    mesh[mesh_id_t::proj_cells][level]);

            overleaves_to_predict.apply_op(prediction<1, false>(field));
            update_bc_for_level(field, level);
        }
    }

    namespace detail
    {
        template<class Mesh, class Field>
        void update_fields(Mesh& new_mesh, Field& field)
        {
            using mesh_id_t = typename Mesh::mesh_id_t;
            constexpr std::size_t pred_order = Field::mesh_t::config::prediction_order;

            Field new_field("new_f", new_mesh);
            new_field.fill(0);

            auto mesh = field.mesh();

            auto min_level = mesh.min_level();
            auto max_level = mesh.max_level();

            for(std::size_t level = min_level; level <= max_level; ++level)
            {
                auto set = intersection(mesh[mesh_id_t::cells][level],
                                        new_mesh[mesh_id_t::cells][level]);
                set.apply_op(copy(new_field, field));
            }

            for(std::size_t level = min_level + 1; level <= max_level; ++level)
            {
                auto set_coarsen = intersection(mesh[mesh_id_t::cells][level],
                                                new_mesh[mesh_id_t::cells][level-1])
                                  .on(level - 1);
                set_coarsen.apply_op(projection(new_field, field));

                auto set_refine = intersection(new_mesh[mesh_id_t::cells][level],
                                               mesh[mesh_id_t::cells][level-1])
                                 .on(level - 1);
                set_refine.apply_op(prediction<pred_order, true>(new_field, field));
            }

            std::swap(field.array(), new_field.array());
        }

        template<class Mesh, class Field, class... Fields>
        void update_fields(Mesh& new_mesh, Field& field, Fields&... fields)
        {
            update_fields(new_mesh, field);
            update_fields(new_mesh, fields...);
        }

    }
    template<class Tag, class... Fields>
    bool update_field(Tag& tag, Fields&... fields)
    {
        static constexpr std::size_t dim = Tag::dim;
        using mesh_t = typename Tag::mesh_t;
        using mesh_id_t = typename Tag::mesh_t::mesh_id_t;
        using cl_type = typename Tag::mesh_t::cl_type;

        auto mesh = tag.mesh();

        cl_type cl;

        for_each_interval(mesh[mesh_id_t::cells], [&](std::size_t level, const auto& interval, const auto& index)
        {
            std::size_t itag = static_cast<std::size_t>(interval.start + interval.index);
            for (auto i=interval.start; i<interval.end; ++i)
            {
                if ( tag[itag] & static_cast<int>(CellFlag::refine))
                {
                    static_nested_loop<dim-1, 0, 2>([&](auto& stencil)
                    {
                        auto new_index = 2*index + stencil;
                        cl[level + 1][new_index].add_interval({2*i, 2*i + 2});
                    });
                }
                else if ( tag[itag] & static_cast<int>(CellFlag::keep))
                {
                    cl[level][index].add_point(i);
                }
                else
                {
                    cl[level - 1][index >> 1].add_point(i >> 1);
                }
                itag++;
            }
        });

        mesh_t new_mesh = {cl, mesh.min_level(), mesh.max_level()};

        if (mesh == new_mesh)
        {
            return true;
        }

        detail::update_fields(new_mesh, fields...);
        tag.mesh_ptr()->swap(new_mesh);
        return false;
    }

    template<class Field, class Tag>
    bool update_field_mr(Field& field, Field& old_field, const Tag& tag)
    {
        static constexpr std::size_t dim = Field::dim;
        using mesh_t = typename Field::mesh_t;
        constexpr std::size_t pred_order = mesh_t::config::prediction_order;
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;
        using interval_t = typename mesh_t::interval_t;
        using coord_index_t = typename interval_t::coord_index_t;
        using cl_type = typename Field::mesh_t::cl_type;

        auto mesh = field.mesh();

        cl_type cl;

        for_each_interval(mesh[mesh_id_t::cells], [&](std::size_t level, const auto& interval, const auto& index)
        {
            std::size_t itag = static_cast<std::size_t>(interval.start + interval.index);
            for (coord_index_t i=interval.start; i<interval.end; ++i)
            {
                if ( tag[itag] & static_cast<int>(CellFlag::refine))
                {
                    static_nested_loop<dim-1, 0, 2>([&](auto& stencil)
                    {
                        auto new_index = 2*index + stencil;
                        cl[level + 1][new_index].add_interval({2*i, 2*i + 2});
                    });
                }
                else if ( tag[itag] & static_cast<int>(CellFlag::keep))
                {
                    cl[level][index].add_point(i);
                }
                else
                {
                    cl[level - 1][index >> 1].add_point(i >> 1);
                }
                itag++;
            }
        });

        mesh_t new_mesh = {cl, mesh.min_level(), mesh.max_level()};

        if (mesh == new_mesh)
        {
            return true;
        }

        Field new_field("new_f", new_mesh);
        new_field.fill(0);

        auto min_level = mesh.min_level();
        auto max_level = mesh.max_level();

        for(std::size_t level = min_level; level <= max_level; ++level)
        {
            auto set = intersection(mesh[mesh_id_t::cells][level],
                                    new_mesh[mesh_id_t::cells][level]);
            set.apply_op(copy(new_field, field));
        }

        for(std::size_t level = min_level + 1; level <= max_level; ++level)
        {
            auto set_coarsen = samurai::intersection(mesh[mesh_id_t::cells][level],
                                                    new_mesh[mesh_id_t::cells][level-1])
                              .on(level - 1);
            set_coarsen.apply_op(projection(new_field, field));

            auto set_refine = intersection(new_mesh[mesh_id_t::cells][level],
                                           mesh[mesh_id_t::cells][level-1])
                             .on(level - 1);
            set_refine.apply_op(prediction<pred_order, true>(new_field, field));
        }

        auto old_mesh = old_field.mesh();
        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            auto subset = intersection(intersection(old_mesh[mesh_id_t::cells][level],
                                            difference(new_mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level])),
                                            mesh[mesh_id_t::cells][level-1]).on(level);

            subset.apply_op(copy(new_field,  old_field));
        }

        field.mesh_ptr()->swap(new_mesh);
        old_field.mesh_ptr()->swap(new_mesh);

        std::swap(field.array(), new_field.array());
        std::swap(old_field.array(), new_field.array());

        return false;
    }
}