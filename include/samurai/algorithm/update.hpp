// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <xtensor/xfixed.hpp>

#include "../bc.hpp"
#include "../mr/operators.hpp"
#include "../numeric/prediction.hpp"
#include "../numeric/projection.hpp"
#include "../subset/subset_op.hpp"
#include "utils.hpp"

namespace samurai
{
    template <class Field, class... Fields>
    void update_ghost(Field& field, Fields&... fields)
    {
        using mesh_id_t                  = typename Field::mesh_t::mesh_id_t;
        constexpr std::size_t pred_order = Field::mesh_t::config::prediction_order;

        auto& mesh            = field.mesh();
        std::size_t max_level = mesh.max_level();

        for (std::size_t level = max_level; level >= 1; --level)
        {
            auto set_at_levelm1 = intersection(mesh[mesh_id_t::proj_cells][level], mesh[mesh_id_t::reference][level - 1]).on(level - 1);
            set_at_levelm1.apply_op(variadic_projection(field, fields...));
        }

        update_bc(0, field, fields...);
        for (std::size_t level = 1; level <= max_level; ++level)
        {
            auto set_at_level = intersection(mesh[mesh_id_t::pred_cells][level], mesh[mesh_id_t::reference][level - 1]).on(level);
            set_at_level.apply_op(variadic_prediction<pred_order, false>(field, fields...));
            update_bc(level, field, fields...);
        }
    }

    template <class Field>
    void update_ghost_mro(Field& field)
    {
        using mesh_id_t                  = typename Field::mesh_t::mesh_id_t;
        constexpr std::size_t pred_order = Field::mesh_t::config::prediction_order;
        auto& mesh                       = field.mesh();

        std::size_t max_level = mesh.max_level();

        for (std::size_t level = max_level; level >= 1; --level)
        {
            auto set_at_levelm1 = intersection(mesh[mesh_id_t::reference][level], mesh[mesh_id_t::proj_cells][level - 1]).on(level - 1);
            set_at_levelm1.apply_op(projection(field));
        }

        update_bc(0, field);
        for (std::size_t level = 1; level <= max_level; ++level)
        {
            // We eliminate the overleaves from the computation since they
            // are done separately
            // auto expr =
            // difference(intersection(difference(mesh[mesh_id_t::all_cells][level],
            //                                                union_(mesh[mesh_id_t::cells][level],
            //                                                       mesh[mesh_id_t::proj_cells][level])),
            //                                     mesh.domain()),
            //                        difference(mesh[mesh_id_t::overleaves][level],
            //                                   union_(mesh[mesh_id_t::union_cells][level],
            //                                          mesh[mesh_id_t::cells_and_ghosts][level])))
            //             .on(level);

            auto expr = intersection(difference(mesh[mesh_id_t::all_cells][level],
                                                union_(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::proj_cells][level])),
                                     mesh.domain())
                            .on(level);
            expr.apply_op(prediction<pred_order, false>(field));
            update_bc(level, field);
        }
    }

    template <class Field, class... Fields>
    void update_ghost_mr(Field& field, Fields&... other_fields)
    {
        using mesh_id_t                  = typename Field::mesh_t::mesh_id_t;
        constexpr std::size_t pred_order = Field::mesh_t::config::prediction_order;

        auto& mesh            = field.mesh();
        std::size_t max_level = mesh.max_level();

        for (std::size_t level = max_level; level >= 1; --level)
        {
            auto set_at_levelm1 = intersection(mesh[mesh_id_t::reference][level], mesh[mesh_id_t::proj_cells][level - 1]).on(level - 1);
            set_at_levelm1.apply_op(variadic_projection(field, other_fields...));
        }

        update_bc(0, field, other_fields...);
        update_ghost_periodic(0, field, other_fields...);
        for (std::size_t level = 1; level <= max_level; ++level)
        {
            auto expr = intersection(difference(mesh[mesh_id_t::all_cells][level],
                                                union_(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::proj_cells][level])),
                                     mesh.domain())
                            .on(level);

            expr.apply_op(variadic_prediction<pred_order, false>(field, other_fields...));
            update_bc(level, field, other_fields...);
            update_ghost_periodic(level, field, other_fields...);
        }
    }

    inline void update_ghost_mr()
    {
    }

    template <class... T>
    inline void update_ghost_mr(std::tuple<T...>& fields)
    {
        std::apply(
            [](T&... tupleArgs)
            {
                update_ghost_mr(tupleArgs...);
            },
            fields);
    }

    template <class... T>
    inline void update_ghost_mr(Field_tuple<T...>& fields)
    {
        update_ghost_mr(fields.elements());
    }

    template <class Field>
    void update_ghost_periodic(std::size_t level, Field& field)
    {
        using mesh_id_t           = typename Field::mesh_t::mesh_id_t;
        using config              = typename Field::mesh_t::config;
        using interval_value_t    = typename Field::interval_t::value_t;
        constexpr std::size_t dim = Field::dim;

        xt::xtensor_fixed<interval_value_t, xt::xshape<dim>> stencil;
        xt::xtensor_fixed<interval_value_t, xt::xshape<dim>> stencil_dir;
        auto& mesh       = field.mesh();
        auto domain      = mesh.domain();
        auto min_indices = domain.min_indices();
        auto max_indices = domain.max_indices();

        std::size_t delta_l = domain.level() - level;
        for (std::size_t d = 0; d < dim; ++d)
        {
            if (mesh.is_periodic(d))
            {
                stencil.fill(0);
                stencil[d] = max_indices[d] - min_indices[d];

                stencil_dir.fill(0);
                stencil_dir[d] = stencil[d] + (config::ghost_width << delta_l);

                auto set1 = intersection(mesh[mesh_id_t::reference][level],
                                         expand(translate(domain, stencil_dir), (config::ghost_width << delta_l)))
                                .on(level);
                set1(
                    [&](const auto& i, const auto& index)
                    {
                        if constexpr (dim == 1)
                        {
                            field(level, i) = field(level, i - (stencil[0] >> delta_l));
                        }
                        else if constexpr (dim == 2)
                        {
                            auto j             = index[0];
                            field(level, i, j) = field(level, i - (stencil[0] >> delta_l), j - (stencil[1] >> delta_l));
                        }
                        else if constexpr (dim == 3)
                        {
                            auto j                = index[0];
                            auto k                = index[1];
                            field(level, i, j, k) = field(level,
                                                          i - (stencil[0] >> delta_l),
                                                          j - (stencil[1] >> delta_l),
                                                          k - (stencil[2] >> delta_l));
                        }
                    });

                auto set2 = intersection(mesh[mesh_id_t::reference][level],
                                         expand(translate(domain, -stencil_dir), (config::ghost_width << delta_l)))
                                .on(level);

                set2(
                    [&](const auto& i, const auto& index)
                    {
                        if constexpr (dim == 1)
                        {
                            field(level, i) = field(level, i + (stencil[0] >> delta_l));
                        }
                        else if constexpr (dim == 2)
                        {
                            auto j             = index[0];
                            field(level, i, j) = field(level, i + (stencil[0] >> delta_l), j + (stencil[1] >> delta_l));
                        }
                        else if constexpr (dim == 3)
                        {
                            auto j                = index[0];
                            auto k                = index[1];
                            field(level, i, j, k) = field(level,
                                                          i + (stencil[0] >> delta_l),
                                                          j + (stencil[1] >> delta_l),
                                                          k + (stencil[2] >> delta_l));
                        }
                    });
            }
        }
    }

    template <class Field, class... Fields>
    void update_ghost_periodic(std::size_t level, Field& field, Fields&... other_fields)
    {
        update_ghost_periodic(level, field);
        update_ghost_periodic(level, other_fields...);
    }

    template <class Field>
    void update_ghost_periodic(Field& field)
    {
        using mesh_id_t       = typename Field::mesh_t::mesh_id_t;
        auto& mesh            = field.mesh();
        std::size_t min_level = mesh[mesh_id_t::reference].min_level();
        std::size_t max_level = mesh[mesh_id_t::reference].max_level();

        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            update_ghost_periodic(level, field);
        }
    }

    template <class Field, class... Fields>
    void update_ghost_periodic(Field& field, Fields&... other_fields)
    {
        update_ghost_periodic(field);
        update_ghost_periodic(other_fields...);
    }

    template <class Tag>
    void update_tag_periodic(std::size_t level, Tag& tag)
    {
        using mesh_id_t           = typename Tag::mesh_t::mesh_id_t;
        using config              = typename Tag::mesh_t::config;
        using interval_value_t    = typename Tag::interval_t::value_t;
        constexpr std::size_t dim = Tag::dim;

        xt::xtensor_fixed<interval_value_t, xt::xshape<dim>> stencil;
        xt::xtensor_fixed<interval_value_t, xt::xshape<dim>> stencil_dir;

        auto& mesh = tag.mesh();

        auto& domain     = mesh.domain();
        auto min_indices = domain.min_indices();
        auto max_indices = domain.max_indices();

        std::size_t delta_l = domain.level() - level;
        for (std::size_t d = 0; d < dim; ++d)
        {
            if (mesh.is_periodic(d))
            {
                stencil.fill(0);
                stencil[d] = max_indices[d] - min_indices[d];

                stencil_dir.fill(0);
                stencil_dir[d] = stencil[d] + (config::ghost_width << delta_l);

                auto set1 = intersection(mesh[mesh_id_t::reference][level],
                                         expand(translate(domain, stencil_dir), (config::ghost_width << delta_l)))
                                .on(level);
                set1(
                    [&](const auto& i, const auto& index)
                    {
                        if constexpr (dim == 1)
                        {
                            tag(level, i) |= tag(level, i - (stencil[0] >> delta_l));
                            tag(level, i - (stencil[0] >> delta_l)) |= tag(level, i);
                        }
                        else if constexpr (dim == 2)
                        {
                            auto j = index[0];
                            tag(level, i, j) |= tag(level, i - (stencil[0] >> delta_l), j - (stencil[1] >> delta_l));
                            tag(level, i - (stencil[0] >> delta_l), j - (stencil[1] >> delta_l)) |= tag(level, i, j);
                        }
                        else if constexpr (dim == 3)
                        {
                            auto j = index[0];
                            auto k = index[1];
                            tag(level, i, j, k) |= tag(level,
                                                       i - (stencil[0] >> delta_l),
                                                       j - (stencil[1] >> delta_l),
                                                       k - (stencil[2] >> delta_l));
                            tag(level, i - (stencil[0] >> delta_l), j - (stencil[1] >> delta_l), k - (stencil[2] >> delta_l)) |= tag(level,
                                                                                                                                     i,
                                                                                                                                     j,
                                                                                                                                     k);
                        }
                    });

                auto set2 = intersection(mesh[mesh_id_t::reference][level],
                                         expand(translate(domain, -stencil_dir), (config::ghost_width << delta_l)))
                                .on(level);
                set2(
                    [&](const auto& i, const auto& index)
                    {
                        if constexpr (dim == 1)
                        {
                            tag(level, i) |= tag(level, i + (stencil[0] >> delta_l));
                            tag(level, i + (stencil[0] >> delta_l)) |= tag(level, i);
                        }
                        else if constexpr (dim == 2)
                        {
                            auto j = index[0];
                            tag(level, i, j) |= tag(level, i + (stencil[0] >> delta_l), j + (stencil[1] >> delta_l));
                            tag(level, i + (stencil[0] >> delta_l), j + (stencil[1] >> delta_l)) |= tag(level, i, j);
                        }
                        else if constexpr (dim == 3)
                        {
                            auto j = index[0];
                            auto k = index[1];
                            tag(level, i, j, k) |= tag(level,
                                                       i + (stencil[0] >> delta_l),
                                                       j + (stencil[1] >> delta_l),
                                                       k + (stencil[2] >> delta_l));
                            tag(level, i + (stencil[0] >> delta_l), j + (stencil[1] >> delta_l), k + (stencil[2] >> delta_l)) |= tag(level,
                                                                                                                                     i,
                                                                                                                                     j,
                                                                                                                                     k);
                        }
                    });
            }
        }
    }

    template <class Field>
    void update_overleaves_mr(Field& field)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;

        auto& mesh            = field.mesh();
        std::size_t min_level = mesh.min_level();
        std::size_t max_level = mesh.max_level();

        update_bc(min_level, field);
        for (std::size_t level = min_level + 1; level <= max_level; ++level)
        {
            // These are the overleaves which are nothing else
            // because when this procedure is called all the rest
            // should be already with the right value.
            auto overleaves_to_predict = difference(difference(mesh[mesh_id_t::overleaves][level], mesh[mesh_id_t::cells_and_ghosts][level]),
                                                    mesh[mesh_id_t::proj_cells][level]);

            overleaves_to_predict.apply_op(prediction<1, false>(field));
            update_bc(level, field);
        }
    }

    namespace detail
    {
        template <class Mesh, class Field>
        void update_fields(Mesh& new_mesh, Field& field)
        {
            using mesh_id_t                  = typename Mesh::mesh_id_t;
            constexpr std::size_t pred_order = Field::mesh_t::config::prediction_order;

            Field new_field("new_f", new_mesh);
            new_field.fill(0);

            auto& mesh = field.mesh();

            auto min_level = mesh.min_level();
            auto max_level = mesh.max_level();

            for (std::size_t level = min_level; level <= max_level; ++level)
            {
                auto set = intersection(mesh[mesh_id_t::cells][level], new_mesh[mesh_id_t::cells][level]);
                set.apply_op(copy(new_field, field));
            }

            for (std::size_t level = min_level + 1; level <= max_level; ++level)
            {
                auto set_coarsen = intersection(mesh[mesh_id_t::cells][level], new_mesh[mesh_id_t::cells][level - 1]).on(level - 1);
                set_coarsen.apply_op(projection(new_field, field));

                auto set_refine = intersection(new_mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level - 1]).on(level - 1);
                set_refine.apply_op(prediction<pred_order, true>(new_field, field));
            }

            std::swap(field.array(), new_field.array());
        }

        template <class Mesh, class Field>
        void update_fields_with_old(Mesh& new_mesh, Field& old_field, Field& field)
        {
            using mesh_id_t                  = typename Mesh::mesh_id_t;
            constexpr std::size_t pred_order = Field::mesh_t::config::prediction_order;

            Field new_field("new_f", new_mesh);
            new_field.fill(0);

            auto& mesh     = field.mesh();
            auto& old_mesh = old_field.mesh();

            auto min_level = mesh.min_level();
            auto max_level = mesh.max_level();

            for (std::size_t level = min_level; level <= max_level; ++level)
            {
                auto set = intersection(mesh[mesh_id_t::cells][level], new_mesh[mesh_id_t::cells][level]);
                set.apply_op(copy(new_field, field));
            }

            for (std::size_t level = min_level + 1; level <= max_level; ++level)
            {
                auto set_coarsen = intersection(mesh[mesh_id_t::cells][level], new_mesh[mesh_id_t::cells][level - 1]).on(level - 1);
                set_coarsen.apply_op(projection(new_field, field));

                auto set_refine = intersection(new_mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level - 1]).on(level - 1);
                set_refine.apply_op(prediction<pred_order, true>(new_field, field));
            }

            for (std::size_t level = 1; level <= max_level; ++level)
            {
                auto subset = intersection(intersection(old_mesh[mesh_id_t::cells][level],
                                                        difference(new_mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level])),
                                           mesh[mesh_id_t::cells][level - 1])
                                  .on(level);

                subset.apply_op(copy(new_field, old_field));
            }

            std::swap(field.array(), new_field.array());
            std::swap(old_field.array(), new_field.array());
        }

        template <class Mesh, class Old_fields, class Fields, std::size_t... Is>
        void update_fields_with_old(Mesh& new_mesh, Old_fields& old_fields, Fields& fields, std::index_sequence<Is...>)
        {
            (update_fields_with_old(new_mesh, std::get<Is>(old_fields), std::get<Is>(fields)), ...);
        }

        template <class Mesh, class Old_fields, class... T>
        void update_fields_with_old(Mesh& new_mesh, Old_fields& old_fields, Field_tuple<T...>& fields)
        {
            update_fields_with_old(new_mesh, old_fields, fields.elements(), std::make_index_sequence<sizeof...(T)>{});
        }

        template <class Mesh, class Field, class... Fields>
        void update_fields(Mesh& new_mesh, Field& field, Fields&... fields)
        {
            update_fields(new_mesh, field);
            update_fields(new_mesh, fields...);
        }

        template <class Mesh>
        void update_fields(Mesh&)
        {
        }

        template <class Mesh, class Field>
        void swap_mesh(Mesh& new_mesh, Field& old_field, Field& field)
        {
            field.mesh().swap(new_mesh);
            old_field.mesh().swap(new_mesh);
        }

        template <class Mesh, class Old_fields, class... T>
        void swap_mesh(Mesh& new_mesh, Old_fields& old_fields, Field_tuple<T...>& fields)
        {
            fields.mesh().swap(new_mesh);
            std::get<0>(old_fields).mesh().swap(new_mesh);
        }
    }

    template <class Tag, class... Fields>
    bool update_field(Tag& tag, Fields&... fields)
    {
        static constexpr std::size_t dim = Tag::dim;
        using mesh_t                     = typename Tag::mesh_t;
        using mesh_id_t                  = typename Tag::mesh_t::mesh_id_t;
        using cl_type                    = typename Tag::mesh_t::cl_type;

        auto& mesh = tag.mesh();

        cl_type cl;

        for_each_interval(mesh[mesh_id_t::cells],
                          [&](std::size_t level, const auto& interval, const auto& index)
                          {
                              auto itag = interval.start + interval.index;
                              for (auto i = interval.start; i < interval.end; ++i)
                              {
                                  if (tag[itag] & static_cast<int>(CellFlag::refine))
                                  {
                                      if (level < mesh.max_level())
                                      {
                                          static_nested_loop<dim - 1, 0, 2>(
                                              [&](const auto& stencil)
                                              {
                                                  auto new_index = 2 * index + stencil;
                                                  cl[level + 1][new_index].add_interval({2 * i, 2 * i + 2});
                                              });
                                      }
                                      else
                                      {
                                          cl[level][index].add_point(i);
                                      }
                                  }
                                  else if (tag[itag] & static_cast<int>(CellFlag::keep))
                                  {
                                      cl[level][index].add_point(i);
                                  }
                                  else
                                  {
                                      if (level > mesh.min_level())
                                      {
                                          cl[level - 1][index >> 1].add_point(i >> 1);
                                      }
                                      else
                                      {
                                          cl[level][index].add_point(i);
                                      }
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
        tag.mesh().swap(new_mesh);
        return false;
    }

    template <class Tag, class Field, class Old_field, class... Fields>
    bool update_field_mr(const Tag& tag, Field& field, Old_field& old_field, Fields&... other_fields)
    {
        using mesh_t                     = typename Field::mesh_t;
        static constexpr std::size_t dim = mesh_t::dim;
        using mesh_id_t                  = typename Field::mesh_t::mesh_id_t;
        using interval_t                 = typename mesh_t::interval_t;
        using value_t                    = typename interval_t::value_t;
        using cl_type                    = typename Field::mesh_t::cl_type;

        auto& mesh = field.mesh();
        cl_type cl;

        for_each_interval(mesh[mesh_id_t::cells],
                          [&](std::size_t level, const auto& interval, const auto& index)
                          {
                              auto itag = interval.start + interval.index;
                              for (value_t i = interval.start; i < interval.end; ++i)
                              {
                                  if (tag[itag] & static_cast<int>(CellFlag::refine))
                                  {
                                      if (level < mesh.max_level())
                                      {
                                          static_nested_loop<dim - 1, 0, 2>(
                                              [&](const auto& stencil)
                                              {
                                                  auto new_index = 2 * index + stencil;
                                                  cl[level + 1][new_index].add_interval({2 * i, 2 * i + 2});
                                              });
                                      }
                                      else
                                      {
                                          cl[level][index].add_point(i);
                                      }
                                  }
                                  else if (tag[itag] & static_cast<int>(CellFlag::keep))
                                  {
                                      cl[level][index].add_point(i);
                                  }
                                  else
                                  {
                                      if (level > mesh.min_level())
                                      {
                                          cl[level - 1][index >> 1].add_point(i >> 1);
                                      }
                                      else
                                      {
                                          cl[level][index].add_point(i);
                                      }
                                  }
                                  itag++;
                              }
                          });

        mesh_t new_mesh = {cl, mesh.min_level(), mesh.max_level(), mesh.periodicity()};

        if (mesh == new_mesh)
        {
            return true;
        }

        detail::update_fields(new_mesh, other_fields...);
        detail::update_fields_with_old(new_mesh, old_field, field);

        detail::swap_mesh(new_mesh, old_field, field);
        // field.mesh().swap(new_mesh);
        // std::get<0>(old_field).mesh().swap(new_mesh);

        return false;
    }
}
