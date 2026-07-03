// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <functional>
#include <tuple>
#include <utility>

#include "../algorithm.hpp"
#include "../concepts.hpp"
#include "../field.hpp"
#include "../numeric/projection.hpp"
#include "../timers.hpp"
#include "utils.hpp"

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
namespace mpi = boost::mpi;
#endif

namespace samurai
{
    namespace detail
    {
        template <class PredictionOp, class Mesh, class Field>
        void update_field(PredictionOp&& prediction_op, Mesh& new_mesh, Field& field)
        {
            ScopedTimer timer("fields update");
            using mesh_id_t = typename Mesh::mesh_id_t;

            Field new_field("new_f", new_mesh);
#ifdef SAMURAI_CHECK_NAN
            new_field.fill(std::nan(""));
#else
            new_field.fill(0);
#endif

            auto& mesh = field.mesh();

            auto min_level = mesh.min_level();
            auto max_level = mesh.max_level();

            for (std::size_t level = min_level; level <= max_level; ++level)
            {
                auto set = intersection(mesh[mesh_id_t::reference][level], new_mesh[mesh_id_t::cells][level]);
                set.apply_op(copy(new_field, field));
            }

            for (std::size_t level = min_level + 1; level <= max_level; ++level)
            {
                auto set_coarsen = intersection(mesh[mesh_id_t::cells][level], new_mesh[mesh_id_t::cells][level - 1]).on(level - 1);
                set_coarsen.apply_op(projection(new_field, field));

                auto set_refine = intersection(new_mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level - 1]).on(level - 1);
                set_refine.apply_op(std::forward<PredictionOp>(prediction_op)(new_field, field));
            }

            swap(field, new_field);
        }
    }

    template <class PredictionFn, class Mesh>
        requires mesh_like<Mesh>
    void update_fields(PredictionFn&&, Mesh&)
    {
    }

    template <class Mesh>
        requires mesh_like<Mesh>
    void update_fields(Mesh&)
    {
    }

    template <class PredictionFn, class Mesh, class Fields, std::size_t... Is>
    void update_fields(PredictionFn&& prediction_fn, Mesh& new_mesh, Fields& fields, std::index_sequence<Is...>)
    {
        (detail::update_field(std::forward<PredictionFn>(prediction_fn), new_mesh, std::get<Is>(fields)), ...);
    }

    template <class Mesh, class Fields, std::size_t... Is>
    void update_fields(Mesh& new_mesh, Fields& fields, std::index_sequence<Is...>)
    {
        using prediction_fn_t = decltype(default_config::default_prediction_fn);
        (detail::update_field(std::forward<prediction_fn_t>(default_config::default_prediction_fn), new_mesh, std::get<Is>(fields)), ...);
    }

    template <class PredictionFn, class Mesh, class... T>
        requires mesh_like<Mesh> && (field_like<T> && ...)
    void update_fields(PredictionFn&& prediction_fn, Mesh& new_mesh, Field_tuple<T...>& fields)
    {
        update_fields(std::forward<PredictionFn>(prediction_fn), new_mesh, fields.elements(), std::make_index_sequence<sizeof...(T)>{});
    }

    template <class Mesh, class... T>
        requires mesh_like<Mesh> && (field_like<T> && ...)
    void update_fields(Mesh& new_mesh, Field_tuple<T...>& fields)
    {
        update_fields(new_mesh, fields.elements(), std::make_index_sequence<sizeof...(T)>{});
    }

    template <class Mesh, class Field, class... Fields>
        requires mesh_like<Mesh> && field_like<Field> && (field_like<Fields> && ...)
    void update_fields(Mesh& new_mesh, Field& field, Fields&... fields)
    {
        using prediction_fn_t = decltype(default_config::default_prediction_fn);
        detail::update_field(std::forward<prediction_fn_t>(default_config::default_prediction_fn), new_mesh, field);
        update_fields(new_mesh, fields...);
    }

    template <class PredictionFn, class Mesh, class Field, class... Fields>
        requires mesh_like<Mesh> && field_like<Field> && (field_like<Fields> && ...)
    void update_fields(PredictionFn&& prediction_fn, Mesh& new_mesh, Field& field, Fields&... fields)
    {
        detail::update_field(std::forward<PredictionFn>(prediction_fn), new_mesh, field);
        update_fields(std::forward<PredictionFn>(prediction_fn), new_mesh, fields...);
    }

    template <class Tag, class... Fields>
    bool update_field(Tag& tag, Fields&... fields)
    {
        static constexpr std::size_t dim = Tag::dim;
        using mesh_t                     = typename Tag::mesh_t;
        using size_type                  = typename Tag::size_type;
        using mesh_id_t                  = typename Tag::mesh_t::mesh_id_t;
        using cl_type                    = typename Tag::mesh_t::cl_type;

        auto& mesh = tag.mesh();

        cl_type cl;

        for_each_interval(mesh[mesh_id_t::cells],
                          [&](std::size_t level, const auto& interval, const auto& index)
                          {
                              auto itag = static_cast<size_type>(interval.start + interval.index);
                              for (auto i = interval.start; i < interval.end; ++i)
                              {
                                  if (tag[itag] & static_cast<std::uint8_t>(CellFlag::refine))
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
                                  else if (tag[itag] & static_cast<std::uint8_t>(CellFlag::keep))
                                  {
                                      cl[level][index].add_point(i);
                                  }
                                  else if (tag[itag] & static_cast<std::uint8_t>(CellFlag::coarsen))
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

        mesh_t new_mesh = {cl, mesh};

#ifdef SAMURAI_WITH_MPI
        mpi::communicator world;
        if (mpi::all_reduce(world, mesh == new_mesh, std::logical_and()))
#else
        if (mesh == new_mesh)
#endif
        {
            return true;
        }

        update_fields(new_mesh, fields...);
        tag.mesh().swap(new_mesh);
        return false;
    }
}
