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
        template <class PredictionFn, class DestTuple, class SrcTuple, class Mesh, std::size_t... Is>
        void update_fields_impl(PredictionFn&& prediction_fn, Mesh& new_mesh, DestTuple& dests, SrcTuple& srcs, std::index_sequence<Is...>)
        {
            ScopedTimer timer("fields update");
            using mesh_id_t = typename Mesh::mesh_id_t;

            auto& mesh     = std::get<0>(srcs).mesh();
            auto min_level = mesh.min_level();
            auto max_level = mesh.max_level();

            // Single loop over levels: copy reference -> cells for ALL fields in
            // a single traversal of the set (variadic copy over the two tuples).
            for (std::size_t level = min_level; level <= max_level; ++level)
            {
                auto set = intersection(mesh[mesh_id_t::reference][level], new_mesh[mesh_id_t::cells][level]);
                set.apply_op(copy(dests, srcs));
            }

            // Single loop over levels: coarsen and refine for ALL fields
            for (std::size_t level = min_level + 1; level <= max_level; ++level)
            {
                auto set_coarsen = intersection(mesh[mesh_id_t::cells][level], new_mesh[mesh_id_t::cells][level - 1]).on(level - 1);
                set_coarsen.apply_op(projection(dests, srcs));

                auto set_refine = intersection(new_mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level - 1]).on(level - 1);
                (set_refine.apply_op(prediction_fn(std::get<Is>(dests), std::get<Is>(srcs))), ...);
            }
        }

        template <class Field, class Mesh>
        Field make_new_field(const std::string& name, Mesh& mesh)
        {
            Field new_field(name, mesh);
#ifdef SAMURAI_CHECK_NAN
            new_field.fill(std::nan(""));
#else
            new_field.fill(0);
#endif
            return new_field;
        }
    }

    namespace detail
    {
        template <class T, class = void>
        struct is_std_tuple : std::false_type
        {
        };

        template <class... Ts>
        struct is_std_tuple<std::tuple<Ts...>, void> : std::true_type
        {
        };
    }

    // Update fields with custom prediction function (tuple version)
    template <class PredictionFn, class DestTuple, class SrcTuple, class Mesh>
        requires(detail::is_std_tuple<std::remove_cvref_t<DestTuple>>::value && detail::is_std_tuple<std::remove_cvref_t<SrcTuple>>::value)
    void update_fields(PredictionFn&& prediction_fn, Mesh& new_mesh, DestTuple& dests, SrcTuple& srcs)
    {
        constexpr std::size_t n = std::tuple_size_v<std::remove_cvref_t<DestTuple>>;
        static_assert(n == std::tuple_size_v<std::remove_cvref_t<SrcTuple>>,
                      "update_fields: dest and src tuples must have the same number of fields");
        detail::update_fields_impl(std::forward<PredictionFn>(prediction_fn), new_mesh, dests, srcs, std::make_index_sequence<n>{});
    }

    template <class Mesh, class... Fields>
        requires(mesh_like<Mesh> && (field_like<Fields> && ...))
    void update_fields(Mesh& new_mesh, Fields&... fields)
    {
        using prediction_fn_t = decltype(default_config::default_prediction_fn);
        update_fields(std::forward<prediction_fn_t>(default_config::default_prediction_fn), new_mesh, fields...);
    }

    template <class PredictionFn, class Mesh, class Field, class... Fields>
        requires(!mesh_like<std::remove_cvref_t<PredictionFn>> && !detail::is_std_tuple<std::remove_cvref_t<Field>>::value)
    void update_fields(PredictionFn&& prediction_fn, Mesh& new_mesh, Field& field, Fields&... fields)
    {
        auto src_tuple = std::tuple_cat(get_elements(field), std::tie(fields...));
        auto dst_tuple = std::apply(
            [&](auto&... args)
            {
                return std::make_tuple(detail::make_new_field<std::decay_t<decltype(args)>>("new_f", new_mesh)...);
            },
            src_tuple);

        update_fields(std::forward<PredictionFn>(prediction_fn), new_mesh, dst_tuple, src_tuple);

        // Swap all fields
        std::apply(
            [&](auto&... dsts)
            {
                std::apply(
                    [&](auto&... srcs)
                    {
                        ((swap(srcs, dsts)), ...);
                    },
                    src_tuple);
            },
            dst_tuple);
    }

    template <class Tag, class... Fields>
    bool update_field(Tag& tag, Fields&... fields)
    {
        ScopedTimer timer("update_field");
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
