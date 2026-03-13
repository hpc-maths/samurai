// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <concepts>

#include <CLI/CLI.hpp>

#include "../algorithm/graduation.hpp"
#include "../algorithm/update.hpp"
#include "../arguments.hpp"
#include "../boundary.hpp"
#include "../field.hpp"
#include "../load_balancing/load_balancing_diffusion.hpp"
#include "../timers.hpp"
#include "config.hpp"
#include "criteria.hpp"
#include "operators.hpp"
#include "rel_detail.hpp"

namespace samurai
{
    struct stencil_graduation
    {
        static auto call(samurai::Dim<1>)
        {
            return xt::xtensor_fixed<int, xt::xshape<2, 1>>{{1}, {-1}};
        }

        static auto call(samurai::Dim<2>)
        {
            return xt::xtensor_fixed<int, xt::xshape<4, 2>>{
                {1,  1 },
                {-1, -1},
                {-1, 1 },
                {1,  -1}
            };
            // return xt::xtensor_fixed<int, xt::xshape<4, 2>> stencil{{ 1,  0},
            //                                                         {-1,  0},
            //                                                         { 0,  1},
            //                                                         { 0,
            //                                                         -1}};
        }

        static auto call(samurai::Dim<3>)
        {
            return xt::xtensor_fixed<int, xt::xshape<8, 3>>{
                {1,  1,  1 },
                {-1, 1,  1 },
                {1,  -1, 1 },
                {-1, -1, 1 },
                {1,  1,  -1},
                {-1, 1,  -1},
                {1,  -1, -1},
                {-1, -1, -1}
            };
            // return xt::xtensor_fixed<int, xt::xshape<6, 3>> stencil{{ 1,  0,
            // 0},
            //                                                         {-1,  0,
            //                                                         0}, { 0,
            //                                                         1,  0},
            //                                                         { 0, -1,
            //                                                         0}, { 0,
            //                                                         0,  1},
            //                                                         { 0,  0,
            //                                                         -1}};
        }
    };

    namespace detail
    {
        template <class... TFields>
        struct get_fields_type
        {
            using fields_t = Field_tuple<TFields...>;
            using mesh_t   = typename fields_t::mesh_t;
            using common_t = typename fields_t::common_t;
            using detail_t = VectorField<mesh_t, common_t, detail::compute_n_comp<TFields...>()>;
        };

        template <class TField>
        struct get_fields_type<TField>
        {
            using fields_t = TField&;
            using mesh_t   = typename TField::mesh_t;
            using detail_t = std::conditional_t<TField::is_scalar,
                                                ScalarField<mesh_t, typename TField::value_type>,
                                                VectorField<mesh_t, typename TField::value_type, TField::n_comp, detail::is_soa_v<TField>>>;
        };
    }

    template <bool enlarge_, class PredictionFn, class TField, class... TFields>
    class Adapt
    {
      public:

        Adapt(PredictionFn&& prediction_fn, TField& field, TFields&... fields);

        template <class... Fields>
        void operator()(mra_config& config, Fields&... other_fields);

        template <class... Fields>
        void operator()(double eps, double regularity, Fields&... other_fields);

      private:

        using inner_fields_type = detail::get_fields_type<TField, TFields...>;
        using fields_t          = typename inner_fields_type::fields_t;
        using mesh_t            = typename inner_fields_type::mesh_t;
        using mesh_id_t         = typename mesh_t::mesh_id_t;
        using detail_t          = typename inner_fields_type::detail_t;
        using tag_t             = ScalarField<mesh_t, std::uint8_t>;

        static constexpr std::size_t dim = mesh_t::dim;
        static constexpr bool enlarge_v  = enlarge_;

        using interval_t    = typename mesh_t::interval_t;
        using coord_index_t = typename interval_t::coord_index_t;
        using cl_type       = typename mesh_t::cl_type;

        template <class... Fields>
        bool harten(std::size_t ite, const mra_config& cfg, Fields&... other_fields);

        PredictionFn m_prediction_fn;
        fields_t m_fields; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
        detail_t m_detail;
        tag_t m_tag;
#ifdef SAMURAI_WITH_MPI
        DiffusionLoadBalancer m_balancer;
        std::size_t m_adapt_ite{0};
#endif
    };

    template <bool enlarge_, class PredictionFn, class TField, class... TFields>
    SAMURAI_INLINE Adapt<enlarge_, PredictionFn, TField, TFields...>::Adapt(PredictionFn&& prediction_fn, TField& field, TFields&... fields)
        : m_prediction_fn(std::forward<PredictionFn>(prediction_fn))
        , m_fields(field, fields...)
        , m_detail("detail", field.mesh())
        , m_tag("tag", field.mesh())
    {
    }

    template <bool enlarge_, class PredictionFn, class TField, class... TFields>
    template <class... Fields>
    void Adapt<enlarge_, PredictionFn, TField, TFields...>::operator()(mra_config& cfg, Fields&... other_fields)
    {
        ScopedTimer timer("mesh adaptation");
        auto& mesh            = m_fields.mesh();
        std::size_t min_level = mesh.min_level();
        std::size_t max_level = mesh.max_level();

        if (min_level == max_level)
        {
            return;
        }

        cfg.parse_args();
        for (std::size_t i = 0; i < max_level - min_level; ++i)
        {
            // std::cout << "MR mesh adaptation " << i << std::endl;
            m_detail.resize();
            m_detail.fill(0);
            m_tag.resize();
            m_tag.fill(0);
            if (harten(i, cfg, other_fields...))
            {
                break;
            }
        }
#ifdef SAMURAI_WITH_MPI
        if constexpr (dim == 2) // only works in 2D for now
        {
            if (args::load_balancing_at > 0 && m_adapt_ite % args::load_balancing_at == 0)
            {
                if constexpr (std::same_as<fields_t, Field_tuple<TField, TFields...>>)
                {
                    std::apply(
                        [&](auto&&... fields)
                        {
                            m_balancer.load_balance(fields..., other_fields...);
                        },
                        m_fields.elements());
                }
                else
                {
                    m_balancer.load_balance(m_fields, other_fields...);
                }
            }
            m_adapt_ite++;
        }
#endif
    }

    template <bool enlarge_, class PredictionFn, class TField, class... TFields>
    template <class... Fields>
    [[deprecated("Use mra_config instead of eps and regularity (see advection_2d example for more information)")]]
    void Adapt<enlarge_, PredictionFn, TField, TFields...>::operator()(double eps, double regularity, Fields&... other_fields)
    {
        operator()(mra_config().epsilon(eps).regularity(regularity), other_fields...);
    }

    // TODO: to remove since it is used at several place
    namespace detail
    {

        template <std::size_t dim>
        auto box_dir();

        template <>
        SAMURAI_INLINE auto box_dir<1>()
        {
            return xt::xtensor_fixed<int, xt::xshape<2, 1>>{{-1}, {1}};
        }

        template <>
        SAMURAI_INLINE auto box_dir<2>()
        {
            return xt::xtensor_fixed<int, xt::xshape<4, 2>>{
                {-1, 1 },
                {1,  1 },
                {-1, -1},
                {1,  -1}
            };
        }

        template <>
        SAMURAI_INLINE auto box_dir<3>()
        {
            return xt::xtensor_fixed<int, xt::xshape<8, 3>>{
                {-1, -1, -1},
                {1,  -1, -1},
                {-1, 1,  -1},
                {1,  1,  -1},
                {-1, -1, 1 },
                {1,  -1, 1 },
                {-1, 1,  1 },
                {1,  1,  1 }
            };
        }
    }

    template <class Mesh>
    void keep_boundary_refined(const Mesh& mesh, auto& tag, const DirectionVector<Mesh::dim>& direction)
    {
        // Since the adaptation process starts at max_level, we just need to flag to `keep` the boundary cells at max_level only.
        // There will never be boundary cells at lower levels.
        auto bdry = domain_boundary_layer(mesh, mesh.max_level(), direction, static_cast<std::size_t>(mesh.max_stencil_radius()));
        for_each_cell(mesh,
                      bdry,
                      [&](auto& cell)
                      {
                          tag[cell] = static_cast<std::uint8_t>(CellFlag::keep);
                      });
    }

    template <class Mesh>
    void keep_boundary_refined(const Mesh& mesh, auto& tag)
    {
        constexpr std::size_t dim = Mesh::dim;

        DirectionVector<dim> direction;
        direction.fill(0);
        for (std::size_t d = 0; d < dim; ++d)
        {
            direction(d) = 1;
            keep_boundary_refined(mesh, tag, direction);
            direction(d) = -1;
            keep_boundary_refined(mesh, tag, direction);
            direction(d) = 0;
        }
    }

    template <bool enlarge_, class PredictionFn, class TField, class... TFields>
    template <class... Fields>
    bool Adapt<enlarge_, PredictionFn, TField, TFields...>::harten(std::size_t ite, const mra_config& cfg, Fields&... other_fields)
    {
        // ScopedTimer timer(fmt::format("harten criterion {}", ite));
        auto& mesh = m_fields.mesh();

        std::size_t min_level = mesh.min_level();
        std::size_t max_level = mesh.max_level();

        for_each_cell(mesh[mesh_id_t::cells],
                      [&](auto& cell)
                      {
                          m_tag[cell] = static_cast<std::uint8_t>(CellFlag::keep);
                      });

        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            update_tag_subdomains(level, m_tag, true);
        }

        update_ghost_mr(m_fields);

        //--------------------//
        // Detail computation //
        //--------------------//

        // We compute the detail in the cells and ghosts below the cells, except near the (non-periodic) boundaries, where we compute
        // the detail only in the cells (justification in the comments below).

        bool periodic_in_all_directions = true;
        std::array<bool, dim> contract_directions;
        for (std::size_t d = 0; d < dim; ++d)
        {
            periodic_in_all_directions = periodic_in_all_directions && mesh.is_periodic(d);
            contract_directions[d]     = !mesh.is_periodic(d);
        }

        times::timers.start("detail computation");
        for (std::size_t level = ((min_level > 0) ? min_level - 1 : 0); level < max_level - ite; ++level)
        {
            auto ghosts_below_cells = samurai::intersection(
                                          mesh[mesh_id_t::all_cells][level],
                                          samurai::union_(mesh[mesh_id_t::cells][level + 1], mesh[mesh_id_t::cells][level + 2]))
                                          .on(level);
            ghosts_below_cells.apply_op(compute_detail(m_detail, m_fields));
        }

        if (cfg.relative_detail())
        {
            compute_relative_detail(m_detail, m_fields);
        }
        times::timers.stop("detail computation");

        update_ghost_subdomains(m_detail);

        times::timers.start("tag cells");
        for (std::size_t level = min_level; level <= max_level - ite; ++level)
        {
            std::size_t exponent = dim * (max_level - level);
            double eps_l         = cfg.epsilon() / (1 << exponent);

            double regularity_to_use = cfg.regularity() + dim;

            auto subset_1 = intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::all_cells][level - 1]).on(level - 1);

            subset_1.apply_op(mr_criteria(m_detail, m_tag, eps_l, regularity_to_use));
            update_tag_subdomains(level, m_tag, true);
        }
        times::timers.stop("tag cells");

        times::timers.start("refine boundary");
        if (args::refine_boundary) // cppcheck-suppress knownConditionTrueFalse
        {
            keep_boundary_refined(mesh, m_tag);
        }
        times::timers.stop("refine boundary");

        times::timers.start("tag computation");
        // for (std::size_t level = min_level; level <= max_level - ite; ++level)
        // {
        //     auto subset_2 = intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]);

        //     subset_2.apply_op(keep_around_refine(m_tag));

        //     if constexpr (enlarge_v)
        //     {
        //         auto subset_3 = intersection(mesh[mesh_id_t::cells_and_ghosts][level], mesh[mesh_id_t::cells_and_ghosts][level]);
        //         subset_2.apply_op(enlarge(m_tag));
        //         subset_3.apply_op(tag_to_keep<0>(m_tag, CellFlag::enlarge));
        //     }

        //     update_tag_periodic(level, m_tag);
        //     update_tag_subdomains(level, m_tag);
        // }

        for (std::size_t level = max_level; level > 0; --level)
        {
            auto keep_subset = intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::all_cells][level - 1]).on(level - 1);

            update_tag_periodic(level, m_tag);
            update_tag_subdomains(level, m_tag);

            keep_subset.apply_op(maximum(m_tag));
        }
        times::timers.stop("tag computation");

        times::timers.start("mesh update");
        using ca_type = typename mesh_t::ca_type;

        ca_type new_ca = update_cell_array_from_tag(mesh[mesh_id_t::cells], m_tag);
        make_graduation(new_ca, mesh.domain(), mesh.mpi_neighbourhood(), mesh.periodicity(), mesh.graduation_width(), mesh.max_stencil_radius());
        mesh_t new_mesh{new_ca, mesh};
#ifdef SAMURAI_WITH_MPI
        mpi::communicator world;
        if (mpi::all_reduce(world, mesh == new_mesh, std::logical_and()))
#else
        if (mesh == new_mesh)
#endif // SAMURAI_WITH_MPI
        {
            times::timers.stop("mesh update");
            return true;
        }
        times::timers.stop("mesh update");
        update_ghost_mr(other_fields...);

        times::timers.start("fields update");
        update_fields(std::forward<PredictionFn>(m_prediction_fn), new_mesh, m_fields, other_fields...);
        m_fields.mesh().swap(new_mesh);
        times::timers.stop("fields update");
        return false;
    }

    template <class... TFields>
        requires(field_like<TFields> && ...)
    auto make_MRAdapt(TFields&... fields)
    {
        using prediction_fn_t = decltype(default_config::default_prediction_fn);
        return Adapt<false, prediction_fn_t, TFields...>(std::forward<prediction_fn_t>(default_config::default_prediction_fn), fields...);
    }

    template <class Prediction_fn, class... TFields>
        requires(!field_like<Prediction_fn>) && (field_like<TFields> && ...)
    auto make_MRAdapt(Prediction_fn&& prediction_fn, TFields&... fields)
    {
        std::cout << "Use custom prediction function for MRAdapt" << std::endl;
        return Adapt<false, Prediction_fn, TFields...>(std::forward<Prediction_fn>(prediction_fn), fields...);
    }

    template <bool enlarge_, class... TFields>
        requires(field_like<TFields> && ...)
    auto make_MRAdapt(TFields&... fields)
    {
        using prediction_fn_t = decltype(default_config::default_prediction_fn);
        return Adapt<enlarge_, prediction_fn_t, TFields...>(std::forward<prediction_fn_t>(default_config::default_prediction_fn), fields...);
    }

    template <bool enlarge_, class Prediction_fn, class... TFields>
        requires(!field_like<Prediction_fn>) && (field_like<TFields> && ...)
    auto make_MRAdapt(Prediction_fn&& prediction_fn, TFields&... fields)
    {
        return Adapt<enlarge_, Prediction_fn, TFields...>(std::forward<Prediction_fn>(prediction_fn), fields...);
    }
} // namespace samurai
