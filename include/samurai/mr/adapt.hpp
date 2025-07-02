// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../algorithm/graduation.hpp"
#include "../algorithm/update.hpp"
#include "../arguments.hpp"
#include "../boundary.hpp"
#include "../field.hpp"
#include "../timers.hpp"
#include "criteria.hpp"
#include "operators.hpp"

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

    template <bool enlarge_, class TField, class... TFields>
    class Adapt
    {
      public:

        Adapt(TField& field, TFields&... fields);

        template <class... Fields>
        void operator()(double eps, double regularity, Fields&... other_fields);

      private:

        using inner_fields_type = detail::get_fields_type<TField, TFields...>;
        using fields_t          = typename inner_fields_type::fields_t;
        using mesh_t            = typename inner_fields_type::mesh_t;
        using mesh_id_t         = typename mesh_t::mesh_id_t;
        using detail_t          = typename inner_fields_type::detail_t;
        using tag_t             = ScalarField<mesh_t, int>;

        static constexpr std::size_t dim = mesh_t::dim;
        static constexpr bool enlarge_v  = enlarge_;

        using interval_t    = typename mesh_t::interval_t;
        using coord_index_t = typename interval_t::coord_index_t;
        using cl_type       = typename mesh_t::cl_type;

        template <class... Fields>
        bool harten(std::size_t ite, double eps, double regularity, Fields&... other_fields);

        fields_t m_fields; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
        detail_t m_detail;
        tag_t m_tag;
    };

    template <bool enlarge_, class TField, class... TFields>
    inline Adapt<enlarge_, TField, TFields...>::Adapt(TField& field, TFields&... fields)
        : m_fields(field, fields...)
        , m_detail("detail", field.mesh())
        , m_tag("tag", field.mesh())
    {
    }

    template <bool enlarge_, class TField, class... TFields>
    template <class... Fields>
    void Adapt<enlarge_, TField, TFields...>::operator()(double eps, double regularity, Fields&... other_fields)
    {
        auto& mesh            = m_fields.mesh();
        std::size_t min_level = mesh.min_level();
        std::size_t max_level = mesh.max_level();

        if (min_level == max_level)
        {
            return;
        }

        times::timers.start("mesh adaptation");
        for (std::size_t i = 0; i < max_level - min_level; ++i)
        {
            // std::cout << "MR mesh adaptation " << i << std::endl;
            m_detail.resize();
            m_detail.fill(0);
            m_tag.resize();
            m_tag.fill(0);
            if (harten(i, eps, regularity, other_fields...))
            {
                break;
            }
        }
        times::timers.stop("mesh adaptation");
    }

    // TODO: to remove since it is used at several place
    namespace detail
    {

        template <std::size_t dim>
        auto box_dir();

        template <>
        inline auto box_dir<1>()
        {
            return xt::xtensor_fixed<int, xt::xshape<2, 1>>{{-1}, {1}};
        }

        template <>
        inline auto box_dir<2>()
        {
            return xt::xtensor_fixed<int, xt::xshape<4, 2>>{
                {-1, 1 },
                {1,  1 },
                {-1, -1},
                {1,  -1}
            };
        }

        template <>
        inline auto box_dir<3>()
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
    void keep_boundary_refined(const Mesh& mesh, ScalarField<Mesh, int>& tag, const DirectionVector<Mesh::dim>& direction)
    {
        // Since the adaptation process starts at max_level, we just need to flag to `keep` the boundary cells at max_level only.
        // There will never be boundary cells at lower levels.
        auto bdry = domain_boundary_layer(mesh, mesh.max_level(), direction, Mesh::config::max_stencil_width);
        for_each_cell(mesh,
                      bdry,
                      [&](auto& cell)
                      {
                          tag[cell] = static_cast<int>(CellFlag::keep);
                      });
    }

    template <class Mesh>
    void keep_boundary_refined(const Mesh& mesh, ScalarField<Mesh, int>& tag)
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

    template <bool enlarge_, class TField, class... TFields>
    template <class... Fields>
    bool Adapt<enlarge_, TField, TFields...>::harten(std::size_t ite, double eps, double regularity, Fields&... other_fields)
    {
        auto& mesh = m_fields.mesh();

        std::size_t min_level = mesh.min_level();
        std::size_t max_level = mesh.max_level();

        for_each_cell(mesh[mesh_id_t::cells],
                      [&](auto& cell)
                      {
                          m_tag[cell] = static_cast<int>(CellFlag::keep);
                      });

        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            update_tag_subdomains(level, m_tag, true);
        }

        times::timers.stop("mesh adaptation");
        update_ghost_mr(m_fields);
        times::timers.start("mesh adaptation");

        //--------------------//
        // Detail computation //
        //--------------------//

        // We compute the detail in the cells and ghosts below the cells, except near the (non-periodic) boundaries, where we compute the
        // detail only in the cells (justification in the comments below).

        bool periodic_in_all_directions = true;
        std::array<bool, dim> contract_directions;
        for (std::size_t d = 0; d < dim; ++d)
        {
            periodic_in_all_directions = periodic_in_all_directions && mesh.is_periodic(d);
            contract_directions[d]     = !mesh.is_periodic(d);
        }

        for (std::size_t level = ((min_level > 0) ? min_level - 1 : 0); level < max_level - ite; ++level)
        {
            // 1. detail computation in the cells (at level+1)
            auto ghosts_below_cells = intersection(mesh[mesh_id_t::all_cells][level], mesh[mesh_id_t::cells][level + 1]).on(level);
            ghosts_below_cells.apply_op(compute_detail(m_detail, m_fields)); // 'compute_detail' applies 1 level above the set it is applied
                                                                             // to, i.e. level+1

            // 2. detail computation in the ghosts below cells (at level)
            if (level >= min_level)
            {
                if (periodic_in_all_directions)
                {
                    auto ghosts_2_levels_below_cells = intersection(mesh[mesh_id_t::all_cells][level - 1], ghosts_below_cells).on(level - 1);
                    ghosts_2_levels_below_cells.apply_op(compute_detail(m_detail, m_fields));
                }
                else
                {
                    // We don't want to compute the detail in the ghosts below the boundary cells. In those ghosts, we want to keep the
                    // detail to 0. We do that because that detail would use the outer ghost cells at level L-2, which holds the BC
                    // projected 2 times, and this method actually does not work well. So we're removing a layer of 4 boundary cells from
                    // the domain. This number of 4 ensures that the outer ghost at level L-2 will not be used in the prediction stencil of
                    // interior ghosts.
                    // Note: where we don't compute the detail, it stays at its initial value of 0.

                    // contract the domain only in non-periodic directions
                    auto domain_without_bdry = contract(self(mesh.domain()).on(level + 1), 4, contract_directions);
                    auto cells_without_bdry  = intersection(mesh[mesh_id_t::cells][level + 1], domain_without_bdry);
                    auto ghosts_below_cells2 = intersection(mesh[mesh_id_t::all_cells][level], cells_without_bdry).on(level);
                    auto ghosts_2_levels_below_cells = intersection(mesh[mesh_id_t::all_cells][level - 1], ghosts_below_cells2).on(level - 1);
                    ghosts_2_levels_below_cells.apply_op(compute_detail(m_detail, m_fields)); // 'compute_detail' applies 1 level above the
                                                                                              // set it is applied to, i.e. 1 level below
                                                                                              // cells
                }
            }
        }

        update_ghost_subdomains(m_detail);

        for (std::size_t level = min_level; level <= max_level - ite; ++level)
        {
            std::size_t exponent = dim * (max_level - level);
            double eps_l         = eps / (1 << exponent);

            double regularity_to_use = regularity + dim;

            auto subset_1 = intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::all_cells][level - 1]).on(level - 1);

            subset_1.apply_op(to_coarsen_mr(m_detail, m_tag, eps_l, min_level),
                              to_refine_mr(m_detail,
                                           m_tag,
                                           (pow(2.0, regularity_to_use)) * eps_l,
                                           max_level)); // Refinement according to Harten
            update_tag_subdomains(level, m_tag, true);
        }

        if (args::refine_boundary) // cppcheck-suppress knownConditionTrueFalse
        {
            keep_boundary_refined(mesh, m_tag);
        }

        for (std::size_t level = min_level; level <= max_level - ite; ++level)
        {
            auto subset_2 = intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]);

            subset_2.apply_op(keep_around_refine(m_tag));

            if constexpr (enlarge_v)
            {
                auto subset_3 = intersection(mesh[mesh_id_t::cells_and_ghosts][level], mesh[mesh_id_t::cells_and_ghosts][level]);
                subset_2.apply_op(enlarge(m_tag));
                subset_3.apply_op(tag_to_keep<0>(m_tag, CellFlag::enlarge));
            }

            update_tag_periodic(level, m_tag);
            update_tag_subdomains(level, m_tag);
        }

        for (std::size_t level = max_level; level > 0; --level)
        {
            auto keep_subset = intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::all_cells][level - 1]).on(level - 1);

            update_tag_periodic(level, m_tag);
            update_tag_subdomains(level, m_tag);

            keep_subset.apply_op(maximum(m_tag));
        }
        using ca_type = typename mesh_t::ca_type;

        // return update_field_mr(m_tag, m_fields, other_fields...);
        // for some reason I do not understand the above code produces the following error :
        // C++ exception with description "Incompatible dimension of arrays, compile in DEBUG for more info" thrown in the test body
        // on test adapt_test/2.mutliple_fields with:
        // linux-mamba (clang-18, ubuntu-24.04, clang, clang-18, clang-18, clang++-18)
        // while the code bellow do not.

        ca_type new_ca = update_cell_array_from_tag(mesh[mesh_id_t::cells], m_tag);
        make_graduation(new_ca,
                        mesh.domain(),
                        mesh.mpi_neighbourhood(),
                        mesh.periodicity(),
                        mesh_t::config::graduation_width,
                        mesh_t::config::max_stencil_width);
        mesh_t new_mesh{new_ca, mesh};
#ifdef SAMURAI_WITH_MPI
        mpi::communicator world;
        if (mpi::all_reduce(world, mesh == new_mesh, std::logical_and()))
#else
        if (mesh == new_mesh)
#endif // SAMURAI_WITH_MPI
        {
            return true;
        }
        detail::update_fields(new_mesh, m_fields, other_fields...);
        m_fields.mesh().swap(new_mesh);
        return false;
    }

    template <class... TFields>
    auto make_MRAdapt(TFields&... fields)
    {
        return Adapt<false, TFields...>(fields...);
    }

    template <bool enlarge_, class... TFields>
    auto make_MRAdapt(TFields&... fields)
    {
        return Adapt<enlarge_, TFields...>(fields...);
    }
} // namespace samurai
