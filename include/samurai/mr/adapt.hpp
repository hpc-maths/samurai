// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../algorithm/graduation.hpp"
#include "../algorithm/update.hpp"
#include "../field.hpp"
#include "../hdf5.hpp"
#include "../static_algorithm.hpp"
#include "criteria.hpp"
#include <type_traits>

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
            using fields_t                     = Field_tuple<TFields...>;
            using mesh_t                       = typename fields_t::mesh_t;
            static constexpr std::size_t nelem = fields_t::nelem;
            using common_t                     = typename fields_t::common_t;
            using detail_t                     = Field<mesh_t, common_t, nelem>;
        };

        template <class TField>
        struct get_fields_type<TField>
        {
            using fields_t = TField&;
            using mesh_t   = typename TField::mesh_t;
            using detail_t = Field<mesh_t, typename TField::value_type, TField::size>;
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
        using tag_t             = Field<mesh_t, int, 1>;

        static constexpr std::size_t dim = mesh_t::dim;
        static constexpr bool enlarge    = enlarge_;

        using interval_t    = typename mesh_t::interval_t;
        using coord_index_t = typename interval_t::coord_index_t;
        using cl_type       = typename mesh_t::cl_type;

        template <class... Fields>
        bool harten(std::size_t ite, double eps, double regularity, Fields&... other_fields);

        fields_t m_fields; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
        detail_t m_detail;
        tag_t m_tag;
    };

    template <bool enlarge, class TField, class... TFields>
    inline Adapt<enlarge, TField, TFields...>::Adapt(TField& field, TFields&... fields)
        : m_fields(field, fields...)
        , m_detail("detail", field.mesh())
        , m_tag("tag", field.mesh())
    {
    }

    template <bool enlarge, class TField, class... TFields>
    template <class... Fields>
    void Adapt<enlarge, TField, TFields...>::operator()(double eps, double regularity, Fields&... other_fields)
    {
        auto& mesh            = m_fields.mesh();
        std::size_t min_level = mesh.min_level();
        std::size_t max_level = mesh.max_level();

        if (min_level == max_level)
        {
            return;
        }
        update_ghost_mr(m_fields);

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

    template <bool enlarge, class TField, class... TFields>
    template <class... Fields>
    bool Adapt<enlarge, TField, TFields...>::harten(std::size_t ite, double eps, double regularity, Fields&... other_fields)
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
        update_ghost_mr(m_fields);

        for (std::size_t level = ((min_level > 0) ? min_level - 1 : 0); level < max_level - ite; ++level)
        {
            auto subset = intersection(mesh[mesh_id_t::all_cells][level], mesh[mesh_id_t::cells][level + 1]).on(level);
            subset.apply_op(compute_detail(m_detail, m_fields));
        }
        update_ghost_subdomains(m_detail);

        for (std::size_t level = min_level; level <= max_level - ite; ++level)
        {
            std::size_t exponent = dim * (max_level - level);
            double eps_l         = eps / (1 << exponent);

            double regularity_to_use = regularity + dim;

            auto subset_1 = intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::all_cells][level - 1]).on(level - 1);

            subset_1.apply_op(to_coarsen_mr(m_detail, m_tag, eps_l, min_level)); // Derefinement
            subset_1.apply_op(to_refine_mr(m_detail,
                                           m_tag,
                                           (pow(2.0, regularity_to_use)) * eps_l,
                                           max_level)); // Refinement according to Harten
            update_tag_subdomains(level, m_tag, true);
        }

        for (std::size_t level = min_level; level <= max_level - ite; ++level)
        {
            auto subset_2 = intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]);

            subset_2.apply_op(keep_around_refine(m_tag));

            if constexpr (enlarge)
            {
                auto subset_3 = intersection(mesh[mesh_id_t::cells_and_ghosts][level], mesh[mesh_id_t::cells_and_ghosts][level]);
                subset_2.apply_op(enlarge(m_tag));
                subset_3.apply_op(tag_to_keep<0>(m_tag, CellFlag::enlarge));
            }

            update_tag_periodic(level, m_tag);
            update_tag_subdomains(level, m_tag);
        }

        // FIXME: this graduation doesn't make the same that the lines below:
        // why? graduation(m_tag,
        // stencil_graduation::call(samurai::Dim<dim>{}));

        // COARSENING GRADUATION
        for (std::size_t level = max_level; level > 0; --level)
        {
            auto keep_subset = intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::all_cells][level - 1]).on(level - 1);

            keep_subset.apply_op(maximum(m_tag));

            int grad_width = static_cast<int>(mesh_t::config::graduation_width);
            auto stencil   = grad_width * detail::box_dir<dim>();

            for (std::size_t is = 0; is < stencil.shape(0); ++is)
            {
                auto s = xt::view(stencil, is);
                auto subset = intersection(mesh[mesh_id_t::cells][level], translate(mesh[mesh_id_t::all_cells][level - 1], s)).on(level - 1);
                subset.apply_op(balance_2to1(m_tag, s));
            }

            update_tag_periodic(level, m_tag);
            update_tag_subdomains(level, m_tag);
        }

        // REFINEMENT GRADUATION
        for (std::size_t level = max_level; level > min_level; --level)
        {
            auto subset_1 = intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]);

            subset_1.apply_op(extend(m_tag));
            update_tag_periodic(level, m_tag);
            update_tag_subdomains<false>(level, m_tag);

            int grad_width = static_cast<int>(mesh_t::config::graduation_width);
            auto stencil   = grad_width * detail::box_dir<dim>();

            for (std::size_t is = 0; is < stencil.shape(0); ++is)
            {
                auto s = xt::view(stencil, is);
                auto subset = intersection(translate(mesh[mesh_id_t::cells][level], s), mesh[mesh_id_t::all_cells][level - 1], mesh.domain())
                                  .on(level);

                subset.apply_op(make_graduation(m_tag));
            }

            update_tag_periodic(level, m_tag);
            update_tag_subdomains<false>(level, m_tag);
        }
        update_tag_subdomains<false>(min_level, m_tag);

        // Prevents the coarsening of child cells where the parent intersects the boundary.
        //
        //   outside           |   inside
        //   =======           |   ======
        //   tag this cell to  |
        //   keep to avoid     |
        //   coarsening        |
        //                     |
        //   level l+1   |-----|-----|
        //                     |
        //   level l     |-----------|
        //
        // for (std::size_t level = mesh[mesh_id_t::cells].min_level(); level <= mesh[mesh_id_t::cells].max_level(); ++level)
        // {
        //     auto set = difference(mesh[mesh_id_t::reference][level], mesh.domain()).on(level);
        //     set(
        //         [&](const auto& i, const auto& index)
        //         {
        //             m_tag(level, i, index) = static_cast<int>(CellFlag::keep);
        //         });
        // }

        for (std::size_t level = max_level; level > 0; --level)
        {
            auto keep_subset = intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::all_cells][level - 1]).on(level - 1);

            keep_subset.apply_op(maximum(m_tag));
            update_tag_periodic(level, m_tag);
            // update_tag_subdomains(level, m_tag);
            update_tag_subdomains<false>(level, m_tag);
        }

        update_ghost_mr(other_fields...);
        keep_only_one_coarse_tag(m_tag);

        return update_field_mr(m_tag, m_fields, other_fields...);
    }

    template <class... TFields>
    auto make_MRAdapt(TFields&... fields)
    {
        return Adapt<false, TFields...>(fields...);
    }

    template <bool enlarge, class... TFields>
    auto make_MRAdapt(TFields&... fields)
    {
        return Adapt<enlarge, TFields...>(fields...);
    }
} // namespace samurai
