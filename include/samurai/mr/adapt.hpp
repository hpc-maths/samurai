// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include "../algorithm/graduation.hpp"
#include "../algorithm/update.hpp"
#include "../field.hpp"
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

    template <class TField>
    class Adapt
    {
      public:

        Adapt(TField& field);

        template <class... Fields>
        void operator()(double eps, double regularity, Fields&... other_fields);

      private:

        using field_type = TField;
        using mesh_t     = typename field_type::mesh_t;
        using mesh_id_t  = typename mesh_t::mesh_id_t;
        using tag_type   = Field<mesh_t, int, 1>;

        static constexpr std::size_t dim = field_type::dim;

        using interval_t    = typename mesh_t::interval_t;
        using coord_index_t = typename interval_t::coord_index_t;
        using cl_type       = typename mesh_t::cl_type;

        template <class... Fields>
        bool harten(std::size_t ite, double eps, double regularity, field_type& field_old, Fields&... other_fields);

        field_type& m_field;
        field_type m_detail;
        tag_type m_tag;
    };

    template <class TField>
    inline Adapt<TField>::Adapt(TField& field)
        : m_field(field)
        , m_detail("detail", field.mesh())
        , m_tag("tag", field.mesh())
    {
    }

    template <class TField>
    template <class... Fields>
    void Adapt<TField>::operator()(double eps, double regularity, Fields&... other_fields)
    {
        auto& mesh            = m_field.mesh();
        std::size_t min_level = mesh.min_level();
        std::size_t max_level = mesh.max_level();

        if (min_level == max_level)
        {
            return;
        }
        update_ghost_mr(m_field);

        mesh_t mesh_old = mesh;
        field_type field_old(m_field.name(), mesh_old);
        field_old.array() = m_field.array();
        for (std::size_t i = 0; i < max_level - min_level; ++i)
        {
            // std::cout << "MR mesh adaptation " << i << std::endl;
            m_detail.resize();
            m_tag.resize();
            m_tag.fill(0);
            if (harten(i, eps, regularity, field_old, other_fields...))
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
        auto box_dir<1>()
        {
            return xt::xtensor_fixed<int, xt::xshape<2, 1>>{{-1}, {1}};
        }

        template <>
        auto box_dir<2>()
        {
            return xt::xtensor_fixed<int, xt::xshape<4, 2>>{
                {-1, 1 },
                {1,  1 },
                {-1, -1},
                {1,  -1}
            };
        }

        template <>
        auto box_dir<3>()
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

    template <class TField>
    template <class... Fields>
    bool Adapt<TField>::harten(std::size_t ite, double eps, double regularity, field_type& field_old, Fields&... other_fields)
    {
        auto& mesh = m_field.mesh();

        std::size_t min_level = mesh.min_level(), max_level = mesh.max_level();

        for_each_cell(mesh[mesh_id_t::cells],
                      [&](auto& cell)
                      {
                          m_tag[cell] = static_cast<int>(CellFlag::keep);
                      });

        update_ghost_mr(m_field);

        for (std::size_t level = ((min_level > 0) ? min_level - 1 : 0); level < max_level - ite; ++level)
        {
            auto subset = intersection(mesh[mesh_id_t::all_cells][level], mesh[mesh_id_t::cells][level + 1]).on(level);
            subset.apply_op(compute_detail(m_detail, m_field));
        }

        for (std::size_t level = min_level; level <= max_level - ite; ++level)
        {
            std::size_t exponent = dim * (max_level - level);
            double eps_l         = eps / (1 << exponent);

            double regularity_to_use = std::min(regularity, 3.0) + dim;

            auto subset_1 = intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::all_cells][level - 1]).on(level - 1);

            subset_1.apply_op(to_coarsen_mr(m_detail, m_tag, eps_l, min_level)); // Derefinement
            subset_1.apply_op(to_refine_mr(m_detail,
                                           m_tag,
                                           (pow(2.0, regularity_to_use)) * eps_l,
                                           max_level)); // Refinement according to Harten
        }

        for (std::size_t level = min_level; level <= max_level - ite; ++level)
        {
            auto subset_2 = intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]);
            auto subset_3 = intersection(mesh[mesh_id_t::cells_and_ghosts][level], mesh[mesh_id_t::cells_and_ghosts][level]);

            subset_2.apply_op(enlarge(m_tag));
            subset_2.apply_op(keep_around_refine(m_tag));
            subset_3.apply_op(tag_to_keep<0>(m_tag, CellFlag::enlarge));
            update_tag_periodic(level, m_tag);
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
        }

        // REFINEMENT GRADUATION
        for (std::size_t level = max_level; level > min_level; --level)
        {
            auto subset_1 = intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]);

            subset_1.apply_op(extend(m_tag));
            update_tag_periodic(level, m_tag);

            int grad_width = static_cast<int>(mesh_t::config::graduation_width);
            auto stencil   = grad_width * detail::box_dir<dim>();

            for (std::size_t is = 0; is < stencil.shape(0); ++is)
            {
                auto s      = xt::view(stencil, is);
                auto subset = intersection(translate(mesh[mesh_id_t::cells][level], s), mesh[mesh_id_t::all_cells][level - 1]).on(level);

                subset.apply_op(make_graduation(m_tag));
            }

            update_tag_periodic(level, m_tag);
        }

        for (std::size_t level = max_level; level > 0; --level)
        {
            auto keep_subset = intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::all_cells][level - 1]).on(level - 1);

            keep_subset.apply_op(maximum(m_tag));
            update_tag_periodic(level, m_tag);
        }

        update_ghost_mr(field_old);
        update_ghost_mr(other_fields...);
        if (update_field_mr(m_tag, m_field, field_old, other_fields...))
        {
            return true;
        }

        return false;
    }

    template <class TField>
    auto make_MRAdapt(TField& field)
    {
        return Adapt<TField>(field);
    }
} // namespace samurai