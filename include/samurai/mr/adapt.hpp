#pragma once

#include <type_traits>
#include "../field.hpp"
#include "pred_and_proj.hpp"

namespace samurai
{
    template <class TField, class Func>
    class Adapt
    {
    public:
        Adapt(TField& field, Func&& update_bc_by_level);

        void operator()(double eps, double regularity);

    private:
        using field_type = TField;
        using mesh_t = typename field_type::mesh_t;
        using mesh_id_t = typename mesh_t::mesh_id_t;
        using tag_type = Field<mesh_t, int, 1>;

        static constexpr std::size_t dim = field_type::dim;

        using interval_t = typename mesh_t::interval_t;
        using coord_index_t = typename interval_t::coord_index_t;
        using cl_type = typename mesh_t::cl_type;

        bool harten(std::size_t ite, double eps, double regularity, field_type& field_old);

        field_type& m_field;
        field_type m_detail;
        tag_type m_tag;
        Func m_update_bc_for_level;
    };

    template <class TField, class Func>
    inline Adapt<TField, Func>::Adapt(TField& field, Func&& update_bc_for_level)
    : m_field(field)
    , m_detail("detail", field.mesh())
    , m_tag("tag", field.mesh())
    , m_update_bc_for_level(std::forward<Func>(update_bc_for_level))
    {}

    template <class TField, class Func>
    void Adapt<TField, Func>::operator()(double eps, double regularity)
    {
        auto mesh = m_field.mesh();
        std::size_t min_level = mesh.min_level();
        std::size_t max_level = mesh.max_level();

        mesh_t mesh_old = mesh;
        field_type field_old(m_field.name(), mesh_old);
        field_old.array() = m_field.array();
        for (std::size_t i = 0; i < max_level - min_level; ++i)
        {
            m_detail.resize();
            m_tag.resize();
            m_tag.fill(0);
            if (harten(i, eps, regularity, field_old))
            {
                break;
            }
        }
    }

    template <class TField, class Func>
    bool Adapt<TField, Func>::harten(std::size_t ite, double eps, double regularity, field_type& field_old)
    {
        auto mesh = m_field.mesh();
        std::size_t min_level = mesh.min_level(), max_level = mesh.max_level();

        for_each_cell(mesh[mesh_id_t::cells], [&](auto &cell)
        {
            m_tag[cell] = static_cast<int>(CellFlag::keep);
        });

        mr_projection(m_field);
        for (std::size_t level = min_level - 1; level <= max_level; ++level)
        {
            m_update_bc_for_level(m_field, level); // It is important to do so
        }
        mr_prediction(m_field, m_update_bc_for_level);

        for (std::size_t level = min_level - 1; level < max_level - ite; ++level)
        {
            auto subset = intersection(mesh[mesh_id_t::all_cells][level],
                                    mesh[mesh_id_t::cells][level + 1])
                        .on(level);
            subset.apply_op(compute_detail(m_detail, m_field));
        }

        for (std::size_t level = min_level; level <= max_level - ite; ++level)
        {
            double exponent = dim * (max_level - level);
            double eps_l = std::pow(2., -exponent) * eps;

            double regularity_to_use = std::min(regularity, 3.0) + dim;

            auto subset_1 = intersection(mesh[mesh_id_t::cells][level],
                                            mesh[mesh_id_t::all_cells][level-1])
                        .on(level-1);

            subset_1.apply_op(to_coarsen_mr(m_detail, m_tag, eps_l, min_level)); // Derefinement
            subset_1.apply_op(to_refine_mr(m_detail, m_tag, (pow(2.0, regularity_to_use)) * eps_l, max_level)); // Refinement according to Harten
        }

        for (std::size_t level = min_level; level <= max_level - ite; ++level)
        {
            auto subset_2 = intersection(mesh[mesh_id_t::cells][level],
                                        mesh[mesh_id_t::cells][level]);
            auto subset_3 = intersection(mesh[mesh_id_t::cells_and_ghosts][level],
                                        mesh[mesh_id_t::cells_and_ghosts][level]);

            subset_2.apply_op(enlarge(m_tag));
            subset_2.apply_op(keep_around_refine(m_tag));
            subset_3.apply_op(tag_to_keep(m_tag));
        }

        // COARSENING GRADUATION
        for (std::size_t level = max_level; level > 0; --level)
        {
            auto keep_subset = intersection(mesh[mesh_id_t::cells][level],
                                            mesh[mesh_id_t::all_cells][level - 1])
                            .on(level - 1);

            keep_subset.apply_op(maximum(m_tag));

            xt::xtensor_fixed<int, xt::xshape<dim>> stencil;
            for (std::size_t d = 0; d < dim; ++d)
            {
                stencil.fill(0);
                for (int s = -1; s <= 1; ++s)
                {
                    if (s != 0)
                    {
                        stencil[d] = s;
                        auto subset = intersection(mesh[mesh_id_t::cells][level],
                                                translate(mesh[mesh_id_t::cells][level - 1], stencil))
                                    .on(level - 1);
                        subset.apply_op(balance_2to1(m_tag, stencil));
                    }
                }
            }
        }

        // REFINEMENT GRADUATION
        for (std::size_t level = max_level; level > min_level; --level)
        {
            auto subset_1 = intersection(mesh[mesh_id_t::cells][level],
                                        mesh[mesh_id_t::cells][level]);

            subset_1.apply_op(extend(m_tag));

            static_nested_loop<dim, -1, 2>(
                [&](auto stencil) {

                auto subset = intersection(translate(mesh[mesh_id_t::cells][level], stencil),
                                        mesh[mesh_id_t::cells][level-1]).on(level);

                subset.apply_op(make_graduation(m_tag));

            });
        }

        for (std::size_t level = max_level; level > 0; --level)
        {
            auto keep_subset = intersection(mesh[mesh_id_t::cells][level],
                                            mesh[mesh_id_t::all_cells][level - 1])
                            .on(level - 1);

            keep_subset.apply_op(maximum(m_tag));
        }

        cl_type cell_list;

        for_each_interval(mesh[mesh_id_t::cells], [&](std::size_t level, const auto& interval, const auto& index_yz)
        {
            for (coord_index_t i = interval.start; i < interval.end; ++i)
            {
                if (m_tag[i + interval.index] & static_cast<int>(CellFlag::refine))
                {
                    static_nested_loop<dim - 1, 0, 2>([&](auto stencil) {
                        auto index = 2 * index_yz + stencil;
                        cell_list[level + 1][index].add_interval({2 * i, 2 * i + 2});
                    });
                }
                else if (m_tag[i + interval.index] & static_cast<int>(CellFlag::keep))
                {
                    cell_list[level][index_yz].add_point(i);
                }
                else
                {
                    cell_list[level-1][index_yz>>1].add_point(i>>1);
                }
            }
        });

        mesh_t new_mesh{cell_list, min_level, max_level};

        if (new_mesh == mesh)
        {
            return true;
        }

        field_type new_u{m_field.name(), new_mesh};
        new_u.fill(0.);

        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            auto subset = intersection(union_(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::proj_cells][level]),
                                            new_mesh[mesh_id_t::cells][level]);

            subset.apply_op(copy(new_u, m_field));
        }

        for_each_interval(mesh[mesh_id_t::cells], [&](std::size_t level, const auto& interval, const auto& index_yz)
        {
            for (coord_index_t i = interval.start; i < interval.end; ++i)
            {
                if (m_tag[i + interval.index] & static_cast<int>(CellFlag::refine))
                {
                    compute_prediction(level, interval_t{i, i + 1}, index_yz, m_field, new_u);
                }
            }
        });

        // START comment to the old fashion
        // which eliminates details of cells first deleted and then re-added by the refinement
        auto old_mesh = field_old.mesh();
        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            auto subset = intersection(intersection(old_mesh[mesh_id_t::cells][level],
                                            difference(new_mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level])),
                                            mesh[mesh_id_t::cells][level-1]).on(level);

            subset.apply_op(copy(new_u,  field_old));
        }
        // END comment

        m_field.mesh_ptr()->swap(new_mesh);
        field_old.mesh_ptr()->swap(new_mesh);

        std::swap(m_field.array(), new_u.array());
        std::swap(field_old.array(), new_u.array());

        return false;
    }


    template<class TField, class Func>
    auto make_MRAdapt(TField& field, Func&& update_bc_for_level)
    {
        return Adapt<TField, Func>(field, std::forward<Func>(update_bc_for_level));
    }
}