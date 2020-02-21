#pragma once

#include <mure/mure.hpp>
#include "criteria.hpp"

template <class Config>
bool coarsening(mure::Field<Config> &u, std::size_t ite, std::size_t nt)
{
    constexpr auto dim = Config::dim;
    using interval_t = typename Config::interval_t;

    auto mesh = u.mesh();
    std::size_t min_level = mesh.min_level(), max_level = mesh.max_level();
    mure::Field<Config> grad{"grad", mesh};
    mure::Field<Config, int> tag{"tag", mesh};
    tag.array().fill(static_cast<int>(mure::CellFlag::keep));

    mure::mr_projection(u);
    mure::amr_prediction(u);
    u.update_bc();

    for (std::size_t level = min_level; level <= max_level; ++level)
    {
        auto subset = mure::intersection(mesh[mure::MeshType::cells][level],
                                        mesh[mure::MeshType::cells][level]);
        subset.apply_op(level, compute_gradient(u, grad));
    }

    for (std::size_t level = min_level; level <= max_level; ++level)
    {
        auto subset = mure::intersection(mesh[mure::MeshType::cells][level],
                                         mesh[mure::MeshType::all_cells][level-1]).on(level-1);
        subset.apply_op(level, to_coarsen_amr(grad, tag, min_level));
    }

    for (std::size_t level = max_level; level > 0; --level)
    {
        auto keep_subset =
            intersection(mesh[mure::MeshType::cells][level],
                         mesh[mure::MeshType::all_cells][level - 1])
                .on(level - 1);

        keep_subset.apply_op(level - 1, maximum(tag));

        xt::xtensor_fixed<int, xt::xshape<dim>> stencil;
        for (std::size_t d = 0; d < dim; ++d)
        {
            for (std::size_t d1 = 0; d1 < dim; ++d1)
                stencil[d1] = 0;
            for (int s = -1; s <= 1; ++s)
            {
                if (s != 0)
                {
                    stencil[d] = s;

                    auto subset =
                        intersection(
                            mesh[mure::MeshType::cells][level],
                            translate(
                                mesh[mure::MeshType::cells][level - 1], stencil))
                            .on(level - 1);

                    subset.apply_op(level - 1, balance_2to1(tag, stencil));
                }
            }
        }
    }

    mure::CellList<Config> cell_list;
    for (std::size_t level = min_level; level <= max_level; ++level)
    {
        auto level_cell_array = mesh[mure::MeshType::cells][level];

        if (!level_cell_array.empty())
        {
            level_cell_array.for_each_interval_in_x([&](auto const &index_yz, auto const &interval) {
                for (int i = interval.start; i < interval.end; ++i)
                {
                    if (tag.array()[i + interval.index] & static_cast<int>(mure::CellFlag::keep))
                    {
                        cell_list[level][index_yz].add_point(i);
                    }
                    else
                    {
                        cell_list[level-1][index_yz>>1].add_point(i>>1);
                    }
                }
            });
        }
    }
    mure::Mesh<Config> new_mesh{cell_list, mesh.initial_mesh(),
                            min_level, max_level};

    if (new_mesh == mesh)
        return true;

    mure::Field<Config> new_u{u.name(), new_mesh, u.bc()};

    for (std::size_t level = min_level; level <= max_level; ++level)
    {
        auto subset = mure::intersection(mesh[mure::MeshType::all_cells][level],
                                   new_mesh[mure::MeshType::cells][level]);
        subset.apply_op(level, copy(new_u, u));
    }

    u.mesh_ptr()->swap(new_mesh);
    std::swap(u.array(), new_u.array());
    return false;
}
