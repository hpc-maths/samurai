#pragma once

#include <mure/mure.hpp>
#include <mure/mr/mesh.hpp>
#include "criteria.hpp"

template <class Config>
double max_detail(mure::Field<Config> &u)
{
    constexpr auto dim = Config::dim;
    using interval_t = typename Config::interval_t;
    auto mesh = u.mesh();
    std::size_t min_level = mesh.min_level(), max_level = mesh.max_level();
    mure::Field<Config> detail{"detail", mesh};
    mure::Field<Config, int> tag{"tag", mesh};
    tag.array().fill(static_cast<int>(mure::CellFlag::keep));

    mure::mr_projection(u);
    mure::mr_prediction(u); 
    u.update_bc();

    double max_detail = 0.0;

    for (std::size_t level = min_level - 1; level < max_level; ++level)   {
        auto subset = intersection(mesh[mure::MeshType::all_cells][level],
                                   mesh[mure::MeshType::cells][level + 1])
                                .on(level);
        subset.apply_op(level, compute_detail(detail, u), max_detail_mr(detail, max_detail));
    }

    return max_detail;
}


template <class Field>
bool coarsening(Field &u, double eps, std::size_t ite)
{
    using Config = typename Field::Config;
    using value_type = typename Field::value_type;
    constexpr auto size = Field::size;
    constexpr auto dim = Config::dim;
    constexpr auto max_refinement_level = Config::max_refinement_level;
    using interval_t = typename Config::interval_t;

    auto mesh = u.mesh();
    std::size_t min_level = mesh.min_level(), max_level = mesh.max_level();

    Field detail{"detail", mesh};

    mure::Field<Config, int, 1> tag{"tag", mesh};
    tag.array().fill(0);
    mesh.for_each_cell([&](auto &cell) {
        tag[cell] = static_cast<int>(mure::CellFlag::keep);
    });
    //std::cout<<std::endl<<"Coarsening "<<ite<<std::flush;
    mure::mr_projection(u);
    // std::cout<<std::endl<<"Calling in coarsesnin"<<std::flush;
    u.update_bc(ite);
    mure::mr_prediction(u);


    // What are the data it uses at min_level - 1 ???
    for (std::size_t level = min_level - 1; level < max_level - ite; ++level)   {
        auto subset = intersection(mesh[mure::MeshType::all_cells][level],
                                   mesh[mure::MeshType::cells][level + 1])
                     .on(level);
        subset.apply_op(level, compute_detail(detail, u));
    }



    // AGAIN I DONT KNOW WHAT min_level - 1 is
    for (std::size_t level = min_level; level <= max_level - ite; ++level)
    {
        int exponent = dim * (level - max_level);

        auto eps_l = std::pow(2, exponent) * eps;

        // COMPRESSION

        auto subset_1 = mure::intersection(mesh[mure::MeshType::cells][level],
                                           mesh[mure::MeshType::all_cells][level-1])
                       .on(level-1);


        // This operations flags the cells to coarsen
        subset_1.apply_op(level, to_coarsen_mr(detail, tag, eps_l, min_level));

        auto subset_2 = intersection(mesh[mure::MeshType::cells][level],
                                     mesh[mure::MeshType::cells][level]);
        auto subset_3 = intersection(mesh[mure::MeshType::cells_and_ghosts][level],
                                     mesh[mure::MeshType::cells_and_ghosts][level]);

        subset_2.apply_op(level, mure::enlarge(tag, mure::CellFlag::keep));
        subset_3.apply_op(level, mure::tag_to_keep(tag));
    }

    //h5file.add_field(tag);


    // FROM NOW ON LOIC HAS TO EXPLAIN

    for (std::size_t level = max_level; level > 0; --level)
    {
        auto keep_subset = intersection(mesh[mure::MeshType::cells][level],
                                        mesh[mure::MeshType::all_cells][level - 1])
                          .on(level - 1);
        keep_subset.apply_op(level - 1, maximum(tag));

        xt::xtensor_fixed<int, xt::xshape<dim>> stencil;
        for (std::size_t d = 0; d < dim; ++d)
        {
            stencil.fill(0);
            for (int s = -1; s <= 1; ++s)
            {
                if (s != 0)
                {
                    stencil[d] = s;
                    auto subset = intersection(mesh[mure::MeshType::cells][level],
                                               translate(mesh[mure::MeshType::cells][level - 1], stencil))
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
    {
        return true;
    }

    Field new_u{u.name(), new_mesh, u.bc()};

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