#pragma once

#include <mure/mure.hpp>
#include <mure/mr/mesh.hpp>
#include "criteria.hpp"

#include "../FiniteVolume/criteria.hpp"


template <class Field>
bool refinement(Field &u, double eps, std::size_t ite)
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

    mure::mr_projection(u);
    mure::mr_prediction(u); 
    u.update_bc();

    typename std::conditional<size == 1,
                              xt::xtensor_fixed<value_type, xt::xshape<max_refinement_level + 1>>,
                              xt::xtensor_fixed<value_type, xt::xshape<max_refinement_level + 1, size>>
                             >::type max_detail;
    max_detail.fill(std::numeric_limits<value_type>::min());

    for (std::size_t level = min_level - 1; level < max_level - ite; ++level)   {
        auto subset = intersection(mesh[mure::MeshType::all_cells][level],
                                   mesh[mure::MeshType::cells][level + 1])
                                .on(level);
        subset.apply_op(level, compute_detail(detail, u),
                               compute_max_detail(detail, max_detail));
    }

    // std::stringstream s;
    // s << "refinement_"<<ite;
    // auto h5file = mure::Hdf5(s.str().data());
    // h5file.add_mesh(mesh);
    // h5file.add_field(detail);
    // h5file.add_field(u);

    // Look carefully at how much of this we have to do...
    for (std::size_t level = min_level; level <= max_level - ite; ++level)
    {

        // int exponent = dim * (level - max_level);

        // auto eps_l = std::pow(2, exponent) * eps;

        // // HARTEN HEURISTICS
        
        // auto subset = mure::intersection(mesh[mure::MeshType::cells][level],
        //                                  mesh[mure::MeshType::cells][level])
        //                .on(level);
        
        // subset.apply_op(level, to_refine_mr(detail, max_detail, tag, 32 * eps_l, max_level));


        int exponent = dim * (level - max_level);
        auto eps_l = std::pow(2, exponent) * eps;

        // HARTEN HEURISTICS
        
        auto subset = mure::intersection(mesh[mure::MeshType::cells][level],
                                         mesh[mure::MeshType::all_cells][level-1])
                       .on(level-1);
        
        //subset.apply_op(level, to_refine_mr(detail, max_detail, tag, 32 * eps_l, max_level));

        double regularity_of_the_function = 100.0;

        double regularity = std::min(regularity_of_the_function, 3.0) + dim;

        subset.apply_op(level, to_refine_mr(detail, max_detail, tag, (pow(2.0, regularity)) * eps_l, max_level));

        //subset.apply_op(level, to_refine_mr_BH(detail, max_detail, tag, 32 * eps_l, max_level));


    }

    //h5file.add_field(tag);


    for (std::size_t level = max_level; level > min_level; --level)
    {
        auto subset_1 = intersection(mesh[mure::MeshType::cells][level],
                                     mesh[mure::MeshType::cells][level]);

        subset_1.apply_op(level, extend(tag));
        
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

                    auto subset = intersection(translate(mesh[mure::MeshType::cells][level], stencil),
                                              mesh[mure::MeshType::cells][level-1])
                                .on(level);

                    subset.apply_op(level, make_graduation(tag));
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
            level_cell_array.for_each_interval_in_x([&](auto const
                                                                    &index_yz,
                                                                auto const
                                                                    &interval) {
                for (int i = interval.start; i < interval.end; ++i)
                {
                    if (tag.array()[i + interval.index] & static_cast<int>(mure::CellFlag::refine))
                    {
                        mure::static_nested_loop<dim - 1, 0, 2>(
                            [&](auto stencil) {
                                auto index = 2 * index_yz + stencil;
                                cell_list[level + 1][index].add_point(2 * i);
                                cell_list[level + 1][index].add_point(2 * i + 1);
                            });
                    }
                    else
                    {
                        cell_list[level][index_yz].add_point(i);
                    }
                }
            });
        }
    }
    mure::Mesh<Config> new_mesh{cell_list, mesh.initial_mesh(),
                            min_level, max_level};

    if (new_mesh == mesh)
        return true;

    Field new_u{u.name(), new_mesh, u.bc()};

    for (std::size_t level = min_level; level <= max_level; ++level)
    {
        auto subset = mure::intersection(mesh[mure::MeshType::all_cells][level],
                                   new_mesh[mure::MeshType::cells][level]);
        subset.apply_op(level, copy(new_u, u));
    }

    for (std::size_t level = min_level; level < max_level; ++level)
    {
        auto level_cell_array = mesh[mure::MeshType::cells][level];

        if (!level_cell_array.empty())
        {
            level_cell_array.for_each_interval_in_x(
                [&](auto const &index_yz, auto const &interval) {
                    for (int i = interval.start; i < interval.end; ++i)
                    {
                        if (tag.array()[i + interval.index] &
                            static_cast<int>(mure::CellFlag::refine))
                        {
                            mure::compute_prediction(level, interval_t{i, i + 1},
                                                     index_yz, u, new_u);
                        }
                    }
                });
        }
    }

    u.mesh_ptr()->swap(new_mesh);
    std::swap(u.array(), new_u.array());

    return false;

 }