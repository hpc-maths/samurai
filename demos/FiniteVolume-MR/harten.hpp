#pragma once

#include <cmath>

#include <samurai/samurai.hpp>
#include <samurai/mr/mesh.hpp>
#include "criteria.hpp"

template <class Field>
bool harten(Field &u, Field &uold, double eps, double regularity, std::size_t ite, std::size_t global_iter)
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

    samurai::Field<Config, int, 1> tag{"tag", mesh};
    tag.array().fill(0);
    samurai::for_each_cell(mesh[samurai::MeshType::cells], [&](auto &cell) {
        tag[cell] = static_cast<int>(samurai::CellFlag::keep);
    });



    // //For an entropy criterion
    // samurai::Field<Config, double, 1> detail_entropy{"entropy detail", mesh};
    // samurai::Field<Config, double, 1> entropy{"entropy", mesh};
    // entropy.array().fill(0);
    // mesh.for_each_cell([&](auto &cell) {
    //     double gm = 1.4;

    //     auto rho = u[cell][0] + u[cell][1] + u[cell][2] + u[cell][3];
    //     auto qx  = u[cell][4] + u[cell][5] + u[cell][6] + u[cell][7];
    //     auto qy  = u[cell][8] + u[cell][9] + u[cell][10]+ u[cell][11];
    //     auto e   = u[cell][12]+ u[cell][13]+ u[cell][14]+ u[cell][15];

    //     // Computing the entropy with multiplicative constant 1 and additive constant 0
    //     auto p = (gm - 1.) * (e - .5 * (std::pow(qx, 2.) + std::pow(qy, 2.)) / rho);
    //     entropy[cell] = std::log(p / std::pow(rho, gm));
    // });
    // samurai::mr_projection(entropy);
    // entropy.update_bc();
    // samurai::mr_prediction(entropy);

    std::size_t save_at_ite = 61;
    samurai::mr_projection(u);

    // if (global_iter == save_at_ite)
    // {
    //     std::stringstream s;
    //     s << "u0_"<<ite;
    //     auto h5file = samurai::Hdf5(s.str().data());
    //     h5file.add_field_by_level(mesh, u);
    // }
    u.update_bc();

    // if (global_iter == save_at_ite)
    // {
    //     std::stringstream s;
    //     s << "u1_"<<ite;
    //     auto h5file = samurai::Hdf5(s.str().data());
    //     h5file.add_field_by_level(mesh, u);
    // }
    samurai::mr_prediction(u);
    // if (global_iter == save_at_ite)
    // {
    //     std::stringstream s;
    //     s << "u2_"<<ite;
    //     auto h5file = samurai::Hdf5(s.str().data());
    //     h5file.add_field_by_level(mesh, u);
    // }



    for (std::size_t level = min_level - 1; level < max_level - ite; ++level)
    {
        auto subset = intersection(mesh[samurai::MeshType::all_cells][level],
                                   mesh[samurai::MeshType::cells][level + 1])
                     .on(level);
        subset.apply_op(compute_detail(detail, u));

        // For entropy
        // subset.apply_op(compute_detail(detail_entropy, entropy));
    }


    // Affichage des details


    // for (std::size_t level = min_level; level <= max_level - ite; ++level)
    // {
    //     auto leaves = intersection(mesh[samurai::MeshType::cells][level],
    //                                mesh[samurai::MeshType::cells][level]);

    //     leaves([&](auto, auto &interval, auto) {

    //         auto k = interval[0]; // Logical index in x

    //         std::cout<<std::endl<<"Level = "<<level<<"  Interval = "<<k<<"   Detail = "<<detail(level, k)<<std::flush;

    //     });
    // }



    for (std::size_t level = min_level; level <= max_level - ite; ++level)
    {
        double exponent = dim * (max_level - level);



        double eps_l = std::pow(2., -exponent) * eps;


        // std::cout<<std::endl<<"level = "<<level<<"  exp = "<<exponent<<std::flush;

        double regularity_to_use = std::min(regularity, 3.0) + dim;

        auto subset_1 = samurai::intersection(mesh[samurai::MeshType::cells][level],
                                           mesh[samurai::MeshType::all_cells][level-1])
                       .on(level-1);


        subset_1.apply_op(to_coarsen_mr(detail, tag, eps_l, min_level)); // Derefinement
        subset_1.apply_op(to_refine_mr(detail, tag, (pow(2.0, regularity_to_use)) * eps_l, max_level)); // Refinement according to Harten

        // Entropy refinement
        // subset_1.apply_op(to_coarsen_mr(detail_entropy, tag, eps_l, min_level)); // Derefinement
        // subset_1.apply_op(to_refine_mr(detail_entropy, tag, (pow(2.0, regularity_to_use)) * eps_l, max_level)); // Refinement according to Harten
    }



    // std::cout<<std::endl<<"Tag after refinement"<<std::endl;
    // for (std::size_t level = min_level; level <= max_level - ite; ++level)
    // {
    //     auto leaves = intersection(mesh[samurai::MeshType::cells][level],
    //                                mesh[samurai::MeshType::cells][level]);

    //     leaves([&](auto, auto &interval, auto) {

    //         auto k = interval[0]; // Logical index in x

    //         std::cout<<std::endl<<"Level = "<<level<<"  Interval = "<<k<<"   Tag = "<<tag(level, k)<<std::flush;

    //     });
    // }
    // if(global_iter == save_at_ite)
    // {
    //     std::cout << "Au debut : u en -20, -19 " << u(7, interval_t{-20, -18}) << "\n";
    //     std::stringstream s;
    //     s << "tagify_before_enlarge_"<<ite;
    //     auto h5file = samurai::Hdf5(s.str().data());
    //     h5file.add_mesh(mesh);
    //     h5file.add_field(tag);
    //     h5file.add_field(detail);
    //     h5file.add_field(u);
    //     samurai::Field<Config> level_{"level", mesh};
    //     mesh.for_each_cell([&](auto &cell) {
    //         level_[cell] = static_cast<double>(cell.level);
    //     });
    //     h5file.add_field(level_);
    // }

    // if(global_iter == save_at_ite)
    // {
    //     std::stringstream s;
    //     s << "tagify_before_enlarge_by_level"<<ite;
    //     auto h5file = samurai::Hdf5(s.str().data());
    //     h5file.add_field_by_level(mesh, tag);
    // }

    for (std::size_t level = min_level; level <= max_level - ite; ++level)
    {
        auto subset_2 = intersection(mesh[samurai::MeshType::cells][level],
                                     mesh[samurai::MeshType::cells][level]);
        auto subset_3 = intersection(mesh[samurai::MeshType::cells_and_ghosts][level],
                                     mesh[samurai::MeshType::cells_and_ghosts][level]);

        subset_2.apply_op(samurai::enlarge(tag));
        subset_2.apply_op(samurai::keep_around_refine(tag));
        subset_3.apply_op(samurai::tag_to_keep(tag));
    }




    // std::cout<<std::endl<<"Tag after keepa around and tag to keep"<<std::endl;
    // for (std::size_t level = min_level; level <= max_level - ite; ++level)
    // {
    //     auto leaves = intersection(mesh[samurai::MeshType::cells][level],
    //                                mesh[samurai::MeshType::cells][level]);

    //     leaves([&](auto, auto &interval, auto) {

    //         auto k = interval[0]; // Logical index in x

    //         std::cout<<std::endl<<"Level = "<<level<<"  Interval = "<<k<<"   Tag = "<<tag(level, k)<<std::flush;

    //     });
    // }

    // if(global_iter == save_at_ite)
    // {
    //     std::stringstream s;
    //     s << "tagify_"<<ite;
    //     auto h5file = samurai::Hdf5(s.str().data());
    //     h5file.add_mesh(mesh);
    //     h5file.add_field(tag);
    //     h5file.add_field(detail);
    //     h5file.add_field(u);
    //     samurai::Field<Config> level_{"level", mesh};
    //     mesh.for_each_cell([&](auto &cell) {
    //         level_[cell] = static_cast<double>(cell.level);
    //     });
    //     h5file.add_field(level_);
    // }

    // if(global_iter == save_at_ite)
    // {
    //     std::stringstream s;
    //     s << "tagify_by_level"<<ite;
    //     auto h5file = samurai::Hdf5(s.str().data());
    //     h5file.add_field_by_level(mesh, tag);
    // }

    // COARSENING GRADUATION
    for (std::size_t level = max_level; level > 0; --level)
    {
        auto keep_subset = intersection(mesh[samurai::MeshType::cells][level],
                                        mesh[samurai::MeshType::all_cells][level - 1])
                        .on(level - 1);
        keep_subset.apply_op(maximum(tag));

        xt::xtensor_fixed<int, xt::xshape<dim>> stencil;
        for (std::size_t d = 0; d < dim; ++d)
        {
            stencil.fill(0);
            for (int s = -1; s <= 1; ++s)
            {
                if (s != 0)
                {
                    stencil[d] = s;
                    auto subset = intersection(mesh[samurai::MeshType::cells][level],
                                            translate(mesh[samurai::MeshType::cells][level - 1], stencil))
                                .on(level - 1);
                    subset.apply_op(balance_2to1(tag, stencil));
                }
            }
        }
    }





    // std::cout<<std::endl<<"Tag after balance_to_one"<<std::endl;
    // for (std::size_t level = min_level; level <= max_level - ite; ++level)
    // {
    //     auto leaves = intersection(mesh[samurai::MeshType::cells][level],
    //                                mesh[samurai::MeshType::cells][level]);

    //     leaves([&](auto, auto &interval, auto) {

    //         auto k = interval[0]; // Logical index in x

    //         std::cout<<std::endl<<"Level = "<<level<<"  Interval = "<<k<<"   Tag = "<<tag(level, k)<<std::flush;

    //     });
    // }

    // REFINEMENT GRADUATION
    for (std::size_t level = max_level; level > min_level; --level)
    {
        auto subset_1 = intersection(mesh[samurai::MeshType::cells][level],
                                    mesh[samurai::MeshType::cells][level]);

        subset_1.apply_op(extend(tag));

        samurai::static_nested_loop<dim, -1, 2>(
            [&](auto stencil) {

            auto subset = intersection(translate(mesh[samurai::MeshType::cells][level], stencil),
                                       mesh[samurai::MeshType::cells][level-1]).on(level);

            subset.apply_op(make_graduation(tag));

        });
    }




    // std::cout<<std::endl<<"Tag after make gradutation"<<std::endl;
    // for (std::size_t level = min_level; level <= max_level - ite; ++level)
    // {
    //     auto leaves = intersection(mesh[samurai::MeshType::cells][level],
    //                                mesh[samurai::MeshType::cells][level]);

    //     leaves([&](auto, auto &interval, auto) {

    //         auto k = interval[0]; // Logical index in x

    //         std::cout<<std::endl<<"Level = "<<level<<"  Interval = "<<k<<"   Tag = "<<tag(level, k)<<std::flush;

    //     });
    // }

    for (std::size_t level = max_level; level > 0; --level)
    {
        auto keep_subset = intersection(mesh[samurai::MeshType::cells][level],
                                        mesh[samurai::MeshType::all_cells][level - 1])
                        .on(level - 1);
        keep_subset.apply_op(maximum(tag));
    }




    // std::cout<<std::endl<<"Tag after maximum"<<std::endl;
    // for (std::size_t level = min_level; level <= max_level - ite; ++level)
    // {
    //     auto leaves = intersection(mesh[samurai::MeshType::cells][level],
    //                                mesh[samurai::MeshType::cells][level]);

    //     leaves([&](auto, auto &interval, auto) {

    //         auto k = interval[0]; // Logical index in x

    //         std::cout<<std::endl<<"Level = "<<level<<"  Interval = "<<k<<"   Tag = "<<tag(level, k)<<std::flush;

    //     });
    // }

    // if(global_iter == save_at_ite)
    // {
    //     std::stringstream s;
    //     s << "graduation_"<<ite;
    //     auto h5file = samurai::Hdf5(s.str().data());
    //     h5file.add_mesh(mesh);
    //     h5file.add_field(tag);
    //     h5file.add_field(u);
    //     samurai::Field<Config> level_{"level", mesh};
    //     mesh.for_each_cell([&](auto &cell) {
    //         level_[cell] = static_cast<double>(cell.level);
    //     });
    //     h5file.add_field(level_);

    // }

    // if(global_iter == save_at_ite)
    // {
    //     std::stringstream s;
    //     s << "graduation_by_level"<<ite;
    //     auto h5file = samurai::Hdf5(s.str().data());
    //     h5file.add_field_by_level(mesh, tag);
    // }

    samurai::CellList<dim, interval_t, max_refinement_level> cell_list;
    for (std::size_t level = min_level; level <= max_level; ++level)
    {
        auto level_cell_array = mesh[samurai::MeshType::cells][level];

        if (!level_cell_array.empty())
        {
            samurai::for_each_interval(level_cell_array, [&](const auto& interval, const auto& index_yz)
            {
                for (int i = interval.start; i < interval.end; ++i)
                {
                    // if (global_iter == save_at_ite & ite == 2)
                    // {
                    //     std::cout << "level: " << level << " interval " << interval << " i: " << i << " j: " << index_yz[0] << " tag: " << tag.array()[i + interval.index] << "\n";
                    // }
                    if (tag.array()[i + interval.index] & static_cast<int>(samurai::CellFlag::refine))
                    {
                        samurai::static_nested_loop<dim - 1, 0, 2>(
                            [&](auto stencil) {
                                auto index = 2 * index_yz + stencil;
                                cell_list[level + 1][index].add_point(2 * i);
                                cell_list[level + 1][index].add_point(2 * i + 1);
                                // if (global_iter == save_at_ite & ite == 2)
                                // {
                                //     std::cout << "Refine:" << index[0] << " " << 2*i << " " << 2*i+1 << "\n";
                                // }
                            });
                    }
                    else if (tag.array()[i + interval.index] & static_cast<int>(samurai::CellFlag::keep))
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

    samurai::Mesh<Config> new_mesh{cell_list, mesh.initial_mesh(), min_level, max_level};



    if (new_mesh == mesh)
    {
        return true;
    }

    Field new_u{u.name(), new_mesh, u.bc()};
    new_u.array().fill(0.);

    for (std::size_t level = min_level; level <= max_level; ++level)
    {
        auto subset = samurai::intersection(samurai::union_(mesh[samurai::MeshType::cells][level], mesh[samurai::MeshType::proj_cells][level]),
                                        new_mesh[samurai::MeshType::cells][level]);
        // auto subset = samurai::intersection(mesh[samurai::MeshType::all_cells][level],
        //                                  new_mesh[samurai::MeshType::cells][level]);
        subset.apply_op(copy(new_u, u));
    }



    // if(global_iter == save_at_ite)
    // {
    //     std::cout << "u en -20, -19 " << u(7, interval_t{-20, -18}) << "\n";
    //     std::cout << "new_u en -20, -19 " << new_u(7, interval_t{-20, -18}) << "\n";
    //     std::stringstream s;
    //     s << "copy_"<<ite;
    //     auto h5file = samurai::Hdf5(s.str().data());
    //     h5file.add_mesh(new_mesh);
    //     h5file.add_field(new_u);
    //     samurai::Field<Config> level_{"level", new_mesh};
    //     new_mesh.for_each_cell([&](auto &cell) {
    //         level_[cell] = static_cast<double>(cell.level);
    //     });
    //     h5file.add_field(level_);

    // }

    for (std::size_t level = min_level; level < max_level; ++level)
    {
        auto level_cell_array = mesh[samurai::MeshType::cells][level];


        if (!level_cell_array.empty())
        {

            samurai::for_each_interval(level_cell_array, [&](const auto& interval, const auto& index_yz)
            {
                // std::cout << "tag on level " << level << " " << tag(level, interval)<< "\n";
                for (int i = interval.start; i < interval.end; ++i)
                {
                    if (tag.array()[i + interval.index] & static_cast<int>(samurai::CellFlag::refine))
                    {

                        samurai::compute_prediction(level, interval_t{i, i + 1}, index_yz, u, new_u);

                        // std::cout<<std::endl<<"#### Predicting on new leaf at level "<<level<<" interval = "<<i<<std::flush;

                    }
                }
            });
        }
    }



    // if(global_iter == save_at_ite)
    // {
    //     std::cout << "after prediction\n";
    //     std::cout << "u en -20, -19 " << u(7, interval_t{-20, -18}) << "\n";
    //     std::cout << "new_u en -20, -19 " << new_u(7, interval_t{-20, -18}) << "\n";
    //     std::stringstream s;
    //     s << "prediction_"<<ite;
    //     auto h5file = samurai::Hdf5(s.str().data());
    //     h5file.add_mesh(new_mesh);
    //     h5file.add_field(new_u);
    //     samurai::Field<Config> level_{"level", new_mesh};
    //     new_mesh.for_each_cell([&](auto &cell) {
    //         level_[cell] = static_cast<double>(cell.level);
    //     });
    //     h5file.add_field(level_);

    // }

    // START comment to the old fashion
    // which eliminates details of cells first deleted and then re-added by the refinement
    auto old_mesh = uold.mesh();
    for (std::size_t level = min_level; level <= max_level; ++level)
    {
        auto subset = samurai::intersection(samurai::intersection(old_mesh[samurai::MeshType::cells][level],
                                         difference(new_mesh[samurai::MeshType::cells][level], mesh[samurai::MeshType::cells][level])),
                                         mesh[samurai::MeshType::cells][level-1]).on(level);

        subset.apply_op(copy(new_u, uold));
    }

    // if(global_iter == save_at_ite)
    // {
    //     std::cout << "after uold \n";
    //     std::cout << "u en -20, -19 " << u(7, interval_t{-20, -18}) << "\n";
    //     std::cout << "new_u en -20, -19 " << new_u(7, interval_t{-20, -18}) << "\n";

    //     std::stringstream s;
    //     s << "copy_uold_"<<ite;
    //     auto h5file = samurai::Hdf5(s.str().data());
    //     h5file.add_mesh(new_mesh);
    //     h5file.add_field(new_u);
    //     samurai::Field<Config> level_{"level", new_mesh};
    //     new_mesh.for_each_cell([&](auto &cell) {
    //         level_[cell] = static_cast<double>(cell.level);
    //     });
    //     h5file.add_field(level_);
    // }

    // END comment

    // std::cout << uold.mesh() << "\n";
    // std::cout << u.mesh() << "\n";
    // std::cout << new_u.mesh() << "\n";
    u.mesh_ptr()->swap(new_mesh);
    uold.mesh_ptr()->swap(new_mesh);

    // std::cout<<std::endl<<u.mesh()<<std::endl;
    // std::cout<<std::endl<<uold.mesh()<<std::endl;

    std::swap(u.array(), new_u.array());
    std::swap(uold.array(), new_u.array());


    // if(global_iter == save_at_ite)
    // {
    //     std::cout << "after swap \n";
    //     std::cout << "u en -20, -19 " << u(7, interval_t{-20, -18}) << "\n";
    //     std::cout << "new_u en -20, -19 " << new_u(7, interval_t{-20, -18}) << "\n";
    // }
    // if (global_iter == save_at_ite)   {
    //     std::stringstream s;
    //     s << "new_u_by_level_"<< ite;
    //     auto h5file = samurai::Hdf5(s.str().data());
    //     h5file.add_field_by_level(u.mesh(), u);
    // }

    // {
    //     std::stringstream s;
    //     s << "new_u_"<<ite;
    //     auto h5file = samurai::Hdf5(s.str().data());
    //     auto mesh = u.mesh();
    //     samurai::Field<Config> level_{"level", mesh};
    //     mesh.for_each_cell([&](auto &cell) {
    //         level_[cell] = static_cast<double>(cell.level);
    //     });
    //     h5file.add_mesh(mesh);
    //     h5file.add_field(u);
    //     h5file.add_field(level_);
    // }




    return false;
}




