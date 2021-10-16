// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <iostream>

#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>

#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/subset/subset_op.hpp>

template <std::size_t dim>
void remove_intersection(samurai::CellArray<dim>& ca)
{
    auto min_level = ca.min_level();
    auto max_level = ca.max_level();
    std::size_t ite = 0;

    while(true)
    {
        // std::cout << "Iteration for remove intersection: " << ite++ << "\n";
        auto tag = samurai::make_field<bool, 1>("tag", ca);
        tag.fill(false);

        for(std::size_t level = min_level + 1; level <= max_level; ++level)
        {
            for(std::size_t level_below = min_level; level_below < level; ++level_below)
            {
                auto set = samurai::intersection(ca[level], ca[level_below]).on(level_below);
                set([&](const auto& i, const auto& index)
                {
                    tag(level_below, i, index[0]) = true;
                });
            }
        }

        samurai::CellList<dim> cl;
        samurai::for_each_cell(ca, [&](auto cell)
        {
            auto i = cell.indices[0];
            auto j = cell.indices[1];
            if (tag[cell])
            {
                cl[cell.level + 1][{2*j}].add_interval({2*i, 2*i+2});
                cl[cell.level + 1][{2*j + 1}].add_interval({2*i, 2*i+2});
            }
            else
            {
                cl[cell.level][{j}].add_point(i);
            }
        });
        samurai::CellArray<dim> new_ca = {cl, true};

        if(new_ca == ca)
        {
            break;
        }

        std::swap(ca, new_ca);
    }
}

template <std::size_t dim>
void make_graduation(samurai::CellArray<dim>& ca)
{
    auto min_level = ca.min_level();
    auto max_level = ca.max_level();
    std::size_t ite = 0;
    // xt::xtensor_fixed<int, xt::xshape<4, dim>> stencil{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    xt::xtensor_fixed<int, xt::xshape<4, dim>> stencil{{1, 1}, {-1, -1}, {-1, 1}, {1, -1}};
    while(true)
    {
        // std::cout << "Iteration for graduation: " << ite++ << "\n";
        auto tag = samurai::make_field<bool, 1>("tag", ca);
        tag.fill(false);

        for(std::size_t level = min_level + 2; level <= max_level; ++level)
        {
            for(std::size_t level_below = min_level; level_below < level - 1; ++level_below)
            {
                for(std::size_t i = 0; i < stencil.shape()[0]; ++i)
                {
                    auto s = xt::view(stencil, i);
                    auto set = samurai::intersection(samurai::translate(ca[level], s), ca[level_below]).on(level_below);
                    set([&](const auto& i, const auto& index)
                    {
                        tag(level_below, i, index[0]) = true;
                    });
                }
            }
        }

        samurai::CellList<dim> cl;
        samurai::for_each_cell(ca, [&](auto cell)
        {
            auto i = cell.indices[0];
            auto j = cell.indices[1];
            if (tag[cell])
            {
                cl[cell.level + 1][{2*j}].add_interval({2*i, 2*i+2});
                cl[cell.level + 1][{2*j + 1}].add_interval({2*i, 2*i+2});
            }
            else
            {
                cl[cell.level][{j}].add_point(i);
            }
        });
        samurai::CellArray<dim> new_ca = {cl, true};

        if(new_ca == ca)
        {
            break;
        }

        std::swap(ca, new_ca);
    }
}

int main()
{
    constexpr std::size_t nb_bubbles = 10;
    constexpr std::size_t dim = 2;
    constexpr std::size_t start_level = 4;
    constexpr std::size_t min_level = 1;
    constexpr std::size_t max_level = 9;

    using container_t = xt::xtensor_fixed<double, xt::xshape<nb_bubbles>>;

    container_t bb_xcenter, bb_ycenter;
    container_t bb0_xcenter, bb0_ycenter;
    container_t bb_radius;
    container_t dy;
    container_t omega;
    container_t aa;

    bb_xcenter = 0.8*xt::random::rand<double>({nb_bubbles}) + 0.1;
    bb0_xcenter = bb_xcenter;
    bb_ycenter = xt::random::rand<double>({nb_bubbles}) - 0.5;
    bb0_ycenter = bb_ycenter;
    bb_radius = 0.1*xt::random::rand<double>({nb_bubbles}) + 0.02;
    dy = 0.005 + 0.05*xt::random::rand<double>({nb_bubbles});
    omega = 0.5*xt::random::rand<double>({nb_bubbles});
    aa = 0.15*xt::random::rand<double>({nb_bubbles});

    samurai::CellArray<dim> mesh;

    mesh[start_level] = {start_level, samurai::Box<int, dim>({0, 0}, {1<<start_level, 1<<start_level})};

    double t0 = 0;
    double t = t0;
    double Tf = 100;
    double Dt = .5;

    std::size_t ite = 0;
    while( t < Tf)
    {
        t += Dt;
        bb_xcenter = bb0_xcenter + aa*xt::cos(omega*t);
        bb_ycenter = bb_ycenter + Dt*dy;

        std::cout << fmt::format("iteration -> {} t -> {}", ite, t) << std::endl;

        for(std::size_t rep = 0; rep < 10; ++rep)
        {
            auto tag = samurai::make_field<int, 1>("tag", mesh);
            tag.fill(static_cast<int>(samurai::CellFlag::keep));

            samurai::for_each_cell(mesh, [&](auto cell)
            {
                bool inside = false;
                std::size_t ib = 0;

                while (!inside && ib < nb_bubbles)
                {
                    double xc = bb_xcenter[ib];
                    double yc = bb_ycenter[ib];
                    double radius = bb_radius[ib];

                    auto corner = cell.corner();
                    auto center = cell.center();
                    double dx = cell.length;

                    for(std::size_t i = 0; i < 2; ++i)
                    {
                        double x = corner[0] + i*dx;
                        for(std::size_t j = 0; j < 2; ++j)
                        {
                            double y = corner[1] + j*dx;
                            if (((!inside) && (std::pow((x - xc), 2.0) + pow((y - yc), 2.0) <= 1.25*std::pow(radius, 2.0)
                                        &&  std::pow((x - xc), 2.0) + pow((y - yc), 2.0) >= 0.75*std::pow(radius, 2.0)))
                            || ((!inside) && (std::pow((center[0] - xc), 2.0) + std::pow((center[1] - yc), 2.0) <= 1.25*std::pow(radius, 2.0)
                                        &&  std::pow((center[0] - xc), 2.0) + std::pow((center[1] - yc), 2.0) >= 0.75*std::pow(radius, 2.0))))
                            {
                                if (cell.level < max_level)
                                {
                                    tag[cell] = static_cast<int>(samurai::CellFlag::refine);
                                }
                                inside = true;
                            }
                        }
                    }
                    ib++;
                }

                if (cell.level > min_level && !inside)
                {
                    tag[cell] = static_cast<int>(samurai::CellFlag::coarsen);
                }
            });

            samurai::CellList<dim> cell_list;

            samurai::for_each_interval(mesh, [&](std::size_t level, const auto& interval, const auto& index_yz)
            {
                for (int i = interval.start; i < interval.end; ++i)
                {
                    if (tag[i + interval.index] & static_cast<int>(samurai::CellFlag::refine))
                    {
                        samurai::static_nested_loop<dim - 1, 0, 2>([&](auto stencil)
                        {
                            auto index = 2 * index_yz + stencil;
                            cell_list[level + 1][index].add_interval({2 * i, 2 * i + 2});
                        });
                    }
                    else if (tag[i + interval.index] & static_cast<int>(samurai::CellFlag::keep))
                    {
                        cell_list[level][index_yz].add_point(i);
                    }
                    else
                    {
                        cell_list[level-1][index_yz>>1].add_point(i>>1);
                    }
                }
            });

            mesh = {cell_list, true};

            remove_intersection(mesh);
            make_graduation(mesh);
        }
        samurai::save(fmt::format("bubble_2d_{}", ite++), mesh);
    }
    return 0;
}