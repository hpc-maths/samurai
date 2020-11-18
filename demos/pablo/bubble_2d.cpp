#include <iostream>

#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>

#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>

int main()
{
    constexpr std::size_t nb_bubbles = 10;
    constexpr std::size_t dim = 2;
    constexpr std::size_t start_level = 3;

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
    double Dt = 0.5;

    for(std::size_t nite = 0; nite < 50; ++nite)
    {
        t += Dt;
        // bb_xcenter = bb0_xcenter + aa*xt::cos(omega*t);
        bb_ycenter = bb_ycenter + Dt*dy;

        std::cout << "iteration " << nite << std::endl;

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

                    auto corner = cell.first_corner();
                    auto center = cell.center;
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
                                if (cell.level < 9)
                                {
                                    tag[cell] = static_cast<int>(samurai::CellFlag::refine);
                                }
                                inside = true;
                            }
                        }
                    }
                    ib++;
                }

                // if (cell.level > 0 && !inside)
                // {
                //     tag[cell] = static_cast<int>(samurai::CellFlag::coarsen);
                // }
            });

            // for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
            // {
            //     auto subset = samurai::intersection(mesh[level], mesh[level])
            //                  .on(level - 1);
            //     subset([&](const auto& interval, const auto& index)
            //     {
            //         auto i = interval;
            //         auto j = index[0];
            //         xt::xtensor<bool, 1> mask = (tag(level,     2*i,     2*j) & static_cast<int>(samurai::CellFlag::keep))
            //                                   | (tag(level, 2*i + 1,     2*j) & static_cast<int>(samurai::CellFlag::keep))
            //                                   | (tag(level,     2*i, 2*j + 1) & static_cast<int>(samurai::CellFlag::keep))
            //                                   | (tag(level, 2*i + 1, 2*j + 1) & static_cast<int>(samurai::CellFlag::keep));

            //         xt::masked_view(tag(level,     2*i,     2*j), mask) |= static_cast<int>(samurai::CellFlag::keep);
            //         xt::masked_view(tag(level, 2*i + 1,     2*j), mask) |= static_cast<int>(samurai::CellFlag::keep);
            //         xt::masked_view(tag(level,     2*i, 2*j + 1), mask) |= static_cast<int>(samurai::CellFlag::keep);
            //         xt::masked_view(tag(level, 2*i + 1, 2*j + 1), mask) |= static_cast<int>(samurai::CellFlag::keep);
            //     });
            // }


            // graduation
            for (std::size_t level = mesh.max_level(); level > 1; --level)
            {
                xt::xtensor_fixed<int, xt::xshape<4, dim>> stencil{{1, 1}, {-1, -1}, {-1, 1}, {1, -1}};

                for(std::size_t i = 0; i < stencil.shape()[0]; ++i)
                {
                    auto s = xt::view(stencil, i);
                    auto subset = samurai::intersection(samurai::translate(mesh[level], s),
                                                    mesh[level - 1])
                                .on(level);

                    subset([&](const auto& interval, const auto& index)
                    {
                        auto j_f = index[0];
                        auto i_f = interval.even_elements();

                        if (i_f.is_valid())
                        {
                            auto mask = tag(level, i_f  - s[0], j_f - s[1]) & static_cast<int>(samurai::CellFlag::refine);
                            auto i_c = i_f >> 1;
                            auto j_c = j_f >> 1;
                            xt::masked_view(tag(level - 1, i_c, j_c), mask) |= static_cast<int>(samurai::CellFlag::refine);
                        }

                        i_f = interval.odd_elements();
                        if (i_f.is_valid())
                        {
                            auto mask = tag(level, i_f  - s[0], j_f - s[1]) & static_cast<int>(samurai::CellFlag::refine);
                            auto i_c = i_f >> 1;
                            auto j_c = j_f >> 1;
                            xt::masked_view(tag(level - 1, i_c, j_c), mask) |= static_cast<int>(samurai::CellFlag::refine);
                        }
                    });
                }
            }


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
                    // else
                    // {
                    //     cell_list[level-1][index_yz>>1].add_point(i>>1);
                    // }
                }
            });

            mesh = {cell_list, true};
        }
        std::stringstream str;
        str << "bubble_2d_" << nite;
        samurai::save(str.str().data(), mesh);
    }
    return 0;
}