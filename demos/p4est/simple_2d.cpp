#include <iostream>
#include <sstream>
#include <chrono>

#include <cxxopts.hpp>

#include <mure/mure.hpp>

/// Timer used in tic & toc
auto tic_timer = std::chrono::high_resolution_clock::now();

/// Launching the timer
void tic()
{
    tic_timer = std::chrono::high_resolution_clock::now();
}

/// Stopping the timer and returning the duration in seconds
double toc()
{
    const auto toc_timer = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> time_span = toc_timer - tic_timer;
    return time_span.count();
}

template<class config>
void refine_1(mure::Mesh<config> &mesh, std::size_t max_level)
{
    constexpr auto dim = config::dim;

    for (std::size_t l=0; l<max_level; ++l)
    {
        mure::Field<config, bool> cell_tag{"tag", mesh};
        cell_tag.array().fill(false);
        mesh.for_each_cell([&](auto &cell) {
                auto corner = cell.first_corner();
                auto x = corner[0];
                auto y = corner[1];

                if (cell.level < max_level)
                {
                    if (x < 0.25)
                    {
                        cell_tag[cell] = true;
                    }
                    if (x == 0.75 and y == 0.75)
                    {
                        cell_tag[cell] = true;
                    }
                }
        });

        mure::CellList<config> cell_list;
        for (std::size_t level = 0; level <= max_level; ++level)
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
                        if (cell_tag.array()[i + interval.index])
                        {
                            mure::static_nested_loop<dim - 1, 0, 2>(
                                [&](auto stencil) {
                                    auto index = 2 * index_yz + stencil;
                                    cell_list[level + 1][index].add_point(2 *
                                                                            i);
                                    cell_list[level + 1][index].add_point(
                                        2 * i + 1);
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

        mesh = mure::Mesh<config>{cell_list};
    }
}

template<class config>
void refine_2(mure::Mesh<config> &mesh, std::size_t max_level)
{
    constexpr auto dim = config::dim;
    mure::Field<config, bool> cell_tag{"tag", mesh};
    cell_tag.array().fill(false);

    for (std::size_t l=0; l<max_level; ++l)
    {
        // {
        //     std::stringstream s;
        //     s << "simple_2d_ite_" << l;
        //     auto h5file = mure::Hdf5(s.str().data());
        //     h5file.add_mesh(mesh);
        //     mure::Field<config> level_{"level", mesh};
        //     mesh.for_each_cell([&](auto &cell) { level_[cell] = static_cast<double>(cell.level); });
        //     h5file.add_field(level_);
        //     h5file.add_field(cell_tag);
        // }

        mesh.for_each_cell([&](auto &cell) {
                auto corner = cell.first_corner();
                auto x = corner[0];
                auto y = corner[1];

                cell_tag[cell] = false;
                if (cell.level < max_level)
                {
                    if (x < 0.25)
                    {
                        cell_tag[cell] = true;
                    }
                    if (x == 0.75 and y == 0.75)
                    {
                        cell_tag[cell] = true;
                    }
                }
        });

        for (std::size_t level = max_level; level > 1; --level)
        {
            xt::xtensor_fixed<int, xt::xshape<dim>> stencil;

            for (int sy = -1; sy <= 1; ++sy)
            {
                stencil[1] = sy;
                for (int sx = -1; sx <= 1; ++sx)
                {
                    stencil[0] = sx;
                    if ((sx == 0 and sy != 0) or (sx != 0 and sy == 0))
                    // if (sx != 0 and sy != 0)
                    {
                        auto subset =
                            mure::intersection(
                                mure::translate(mesh[mure::MeshType::cells][level], stencil),
                                mesh[mure::MeshType::cells][level - 1])
                                .on(level);

                        subset([&](auto & index, auto &interval, auto&)
                        {
                            auto i_f = interval[0];
                            i_f.step = 2;
                            auto j_f = index[0];

                            auto i_c = interval[0] >> 1;
                            auto j_c = index[0] >> 1;

                            // std::cout << "stencil: "<< stencil << "\n";
                            // std::cout << level << " " << i_f << " " << j_f << "\n";
                            // std::cout << level-1 << " " << i_c << " " << j_c << "\n";
                            cell_tag(level - 1, i_c, j_c) |= cell_tag(level, i_f  - stencil[0], j_f - stencil[1]);
                        });
                    }
                }
            }
        }

        mure::CellList<config> cell_list;
        for (std::size_t level = 0; level <= max_level; ++level)
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
                        if (cell_tag.array()[i + interval.index])
                        {
                            mure::static_nested_loop<dim - 1, 0, 2>(
                                [&](auto stencil) {
                                    auto index = 2 * index_yz + stencil;
                                    cell_list[level + 1][index].add_point(2 *
                                                                            i);
                                    cell_list[level + 1][index].add_point(
                                        2 * i + 1);
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

        mesh = mure::Mesh<config>{cell_list};
        cell_tag.array().resize({mesh.nb_cells(mure::MeshType::all_cells)});
    }
}

int main(int argc, char *argv[])
{
    constexpr size_t dim = 2;
    using Config = mure::MRConfig<dim>;
    mure::CellList<Config> cell_list;

    cxxopts::Options options("simple_2d", "simple 2d p4est examples");

    options.add_options()
                       ("max_level", "maximum level", cxxopts::value<std::size_t>()->default_value("8"));
                    //    ("test", "test case (1: gaussian, 2: diamond, 3: circle)", cxxopts::value<std::size_t>()->default_value("1"))
                    //    ("log", "log level", cxxopts::value<std::string>()->default_value("warning"))
                    //    ("h, help", "Help");
    auto result = options.parse(argc, argv);

    std::size_t max_level = result["max_level"].as<std::size_t>();

    cell_list[1][{0}].add_interval({1, 2});
    cell_list[1][{1}].add_interval({0, 1});

    cell_list[2][{0}].add_interval({0, 2});
    cell_list[2][{1}].add_interval({0, 2});
    cell_list[2][{2}].add_interval({2, 4});
    cell_list[2][{3}].add_interval({2, 4});

    mure::Mesh<Config> mesh_1(cell_list);
    std::cout << "nb_cells: " << mesh_1.nb_cells(mure::MeshType::cells) << "\n";

    tic();
    refine_1(mesh_1, max_level);
    auto duration = toc();
    std::cout << "Version 1: " << duration << "s" << std::endl;toc();
    std::cout << "nb_cells: " << mesh_1.nb_cells(mure::MeshType::cells) << "\n";
    
    mure::Mesh<Config> mesh_2(cell_list);

    tic();
    refine_2(mesh_2, max_level);
    duration = toc();
    std::cout << "Version 2: " << duration << "s" << std::endl;toc();
    std::cout << "nb_cells: " << mesh_2.nb_cells(mure::MeshType::cells) << "\n";

    std::stringstream s;
    s << "simple_2d";
    auto h5file = mure::Hdf5(s.str().data());
    h5file.add_mesh(mesh_2);
    mure::Field<Config> level_{"level", mesh_2};
    mesh_2.for_each_cell([&](auto &cell) { level_[cell] = static_cast<double>(cell.level); });
    h5file.add_field(level_);
    return 0;
}