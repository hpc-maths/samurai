#include <iostream>
#include <sstream>
#include <chrono>

#include <cxxopts.hpp>

#include <mure/cell_list.hpp>
#include <mure/cell_array.hpp>
#include <mure/field.hpp>
#include <mure/hdf5.hpp>

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

template<class mesh_t>
void refine_1(mesh_t& mesh, std::size_t max_level)
{
    constexpr std::size_t dim = mesh_t::dim;
    using cl_type = typename mesh_t::cl_type;
    using coord_index_t = typename mesh_t::interval_t::coord_index_t;

    for (std::size_t l = 0; l < max_level; ++l)
    {
        auto cell_tag = mure::make_field<bool, 1>("tag", mesh);
        cell_tag.fill(false);

        mure::for_each_cell(mesh, [&](auto cell)
        {
            auto corner = cell.first_corner();
            auto x = corner[0];
            auto y = corner[1];

            if (cell.level < max_level)
            {
                if (x < 0.25 or (x == 0.75 and y == 0.75))
                {
                    cell_tag[cell] = true;
                }
            }
        });

        cl_type cl;
        mure::for_each_interval(mesh, [&](std::size_t level, const auto& interval, const auto& index_yz)
        {
            for (coord_index_t i = interval.start; i < interval.end; ++i)
            {
                if (cell_tag[i + interval.index])
                {
                    mure::static_nested_loop<dim - 1, 0, 2>([&](auto stencil)
                    {
                        auto index = 2 * index_yz + stencil;
                        cl[level + 1][index].add_interval({2 * i, 2 * i + 2});
                    });
                }
                else
                {
                    cl[level][index_yz].add_point(i);
                }
            }
        });

        mesh = {cl};
    }
}

template<class mesh_t>
void refine_2(mesh_t& mesh, std::size_t max_level)
{
    constexpr std::size_t dim = mesh_t::dim;
    using cl_type = typename mesh_t::cl_type;
    using coord_index_t = typename mesh_t::interval_t::coord_index_t;

    for (std::size_t l = 0; l < max_level; ++l)
    {
        auto cell_tag = mure::make_field<bool, 1>("tag", mesh);
        cell_tag.fill(false);

        mure::for_each_cell(mesh, [&](auto cell)
        {
            auto corner = cell.first_corner();
            auto x = corner[0];
            auto y = corner[1];

            if (cell.level < max_level)
            {
                if (x < 0.25 or (x == 0.75 and y == 0.75))
                {
                    cell_tag[cell] = true;
                }
            }
        });

        // graduation
        for (std::size_t level = max_level; level > 1; --level)
        {
            // xt::xtensor_fixed<int, xt::xshape<4, dim>> stencil{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            xt::xtensor_fixed<int, xt::xshape<4, dim>> stencil{{1, 1}, {-1, -1}, {-1, 1}, {1, -1}};

            for(std::size_t i = 0; i < stencil.shape()[0]; ++i)
            {
                auto s = xt::view(stencil, i);
                auto subset = mure::intersection(mure::translate(mesh[level], s),
                                                 mesh[level - 1])
                             .on(level);

                subset([&](const auto& interval, const auto& index)
                {
                    auto j_f = index[0];
                    auto i_f = interval.even_elements();

                    if (i_f.is_valid())
                    {

                        auto i_c = i_f >> 1;
                        auto j_c = j_f >> 1;
                        cell_tag(level - 1, i_c, j_c) |= cell_tag(level, i_f  - s[0], j_f - s[1]);
                    }

                    i_f = interval.odd_elements();
                    if (i_f.is_valid())
                    {
                        auto i_c = i_f >> 1;
                        auto j_c = j_f >> 1;

                        cell_tag(level - 1, i_c, j_c) |= cell_tag(level, i_f  - s[0], j_f - s[1]);
                    }
                });
            }
        }

        cl_type cl;
        mure::for_each_interval(mesh, [&](std::size_t level, const auto& interval, const auto& index_yz)
        {
            for (coord_index_t i = interval.start; i < interval.end; ++i)
            {
                if (cell_tag[i + interval.index])
                {
                    mure::static_nested_loop<dim - 1, 0, 2>([&](auto stencil)
                    {
                        auto index = 2 * index_yz + stencil;
                        cl[level + 1][index].add_interval({2 * i, 2 * i + 2});
                    });
                }
                else
                {
                    cl[level][index_yz].add_point(i);
                }
            }
        });

        mesh = {cl};
    }
}

int main(int argc, char *argv[])
{
    constexpr size_t dim = 2;
    mure::CellList<dim> cl;

    cxxopts::Options options("simple_2d", "simple 2d p4est examples");

    options.add_options()
                       ("max_level", "maximum level", cxxopts::value<std::size_t>()->default_value("8"));
                    //    ("test", "test case (1: gaussian, 2: diamond, 3: circle)", cxxopts::value<std::size_t>()->default_value("1"))
                    //    ("log", "log level", cxxopts::value<std::string>()->default_value("warning"))
                    //    ("h, help", "Help");
    auto result = options.parse(argc, argv);

    std::size_t max_level = result["max_level"].as<std::size_t>();

    cl[1][{0}].add_interval({1, 2});
    cl[1][{1}].add_interval({0, 1});

    cl[2][{0}].add_interval({0, 2});
    cl[2][{1}].add_interval({0, 2});
    cl[2][{2}].add_interval({2, 4});
    cl[2][{3}].add_interval({2, 4});

    mure::CellArray<dim> mesh_1(cl);
    std::cout << "nb_cells: " << mesh_1.nb_cells() << "\n";

    tic();
    refine_1(mesh_1, max_level);
    auto duration = toc();
    std::cout << "Version 1: " << duration << "s" << std::endl;toc();
    std::cout << "nb_cells: " << mesh_1.nb_cells() << "\n";

    mure::CellArray<dim> mesh_2(cl);

    tic();
    refine_2(mesh_2, max_level);
    duration = toc();
    std::cout << "Version 2: " << duration << "s" << std::endl;toc();
    std::cout << "nb_cells: " << mesh_2.nb_cells() << "\n";

    auto level = mure::make_field<std::size_t, 1>("level", mesh_2);
    mure::for_each_cell(mesh_2, [&](auto cell)
    {
        level[cell] = cell.level;
    });
    mure::save("simple_2d", mesh_2, level);

    return 0;
}