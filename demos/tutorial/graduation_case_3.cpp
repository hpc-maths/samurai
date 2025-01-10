// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <cmath>

#include <filesystem>

#include <xtensor/xmasked_view.hpp>

#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/samurai.hpp>
#include <samurai/subset/subset_op.hpp>

namespace fs = std::filesystem;

auto generate_mesh(std::size_t start_level)
{
    constexpr std::size_t dim = 2; // cppcheck-suppress unreadVariable
    const samurai::Box<int, dim> box({-(2 << start_level), -(2 << start_level)}, {2 << start_level, 2 << start_level});
    samurai::CellArray<dim> ca;

    ca[start_level] = {start_level, box};

    return ca;
}

int main(int argc, char* argv[])
{
    auto& app = samurai::initialize("Graduation example: test case 3", argc, argv);

    constexpr std::size_t dim = 2; // cppcheck-suppress unreadVariable
    std::size_t start_level   = 1;
    std::size_t max_level     = 6;
    bool with_graduation      = true;

    double PI = xt::numeric_constants<double>::PI;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "graduation_case_3";

    app.add_option("--start-level", start_level, "where to start the mesh generator")->capture_default_str();
    app.add_option("--max-level", max_level, "Maximum level of the mesh generator")->capture_default_str();
    app.add_flag("--with-graduation", with_graduation, "Make the mesh graduated")->capture_default_str();
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
    SAMURAI_PARSE(argc, argv);

    if (!fs::exists(path))
    {
        fs::create_directory(path);
    }

    auto ca = generate_mesh(start_level);

    std::size_t ite = 0;
    while (true)
    {
        std::cout << "Iteration for remove intersection: " << ite++ << "\n";

        auto tag = samurai::make_field<bool, 1>("tag", ca);
        tag.fill(false);

        samurai::for_each_cell(ca,
                               [&](auto cell)
                               {
                                   auto corner     = cell.corner();
                                   const double dx = cell.length;

                                   const std::size_t npoints = 1 << (max_level + 4);
                                   const double dt           = 2. * PI / static_cast<double>(npoints);
                                   double t                  = 0;

                                   for (std::size_t it = 0; it < npoints; ++it)
                                   {
                                       const double a     = 3;
                                       const double b     = 2;
                                       const double delta = PI * .5;
                                       const double xc    = std::sin(a * t + delta);
                                       const double yc    = std::sin(b * t);

                                       if ((corner[0] < xc) && (corner[0] + dx > xc) && (corner[1] < yc) && (corner[1] + dx > yc))
                                       {
                                           tag[cell] = true;
                                           break;
                                       }
                                       t += dt;
                                   }
                               });

        // graduation
        if (with_graduation)
        {
            xt::xtensor_fixed<int, xt::xshape<4, dim>> stencil{
                {1,  1 },
                {-1, -1},
                {-1, 1 },
                {1,  -1}
            };

            for (std::size_t level = ca.max_level(); level > 1; --level)
            {
                for (std::size_t i = 0; i < stencil.shape()[0]; ++i)
                {
                    auto s      = xt::view(stencil, i);
                    auto subset = samurai::intersection(samurai::translate(ca[level], s), ca[level - 1]);

                    subset(
                        [&](const auto& interval, const auto& index)
                        {
                            auto j_f = index[0];
                            auto i_f = interval.even_elements();

                            if (i_f.is_valid())
                            {
                                auto mask = tag(level, i_f - s[0], j_f - s[1]);
                                auto i_c  = i_f >> 1;
                                auto j_c  = j_f >> 1;
                                samurai::apply_on_masked(tag(level - 1, i_c, j_c),
                                                         mask,
                                                         [](auto& a)
                                                         {
                                                             a = true;
                                                         });
                            }

                            i_f = interval.odd_elements();
                            if (i_f.is_valid())
                            {
                                auto mask = tag(level, i_f - s[0], j_f - s[1]);
                                auto i_c  = i_f >> 1;
                                auto j_c  = j_f >> 1;
                                samurai::apply_on_masked(tag(level - 1, i_c, j_c),
                                                         mask,
                                                         [](auto& a)
                                                         {
                                                             a = true;
                                                         });
                            }
                        });
                }
            }
        }

        samurai::CellList<dim> cl;
        samurai::for_each_interval(ca,
                                   [&](std::size_t level, const auto& interval, const auto& index)
                                   {
                                       using size_type = typename decltype(tag)::size_type;
                                       auto j          = index[0];
                                       auto itag       = static_cast<size_type>(interval.start + interval.index);
                                       for (auto i = interval.start; i < interval.end; ++i)
                                       {
                                           if (tag[itag] && level < max_level)
                                           {
                                               cl[level + 1][{2 * j}].add_interval({2 * i, 2 * i + 2});
                                               cl[level + 1][{2 * j + 1}].add_interval({2 * i, 2 * i + 2});
                                           }
                                           else
                                           {
                                               cl[level][index].add_point(i);
                                           }
                                           itag++;
                                       }
                                   });

        samurai::CellArray<dim> new_ca = {cl, true};

        if (new_ca == ca)
        {
            break;
        }

        std::swap(ca, new_ca);
    }

    samurai::save(path, filename, ca);

    samurai::finalize();
    return 0;
}
