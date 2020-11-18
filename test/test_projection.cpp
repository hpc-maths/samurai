#include <gtest/gtest.h>
#include <xtensor/xarray.hpp>

// #include <samurai/field.hpp>
// #include <samurai/mr/mesh.hpp>
// #include <samurai/mr/mr_config.hpp>
// #include <samurai/mr/pred_and_proj.hpp>

namespace samurai
{
    TEST(cell, projection)
    {
        constexpr size_t dim = 1;
        // using Config = MRConfig<dim>;

        // CellList<Config> cell_list;
        // cell_list[4][{}].add_interval({24, 40});
        // cell_list[3][{}].add_interval({10, 12});
        // cell_list[3][{}].add_interval({20, 22});
        // cell_list[2][{}].add_interval({4, 5});
        // cell_list[2][{}].add_interval({11, 12});
        // cell_list[1][{}].add_interval({0, 2});
        // cell_list[1][{}].add_interval({6, 8});

        // Mesh<Config> mesh(cell_list);
        // std::cout << mesh << "\n";

        // Field<Config> u("u", mesh);
        // u.array().fill(0);

        // // mesh.for_each_cell([&](auto &cell) {
        // //     auto center = cell.center();
        // //     auto x = center[0];
        // //     double mid = 1.25;
        // //     if (x < mid)
        // //     {
        // //         u[cell] = x / mid;
        // //     }
        // //     else
        // //     {
        // //         u[cell] = (-x + 2 * mid) / mid;
        // //     }
        // // });

        // mesh.for_each_cell(
        //     [&](auto &cell) {
        //         auto center = cell.center();
        //         auto x = center[0];
        //         double mid = 2;
        //         if (x < mid)
        //         {
        //             u[cell] = x / mid;
        //         }
        //         else
        //         {
        //             u[cell] = (-x + 2 * mid) / mid;
        //         }
        //     },
        //     MeshType::all_cells);

        // mesh.for_each_cell([&](auto &cell) { u[cell] = 0; },
        //                    MeshType::proj_cells);

        // auto h5file = samurai::Hdf5("test_proj");
        // h5file.add_mesh(mesh);
        // h5file.add_field(u);

        // samurai::Field<Config> level_end{"level", mesh};
        // mesh.for_each_cell([&](auto &cell) {
        //     level_end[cell] = static_cast<double>(cell.level);
        // });
        // h5file.add_field(level_end);

        // mr_projection(u);

        // std::cout << u << "\n";
    }
}