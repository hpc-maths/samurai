#include <gtest/gtest.h>

#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>

namespace samurai
{
    TEST(find, coords)
    {
        static constexpr std::size_t dim = 2;
        using Box                        = samurai::Box<double, dim>;
        Box box({-1., -1.}, {1., 1.});

        auto mesh_cfg = mesh_config<dim>().min_level(2).max_level(6);
        auto mesh     = mra::make_mesh(box, mesh_cfg);

        auto u = samurai::make_scalar_field<double>("u",
                                                    mesh,
                                                    [](const auto& coords)
                                                    {
                                                        const auto& x = coords(0);
                                                        const auto& y = coords(1);
                                                        return (x >= -0.8 && x <= -0.3 && y >= 0.3 && y <= 0.8) ? 1. : 0.;
                                                    });

        auto MRadaptation = samurai::make_MRAdapt(u);
        auto mra_config   = samurai::mra_config().epsilon(1e-3);
        MRadaptation(mra_config);

        using coords_t  = typename decltype(mesh)::cell_t::coords_t;
        coords_t coords = {0.4, 0.8};
        auto cell       = samurai::find_cell(mesh, coords);

        EXPECT_TRUE(cell.length > 0);                                                             // cell found
        EXPECT_TRUE(static_cast<std::size_t>(cell.index) < mesh.nb_cells());                      // cell index makes sense
        EXPECT_TRUE(xt::all(cell.corner() <= coords && coords <= (cell.corner() + cell.length))); // coords in cell
    }

    TEST(find, indices)
    {
        static constexpr std::size_t dim = 2;
        using Box                        = samurai::Box<double, dim>;
        Box box({-1., -1.}, {1., 1.});

        auto mesh_cfg = mesh_config<dim>().min_level(2).max_level(6);
        auto mesh     = mra::make_mesh(box, mesh_cfg);

        {
            using coords_t = typename decltype(mesh)::cell_t::coords_t;

            const coords_t coords = {0.4, 0.8};
            const auto cell1      = samurai::find_cell(mesh, coords);

            EXPECT_TRUE(cell1.length > 0);

            const auto cell2 = samurai::find_cell(mesh, cell1.indices, cell1.level);

            EXPECT_EQ(cell1, cell2);
        }
        {
            using indices_t = typename decltype(mesh)::cell_t::indices_t;

            const indices_t indices = {5, 10};

            const auto cell1 = samurai::find_cell(mesh, indices, mesh.max_level());

            EXPECT_TRUE(cell1.length > 0);

            const auto cell2 = samurai::find_cell(mesh, cell1.center());

            EXPECT_EQ(cell1, cell2);

            const auto cell3 = samurai::find_cell(mesh, indices, mesh.max_level() + 1);

            EXPECT_EQ(cell1, cell2);

            const auto cell4 = samurai::find_cell(mesh, indices, std::size_t(mesh.min_level() - 1));

            EXPECT_EQ(cell4.length, 0);
        }
    }
}
