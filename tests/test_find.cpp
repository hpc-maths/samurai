#include <gtest/gtest.h>

#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>

namespace samurai
{
    TEST(find, test)
    {
        static constexpr std::size_t dim = 2;
        using Config                     = samurai::MRConfig<dim>;
        using Box                        = samurai::Box<double, dim>;
        using Mesh                       = samurai::MRMesh<Config>;
        using coords_t                   = typename Mesh::cell_t::coords_t;

        Box box({-1., -1.}, {1., 1.});

        std::size_t min_level = 2;
        std::size_t max_level = 6;

        auto mesh_cfg = samurai::mesh_config<dim>().min_level(min_level).max_level(max_level);
        Mesh mesh{mesh_cfg, box};

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

        coords_t coords = {0.4, 0.8};
        auto cell       = samurai::find_cell(mesh, coords);

        EXPECT_TRUE(cell.length > 0);                                                             // cell found
        EXPECT_TRUE(static_cast<std::size_t>(cell.index) < mesh.nb_cells());                      // cell index makes sense
        EXPECT_TRUE(xt::all(cell.corner() <= coords && coords <= (cell.corner() + cell.length))); // coords in cell
    }
}
