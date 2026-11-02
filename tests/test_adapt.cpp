#include <cmath>

#include <gtest/gtest.h>

#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>

#include "test_common.hpp"

namespace samurai
{

    template <typename T>
    class adapt_test : public ::testing::Test
    {
    };

#if __GNUC__ == 11
    // The 3d case is broken for gcc-11.4 with gtest for release build type and only in gtest. It works outside...
    using adapt_test_types = ::testing::Types<std::integral_constant<std::size_t, 1>, std::integral_constant<std::size_t, 2>>; //,
                                                                                                                               // std::integral_constant<std::size_t,
                                                                                                                               // 3>>;
#else
    using adapt_test_types = ::testing::
        Types<std::integral_constant<std::size_t, 1>, std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>>;
#endif

    TYPED_TEST_SUITE(adapt_test, adapt_test_types, );

    TYPED_TEST(adapt_test, mutliple_fields)
    {
        static constexpr std::size_t dim = TypeParam::value;

        using box_t   = Box<double, dim>;
        auto mesh_cfg = mesh_config<dim>().min_level(2).max_level(4);
        auto mesh     = mra::make_mesh(box_t{xt::zeros<double>({dim}), xt::ones<double>({dim})}, mesh_cfg);
        auto u_1      = make_scalar_field<double>("u_1", mesh);
        auto u_2      = make_vector_field<double, 3>("u_2", mesh);
        auto u_3      = make_vector_field<double, 2>("u_3", mesh);

        auto adapt      = make_MRAdapt(u_1, u_2, u_3);
        auto mra_config = samurai::mra_config().regularity(2.);
        adapt(mra_config);

        // After adaptation the fields must live on the new mesh topology.
        EXPECT_EQ(u_1.array().size(), u_1.mesh().nb_cells());
        EXPECT_EQ(u_2.array().size(), u_2.mesh().nb_cells() * u_2.n_comp);
        EXPECT_EQ(u_3.array().size(), u_3.mesh().nb_cells() * u_3.n_comp);
    }

    // With min_level 0, the adaptation coarsens the smooth part of the field down to level 0 while
    // keeping finer cells around a narrow bump. Building the mesh with cells at level 0 used to index
    // the cell list at level 0 - 1 and crash.
    TYPED_TEST(adapt_test, level_0_cells)
    {
        static constexpr std::size_t dim = TypeParam::value;
        constexpr std::size_t max_level  = 3;

        using box_t = Box<double, dim>;
        // A scaling factor of 1 gives 8 cells per direction at level 0.
        auto mesh_cfg = mesh_config<dim>().min_level(0).max_level(max_level).scaling_factor(1.).disable_args_parse();
        auto mesh     = mra::make_mesh(box_t{xt::zeros<double>({dim}), 8. * xt::ones<double>({dim})}, mesh_cfg);
        auto u        = make_scalar_field<double>("u", mesh);

        auto init = [&]()
        {
            u.resize();
            for_each_cell(mesh,
                          [&](const auto& cell)
                          {
                              const auto x = cell.center();
                              double r2    = 0.;
                              for (std::size_t d = 0; d < dim; ++d)
                              {
                                  r2 += (x[d] - 1.) * (x[d] - 1.);
                              }
                              u[cell] = 1. + std::exp(-r2 / 0.01);
                          });
        };
        init();

        auto adapt      = make_MRAdapt(u);
        auto mra_config = samurai::mra_config().epsilon(1e-3);
        adapt(mra_config);

        using mesh_id_t  = typename decltype(mesh)::mesh_id_t;
        auto nb_cells_at = [&](std::size_t level)
        {
            return test::global_sum(mesh[mesh_id_t::cells][level].nb_cells());
        };
        EXPECT_GT(nb_cells_at(0), 0);
        EXPECT_GT(nb_cells_at(max_level), 0);

        // The ghosts of the adapted mesh can be filled, and a second adaptation runs on it.
        init();
        update_ghost_mr(u);
        bool all_finite = true;
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          all_finite = all_finite && std::isfinite(u[cell]);
                      });
        EXPECT_EQ(test::global_min(static_cast<int>(all_finite)), 1);

        adapt(mra_config);
        EXPECT_GT(nb_cells_at(0), 0);
        EXPECT_EQ(u.array().size(), mesh.nb_cells());
    }
}
