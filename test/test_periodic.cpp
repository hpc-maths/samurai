#include <gtest/gtest.h>

#include <xtensor/xfixed.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>

namespace samurai
{

    template <typename T>
    class dim_test : public ::testing::Test
    {
    };

    using dim_test_types = ::testing::
        Types<std::integral_constant<std::size_t, 1>, std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>>;

    TYPED_TEST_SUITE(dim_test, dim_test_types);

    template <class Mesh>
    auto init(Mesh& mesh)
    {
        double dx = 1. / (1 << mesh.max_level());
        auto u    = make_field<double, 1>("u", mesh);
        u.fill(0.);

        for_each_cell(mesh,
                      [&](auto& cell)
                      {
                          auto center   = cell.center();
                          double radius = std::floor(.2 / dx) * dx;

                          if (xt::all(xt::abs(center) <= radius))
                          {
                              u[cell] = 1;
                          }
                      });

        return u;
    }

    TYPED_TEST(dim_test, periodic)
    {
        static constexpr std::size_t dim = TypeParam::value;
        using Config                     = MRConfig<dim, 2>;

        // Simulation parameters
        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner, max_corner;
        min_corner.fill(-1);
        max_corner.fill(1);

        // Multiresolution parameters
        std::size_t min_level = 2, max_level = 5;
        double mr_epsilon    = 2.e-4; // Threshold used by multiresolution
        double mr_regularity = 1.;    // Regularity guess for multiresolution

        Box<double, dim> box(min_corner, max_corner);
        std::array<bool, dim> bc;
        bc.fill(true);
        MRMesh<Config> mesh{box, min_level, max_level, bc};
        using mesh_id_t = typename MRMesh<Config>::mesh_id_t;

        double dt = 1;
        double Tf = 2 * (1 << max_level);
        double t  = 0.;

        auto u    = init(mesh);
        auto unp1 = make_field<double, 1>("unp1", mesh);
        unp1.fill(0);

        auto MRadaptation = make_MRAdapt(u);
        MRadaptation(mr_epsilon, mr_regularity);

        while (t != Tf)
        {
            MRadaptation(mr_epsilon, mr_regularity);

            t += dt;
            if (t > Tf)
            {
                dt += Tf - t;
                t = Tf;
            }

            update_ghost_mr(u);
            unp1.resize();
            for_each_interval(mesh[mesh_id_t::cells],
                              [&](std::size_t level, auto& i, auto& index)
                              {
                                  if constexpr (dim == 1)
                                  {
                                      unp1(level, i) = u(level, i - 1);
                                  }
                                  else if constexpr (dim == 2)
                                  {
                                      auto j            = index[0];
                                      unp1(level, i, j) = u(level, i - 1, j - 1);
                                  }
                                  else if constexpr (dim == 3)
                                  {
                                      auto j               = index[0];
                                      auto k               = index[1];
                                      unp1(level, i, j, k) = u(level, i - 1, j - 1, k - 1);
                                  }
                              });

            std::swap(u.array(), unp1.array());
        }
        auto u_init = init(mesh);

        ASSERT_EQ(u, u_init);
    }
}
