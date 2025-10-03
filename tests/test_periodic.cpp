#include <gtest/gtest.h>

#include <xtensor/containers/xfixed.hpp>

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

    TYPED_TEST_SUITE(dim_test, dim_test_types, );

    template <class Mesh>
    auto init(Mesh& mesh)
    {
        double dx = mesh.cell_length(mesh.max_level());
        auto u    = make_scalar_field<double>("u", mesh);
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

        // Simulation parameters
        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner, max_corner;
        min_corner.fill(-1);
        max_corner.fill(1);

        // Multiresolution parameters
        Box<double, dim> box(min_corner, max_corner);
        auto mesh_cfg   = samurai::mesh_config<dim>().min_level(3).max_level(6).periodic(true).graduation_width(1);
        auto mesh       = samurai::make_MRMesh(mesh_cfg, box);
        using mesh_id_t = typename decltype(mesh)::mesh_id_t;

        double dt = 1;
        double Tf = 2 / mesh.cell_length(mesh.max_level());
        double t  = 0.;

        auto u    = init(mesh);
        auto unp1 = make_scalar_field<double>("unp1", mesh);
        unp1.fill(0);

        auto MRadaptation = make_MRAdapt(u);
        auto mra_config   = samurai::mra_config();
        MRadaptation(mra_config);

        while (t != Tf)
        {
            MRadaptation(mra_config);

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
                                  else
                                  {
                                      unp1(level, i, index) = u(level, i - 1, index - 1);
                                  }
                              });

            std::swap(u.array(), unp1.array());
        }
        auto u_init = init(mesh);

        ASSERT_EQ(u, u_init);
    }
}
