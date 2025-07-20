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

    //~ using dim_test_types = ::testing::
    //~ Types<std::integral_constant<std::size_t, 1>, std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>>;
    using dim_test_types = ::testing::Types<std::integral_constant<std::size_t, 1>>;

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
        using Config                     = MRConfig<dim, 1>;

        // Simulation parameters
        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner, max_corner;
        min_corner.fill(-1);
        max_corner.fill(1);

        // Multiresolution parameters
        std::size_t min_level = 3, max_level = 6;
        double mr_epsilon    = 1.e-4; // Threshold used by multiresolution
        double mr_regularity = 1.;    // Regularity guess for multiresolution

        Box<double, dim> box(min_corner, max_corner);
        std::array<bool, dim> bc;
        bc.fill(true);
        MRMesh<Config> mesh{box, min_level, max_level, bc};
        using mesh_id_t = typename MRMesh<Config>::mesh_id_t;

        double dt = 1;
        double Tf = 2 / mesh.cell_length(max_level);
        double t  = 0.;

        auto u    = init(mesh);
        auto unp1 = make_scalar_field<double>("unp1", mesh);
        unp1.fill(0);

        //~ save("before_addapt", mesh, u);

        auto MRadaptation = make_MRAdapt(u);
        MRadaptation(mr_epsilon, mr_regularity);

        //~ save("after_addapt", mesh, u);

        //~ while (t != Tf)
        for (int ite = 0; ite != 8; ++ite)
        {
            fmt::print("==================================================\n");
            //~ save(fmt::format("before_addapt_{}", ite).c_str(), mesh, u);
            MRadaptation(mr_epsilon, mr_regularity);
            //~ save(fmt::format("after_addapt_{}", ite++).c_str(), mesh, u);

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
