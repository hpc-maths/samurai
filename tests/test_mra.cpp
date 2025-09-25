#include <gtest/gtest.h>

#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>

namespace samurai
{
    auto init_mesh()
    {
        constexpr std::size_t dim   = 2;
        using Config                = samurai::MRConfig<dim>;
        auto box                    = samurai::Box<double, dim>({0., 0.}, {1., 1.});
        const std::size_t min_level = 2;
        const std::size_t max_level = 10;

        auto mesh_cfg = samurai::mesh_config<dim>().min_level(min_level).max_level(max_level);
        return samurai::MRMesh<Config>(mesh_cfg, box);
    }

    void init_field(auto& u, double factor = 1.0)
    {
        auto& mesh = u.mesh();
        u.resize();

        samurai::for_each_cell(
            mesh,
            [&](auto& cell)
            {
                auto center           = cell.center();
                const double radius   = .2;
                const double x_center = 0.3;
                const double y_center = 0.3;
                if (((center[0] - x_center) * (center[0] - x_center) + (center[1] - y_center) * (center[1] - y_center)) <= radius * radius)
                {
                    u[cell] = factor;
                }
                else
                {
                    u[cell] = 0;
                }
            });

        if constexpr (std::decay_t<decltype(u)>::is_scalar)
        {
            samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);
        }
        else
        {
            samurai::make_bc<samurai::Dirichlet<1>>(u, 0., 0.);
        }
    }

    auto scalar_test(bool relative_detail = false)
    {
        auto mesh  = init_mesh();
        auto field = samurai::make_scalar_field<double>("field", mesh);
        init_field(field);
        auto adapt      = samurai::make_MRAdapt(field);
        auto mra_config = samurai::mra_config().relative_detail(relative_detail);
        adapt(mra_config);
    }

    auto vector_test(bool relative_detail = false)
    {
        auto mesh  = init_mesh();
        auto field = samurai::make_vector_field<double, 2>("field", mesh);
        init_field(field);
        auto adapt      = samurai::make_MRAdapt(field);
        auto mra_config = samurai::mra_config().relative_detail(relative_detail);
        adapt(mra_config);
    }

    auto tuple_test(bool relative_detail = false)
    {
        auto mesh    = init_mesh();
        auto field_1 = samurai::make_scalar_field<double>("field_1", mesh);
        auto field_2 = samurai::make_vector_field<double, 2>("field_2", mesh);
        init_field(field_1);
        init_field(field_2, 2.0);
        auto adapt      = samurai::make_MRAdapt(field_1, field_2);
        auto mra_config = samurai::mra_config().relative_detail(relative_detail);
        adapt(mra_config);
    }

    TEST(MRA, scalar)
    {
        scalar_test(true);
        scalar_test(false);
    }

    TEST(MRA, vector)
    {
        vector_test(true);
        vector_test(false);
    }

    TEST(MRA, tuple)
    {
        tuple_test(true);
        tuple_test(false);
    }
}
