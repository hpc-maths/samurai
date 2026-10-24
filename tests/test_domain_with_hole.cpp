#include <gtest/gtest.h>
#include <samurai/domain_builder.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/schemes/fv.hpp>

namespace samurai
{

    TEST(domain_with_hole, substract_box)
    {
        static constexpr std::size_t dim = 2;

        std::size_t level = 3;
        auto mesh_cfg     = mesh_config<dim>().min_level(level).max_level(level);

        using Mesh      = decltype(mra::make_empty_mesh(mesh_cfg));
        using mesh_id_t = typename Mesh::mesh_id_t;
        using cl_t      = typename Mesh::cl_type;
        using lca_t     = typename Mesh::lca_type;
        using Box       = samurai::Box<double, dim>;

        const Box domain_box({-1., -1.}, {1., 1.});
        const Box hole_box({0.0, 0.0}, {0.2, 0.2});

        auto origin_point     = domain_box.min_corner();
        double scaling_factor = 0.2; // this value ensures that the hole is representable at level 0

        auto domain_lca = lca_t(level, domain_box, -1, scaling_factor);
        auto hole_lca   = lca_t(level, hole_box, origin_point, -1, scaling_factor);

        auto domain_with_hole_set = samurai::difference(domain_lca, hole_lca);

        cl_t domain_with_hole_cl(origin_point, scaling_factor);
        domain_with_hole_set(
            [&](const auto& interval, const auto& index_y)
            {
                domain_with_hole_cl[level][index_y].add_interval({interval});
            });

        auto mesh = samurai::mra::make_mesh(domain_with_hole_cl, mesh_cfg);

        EXPECT_EQ(mesh.nb_cells(mesh_id_t::cells), domain_lca.nb_cells() - hole_lca.nb_cells());
    }

    // The hole [0.25, 0.75)^2 is 4 x 4 cells of level 3 with a scaling factor of 1,
    // even though its side is not a multiple of the scaling factor.
    TEST(domain_builder, hole_with_given_scaling_factor)
    {
        static constexpr std::size_t dim = 2;
        DomainBuilder<dim> domain({0., 0.}, {1., 1.});
        domain.remove({0.25, 0.25}, {0.75, 0.75});

        auto mesh_cfg   = mesh_config<dim>().min_level(3).max_level(3).scaling_factor(1.).disable_args_parse();
        auto mesh       = mra::make_mesh(domain, mesh_cfg);
        using mesh_id_t = typename decltype(mesh)::mesh_id_t;

        // the 8 x 8 cells of [0, 1)^2 at level 3, minus the 4 x 4 cells of the hole
        EXPECT_EQ(mesh.nb_cells(mesh_id_t::cells), 48);
    }

}
