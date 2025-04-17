#include <gtest/gtest.h>
#include <samurai/mr/mesh.hpp>
#include <samurai/schemes/fv.hpp>

namespace samurai
{
    template <std::size_t dim>
    auto create_mesh(double box_boundary, std::size_t level)
    {
        using Config  = samurai::MRConfig<dim>;
        using Mesh    = samurai::MRMesh<Config>;
        using Box     = samurai::Box<double, dim>;
        using point_t = typename Box::point_t;

        point_t box_corner1, box_corner2;
        box_corner1.fill(0);
        box_corner2.fill(box_boundary);
        Box box(box_corner1, box_corner2);

        return Mesh(box, level, level);
    }

    TEST(domain_with_hole, substract_box)
    {
        static constexpr std::size_t dim = 2;

        using Config    = samurai::MRConfig<dim>;
        using Mesh      = samurai::MRMesh<Config>;
        using mesh_id_t = typename Mesh::mesh_id_t;
        using cl_t      = typename Mesh::cl_type;
        using lca_t     = typename Mesh::lca_type;
        using Box       = samurai::Box<double, dim>;

        std::size_t level = 3;

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

        samurai::MRMesh<Config> mesh{domain_with_hole_cl, level, level};

        EXPECT_EQ(mesh.nb_cells(mesh_id_t::cells), domain_lca.nb_cells() - hole_lca.nb_cells());
    }

}
