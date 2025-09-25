#include <gtest/gtest.h>
#include <samurai/mr/adapt.hpp>
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
        auto mesh_cfg = samurai::mesh_config<dim>().min_level(level).max_level(level);

        return Mesh(mesh_cfg, box);
    }

    /**
     * Tests if the scaling factor corresponds to length of the domain.
     */
    TEST(scaling, box_scaling_factor)
    {
        static constexpr std::size_t dim = 1;
        double box_boundary              = 123;
        std::size_t level                = 3;

        auto mesh = create_mesh<dim>(box_boundary, level);

        EXPECT_EQ(mesh.scaling_factor(), box_boundary);
    }

    /**
     * Tests if
     *    - the number of cells is correct and doesn't depend on the domain size;
     *    - cell.length == mesh.cell_length(...)
     */
    TEST(scaling, nb_cells)
    {
        static constexpr std::size_t dim = 2;
        double box_boundary              = 123;
        std::size_t level                = 3;

        auto mesh = create_mesh<dim>(box_boundary, level);

        using mesh_id_t = typename decltype(mesh)::mesh_id_t;

        EXPECT_EQ(mesh[mesh_id_t::cells].nb_cells(), std::pow(2, level * dim));

        for_each_cell(mesh,
                      [&](auto cell)
                      {
                          EXPECT_EQ(mesh.cell_length(cell.level), cell.length);
                      });
    }

    /**
     * For a box of size 0.1, tests if the boundary cells' coordinates are indeed close to 0.1 (up to cell.length),
     * i.e. if the domain is correctly meshed.
     */
    TEST(scaling, boundary_cells)
    {
        static constexpr std::size_t dim = 2;
        double box_boundary              = 0.1;
        std::size_t level                = 3;

        auto mesh = create_mesh<dim>(box_boundary, level);

        for_each_boundary_interface(mesh,
                                    [&](auto cell, auto&)
                                    {
                                        bool one_coord_on_bdry = false;
                                        for (std::size_t d = 0; d < dim; ++d)
                                        {
                                            one_coord_on_bdry = one_coord_on_bdry || abs(cell.center(d)) < cell.length
                                                             || abs(cell.center(d) - box_boundary) < cell.length;
                                        }
                                        EXPECT_TRUE(one_coord_on_bdry);
                                    });
    }

    /**
     * Same test as before, but using the [0,1] box and and the scale_domain() instead.
     */
    TEST(scaling, scale_domain)
    {
        static constexpr std::size_t dim = 2;
        double box_boundary              = 123;
        std::size_t level                = 3;

        auto mesh = create_mesh<dim>(1, level);
        mesh.scale_domain(box_boundary);

        for_each_boundary_interface(mesh,
                                    [&](auto cell, auto&)
                                    {
                                        bool one_coord_on_bdry = false;
                                        for (std::size_t d = 0; d < dim; ++d)
                                        {
                                            one_coord_on_bdry = one_coord_on_bdry || abs(cell.center(d)) < cell.length
                                                             || abs(cell.center(d) - box_boundary) < cell.length;
                                        }
                                        EXPECT_TRUE(one_coord_on_bdry);
                                    });
    }
}
