#include <cmath>
#include <limits>
#include <utility>

#include <gtest/gtest.h>
#include <samurai/algorithm/graduation.hpp>
#include <samurai/algorithm/update.hpp>
#include <samurai/bc.hpp>
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
#ifdef SAMURAI_WITH_MPI
        GTEST_SKIP() << "DomainBuilder is not implemented with MPI";
#endif
        static constexpr std::size_t dim = 2;
        DomainBuilder<dim> domain({0., 0.}, {1., 1.});
        domain.remove({0.25, 0.25}, {0.75, 0.75});

        auto mesh_cfg   = mesh_config<dim>().min_level(3).max_level(3).scaling_factor(1.).disable_args_parse();
        auto mesh       = mra::make_mesh(domain, mesh_cfg);
        using mesh_id_t = typename decltype(mesh)::mesh_id_t;

        // the 8 x 8 cells of [0, 1)^2 at level 3, minus the 4 x 4 cells of the hole
        EXPECT_EQ(mesh.nb_cells(mesh_id_t::cells), 48);
    }

    // At the re-entrant corner of a hole, a coarse ghost can cover finer cells (so its outer ghosts
    // are extrapolated) and be in the reference mesh while its inner neighbour is not: it is only
    // there to compute the details of the finer cells on the other side of the corner. The
    // extrapolation of the second ghost layer read that inner neighbour: along y the ghost update
    // threw, along x the far ghost silently got a wrong value.
    //
    // The domain is [-2, 2)^2 with the hole [-1, 1)^2. With `transposed == false`, the mesh is at
    // level 5 above the hole, level 4 on its right and level 2 elsewhere, and the corner is crossed
    // along y. With `transposed == true`, i and j swap roles and the corner is crossed along x.
    void check_ghost_update_at_reentrant_corner(bool transposed)
    {
        static constexpr std::size_t dim = 2;
        using coords_t                   = xt::xtensor_fixed<int, xt::xshape<dim>>;

        DomainBuilder<dim> domain({-2., -2.}, {2., 2.});
        domain.remove({-1., -1.}, {1., 1.});

        // At level 2, the domain is [0, 16)^2 and the hole is [4, 12)^2.
        const std::size_t level = 2;
        auto mesh_cfg           = mesh_config<dim>().min_level(level).max_level(5).disable_args_parse();
        auto domain_mesh        = mra::make_mesh(domain, mesh_cfg);
        using Mesh              = decltype(domain_mesh);
        using mesh_id_t         = typename Mesh::mesh_id_t;

        // Level-2 indices (along, across): `along` is the direction in which the corner is crossed.
        auto at = [&](int along_i, int along_j)
        {
            return transposed ? coords_t{along_j, along_i} : coords_t{along_i, along_j};
        };
        auto level_of = [&](int i, int j) -> std::size_t
        {
            if (transposed)
            {
                std::swap(i, j);
            }
            if (j >= 12)
            {
                return 5;
            }
            if (i >= 12 && j >= 4)
            {
                return 4;
            }
            return level;
        };

        // The cells of domain_mesh are at level 5, so the mesh is built from the domain at level 2.
        typename Mesh::cl_type cl(domain_mesh.origin_point(), domain_mesh.scaling_factor());
        for_each_cell(domain_mesh.domain_pyramid()[level],
                      [&](const auto& cell)
                      {
                          const auto i         = cell.indices[0];
                          const auto j         = cell.indices[1];
                          const std::size_t l  = level_of(i, j);
                          const int refinement = 1 << (l - level);
                          for (int fine_j = j * refinement; fine_j < (j + 1) * refinement; ++fine_j)
                          {
                              cl[l][{fine_j}].add_interval({i * refinement, (i + 1) * refinement});
                          }
                      });
        typename Mesh::ca_type ca(cl, true);
        make_graduation(ca,
                        domain_mesh.domain_pyramid(),
                        domain_mesh.mpi_neighbourhood(),
                        domain_mesh.periodicity(),
                        domain_mesh.graduation_width(),
                        domain_mesh.max_stencil_radius());
        Mesh mesh(ca, domain_mesh);

        // The configuration: the ghost at(11, 12), just past the corner of the hole, covers finer
        // cells. It and the two ghosts before it, in the hole, are in the reference mesh. Its inner
        // neighbour at(11, 13) is not.
        auto in = [](const auto& lca, const coords_t& indices)
        {
            return find(lca, indices) >= 0;
        };
        ASSERT_TRUE(in(mesh.get_union()[level], at(11, 12)));
        ASSERT_FALSE(in(mesh[mesh_id_t::cells][level], at(11, 12)));
        ASSERT_TRUE(in(mesh[mesh_id_t::reference][level], at(11, 12)));
        ASSERT_TRUE(in(mesh[mesh_id_t::reference][level], at(11, 11)));
        ASSERT_TRUE(in(mesh[mesh_id_t::reference][level], at(11, 10)));
        ASSERT_FALSE(in(mesh[mesh_id_t::reference][level], at(11, 13)));

        // An affine field with an exact Dirichlet condition, known only on the leaves.
        auto exact = [](const auto& x)
        {
            return 2. * x[0] - 3. * x[1] + 1.;
        };
        const double grad_norm = std::sqrt(2. * 2. + 3. * 3.);
        auto u                 = make_scalar_field<double>("u", mesh, std::numeric_limits<double>::quiet_NaN());
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          u[cell] = exact(cell.center());
                      });
        make_bc<Dirichlet<1>>(u,
                              [&](const auto&, const auto&, const auto& coords)
                              {
                                  return exact(coords);
                              });
        ASSERT_NO_THROW(update_ghost_mr(u));

        for_each_cell(mesh[mesh_id_t::reference],
                      [&](const auto& cell)
                      {
                          EXPECT_FALSE(std::isnan(u[cell])) << cell;
                      });

        // The ghosts extrapolated from at(11, 12). The extrapolation skips the far ghost at(11, 10), so
        // its value comes from the other ghost updates, which are not exact for affine data near the
        // corner: hence a bound on the variation over two cells, not exactness.
        const double h = mesh.cell_length(level);
        for (const auto& indices : {at(11, 11), at(11, 10)})
        {
            const auto cell = mesh.get_cell(level, indices);
            EXPECT_LE(std::abs(u[cell] - exact(cell.center())), 2. * h * grad_norm) << cell;
        }
    }

    TEST(domain_builder, ghost_update_at_reentrant_corner_along_y)
    {
#ifdef SAMURAI_WITH_MPI
        GTEST_SKIP() << "DomainBuilder is not implemented with MPI";
#endif
        check_ghost_update_at_reentrant_corner(false);
    }

    TEST(domain_builder, ghost_update_at_reentrant_corner_along_x)
    {
#ifdef SAMURAI_WITH_MPI
        GTEST_SKIP() << "DomainBuilder is not implemented with MPI";
#endif
        check_ghost_update_at_reentrant_corner(true);
    }

}
