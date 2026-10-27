#include <gtest/gtest.h>

#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/reconstruction.hpp>

namespace samurai
{
    // transfer() must write into a destination field whose mesh has grown since the field was
    // sized. This is what happens to a field living on a mesh adapted by an MRAdapt it was not
    // given: the lid-driven cavity demo transfers its velocity onto the ink mesh after each ink
    // adaptation, and used to abort with xt::broadcast_error once that mesh held more cells.
    TEST(transfer, destination_mesh_grown_since_the_field_was_sized)
    {
        constexpr std::size_t dim = 2;

        const Box<double, dim> box({0, 0}, {1, 1});
        auto cfg = mesh_config<dim>().min_level(1).max_level(4).disable_args_parse();

        auto mesh_src    = mra::make_mesh(box, cfg);
        using mesh_t     = decltype(mesh_src);
        using interval_t = typename mesh_t::interval_t;

        auto u_src = make_vector_field<double, 2>("u_src", mesh_src);
        for_each_cell(mesh_src,
                      [&](const auto& cell)
                      {
                          u_src[cell][0] = cell.center(0) + 2 * cell.center(1);
                          u_src[cell][1] = 3 * cell.center(0) - cell.center(1);
                      });

        // The destination field is sized on a mesh with the 4 cells of level 1 ...
        typename mesh_t::cl_type cl;
        for (int j = 0; j < 2; ++j)
        {
            cl[1][{j}].add_interval(interval_t{0, 2});
        }
        auto mesh_dst = mra::make_mesh(cl, cfg);
        auto u_dst    = make_vector_field<double, 2>("u_dst", mesh_dst);

        // ... then that mesh changes under it, as an adaptation does, and now holds the finest level.
        auto grown = mra::make_mesh(box, cfg);
        mesh_dst.swap(grown);
        ASSERT_GT(mesh_dst.nb_cells(), grown.nb_cells());

        transfer(u_src, u_dst);

        EXPECT_EQ(u_dst.array().size(), mesh_dst.nb_cells() * u_dst.n_comp);
        for_each_cell(mesh_dst,
                      [&](const auto& cell)
                      {
                          EXPECT_DOUBLE_EQ(u_dst[cell][0], cell.center(0) + 2 * cell.center(1));
                          EXPECT_DOUBLE_EQ(u_dst[cell][1], 3 * cell.center(0) - cell.center(1));
                      });
    }
}
