#include <cmath>

#include <gtest/gtest.h>
#include <samurai/algorithm.hpp>
#include <samurai/algorithm/update_ghost_mr.hpp>
#include <samurai/box.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/reconstruction.hpp>
#include <samurai/uniform_mesh.hpp>

// Tests for portion() (include/samurai/reconstruction.hpp): the reconstructed value a field would
// take on a virtually finer grid. Two forms are exercised:
//   - scalar child index  ii = {k...}          : one fine child, used by the FV flux and transfer();
//   - interval child index ii = {[a, b)...}    : a whole slice of children summed at once, used by
//                                                the LBM stream (LBMScheme::portion_column).
// The reconstruction is done on a single-level uniform mesh, so portion reads only real cells; the
// coarse cells are kept in the interior so the prediction stencil never reaches outside the domain.

namespace samurai
{
    // Affine field u = sum of the cell-centre coordinates (u = x in 1D, x + y in 2D, ...). The
    // radius-1 wavelet prediction reproduces affine fields exactly, so the reconstruction of any
    // child must equal that child's exact centre coordinate sum.
    template <class Mesh>
    auto init_affine(Mesh& mesh)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        auto u          = make_scalar_field<double>("u", mesh);
        for_each_cell(mesh[mesh_id_t::cells],
                      [&](const auto& cell)
                      {
                          u[cell] = xt::sum(cell.center())[0];
                      });
        return u;
    }

    // A deliberately non-affine field: the linearity and mean-conservation identities below hold for
    // any field, so using one the predictor does NOT reproduce exactly makes sure those tests check
    // the aggregation logic rather than accidental exactness.
    template <class Mesh>
    auto init_nonlinear(Mesh& mesh)
    {
        static constexpr std::size_t dim = Mesh::dim;
        using mesh_id_t                  = typename Mesh::mesh_id_t;
        auto u                           = make_scalar_field<double>("u", mesh);
        for_each_cell(mesh[mesh_id_t::cells],
                      [&](const auto& cell)
                      {
                          const auto c = cell.center();
                          double v     = 1.;
                          for (std::size_t d = 0; d < dim; ++d)
                          {
                              v += static_cast<double>(d + 1) * c[d] * c[d] + std::cos(static_cast<double>(d + 2) * c[d]);
                          }
                          u[cell] = v;
                      });
        return u;
    }

    // ------------------------------------------------------------------ exactness on an affine field
    // portion of a single child must return that child's exact value. The radius-1 predictor is exact
    // on affine fields, so this pins the scalar form (the per-child usage of FV flux / transfer) down
    // to the analytic cell-centre value, for every child of a coarse cell, in 1D/2D/3D.

    TEST(portion, affine_all_children_1d)
    {
        constexpr std::size_t dim = 1;
        using config              = UniformConfig<dim>;
        using interval_t          = typename config::interval_t;
        constexpr std::size_t L   = 5;
        constexpr std::size_t dl  = 4; // reconstruct four levels below -> level 9
        auto mesh                 = UniformMesh<config>(Box<double, dim>({0}, {1}), L);
        auto u                    = init_affine(mesh);

        const int i = 2;
        for (int ii = 0; ii < (1 << dl); ++ii)
        {
            auto p                = portion<1>(u, L, dl, std::make_tuple(interval_t{i, i + 1}), std::make_tuple(ii));
            const double fine_ctr = ((i << dl) + ii + .5) / (1 << (L + dl));
            EXPECT_DOUBLE_EQ(p[0], fine_ctr) << "child " << ii;
        }
    }

    TEST(portion, affine_all_children_2d)
    {
        constexpr std::size_t dim = 2;
        using config              = UniformConfig<dim>;
        using interval_t          = typename config::interval_t;
        constexpr std::size_t L   = 5;
        constexpr std::size_t dl  = 4;
        auto mesh                 = UniformMesh<config>(Box<double, dim>({0, 0}, {1, 1}), L);
        auto u                    = init_affine(mesh);

        const int i = 2, j = 2;
        for (int jj = 0; jj < (1 << dl); ++jj)
        {
            for (int ii = 0; ii < (1 << dl); ++ii)
            {
                auto p             = portion<1>(u, L, dl, std::make_tuple(interval_t{i, i + 1}, j), std::make_tuple(ii, jj));
                const double x_ctr = ((i << dl) + ii + .5) / (1 << (L + dl));
                const double y_ctr = ((j << dl) + jj + .5) / (1 << (L + dl));
                EXPECT_DOUBLE_EQ(p[0], x_ctr + y_ctr) << "child (" << ii << ", " << jj << ")";
            }
        }
    }

    TEST(portion, affine_all_children_3d)
    {
        constexpr std::size_t dim = 3;
        using config              = UniformConfig<dim>;
        using interval_t          = typename config::interval_t;
        constexpr std::size_t L   = 4;
        constexpr std::size_t dl  = 3;
        auto mesh                 = UniformMesh<config>(Box<double, dim>({0, 0, 0}, {1, 1, 1}), L);
        auto u                    = init_affine(mesh);

        const int i = 2, j = 2, k = 2;
        for (int kk = 0; kk < (1 << dl); ++kk)
        {
            for (int jj = 0; jj < (1 << dl); ++jj)
            {
                for (int ii = 0; ii < (1 << dl); ++ii)
                {
                    auto p             = portion<1>(u, L, dl, std::make_tuple(interval_t{i, i + 1}, j, k), std::make_tuple(ii, jj, kk));
                    const double x_ctr = ((i << dl) + ii + .5) / (1 << (L + dl));
                    const double y_ctr = ((j << dl) + jj + .5) / (1 << (L + dl));
                    const double z_ctr = ((k << dl) + kk + .5) / (1 << (L + dl));
                    EXPECT_DOUBLE_EQ(p[0], x_ctr + y_ctr + z_ctr);
                }
            }
        }
    }

    // ------------------------------------------------------------------ delta_l == 0 (identity)
    // Base case of the prediction recursion: no refinement, so portion must return the coarse value
    // unchanged. This is the path taken at the finest level of the LBM stream and by the same-level
    // copy of transfer().

    TEST(portion, identity_delta_l_zero)
    {
        constexpr std::size_t dim = 1;
        using config              = UniformConfig<dim>;
        using interval_t          = typename config::interval_t;
        constexpr std::size_t L   = 5;
        auto mesh                 = UniformMesh<config>(Box<double, dim>({0}, {1}), L);
        auto u                    = init_nonlinear(mesh);

        for (int i = 4; i < 8; ++i)
        {
            auto p = portion<1>(u, L, 0, std::make_tuple(interval_t{i, i + 1}), std::make_tuple(0));
            EXPECT_DOUBLE_EQ(p[0], u(L, interval_t{i, i + 1})(0)) << "cell " << i;
        }
    }

    // ------------------------------------------------------------------ mean conservation (slice form)
    // The predictor preserves the cell average: the 2^{dl.dim} children of a coarse cell reconstruct
    // to values whose mean is exactly the coarse value, for ANY field. The slice form sums that full
    // column (without the 1/2^{dl.dim} weight the caller applies), so it must return 2^{dl.dim} times
    // the coarse value. This is precisely the projection the LBM stream relies on for mass conservation.

    TEST(portion, mean_conservation_full_column_1d)
    {
        constexpr std::size_t dim = 1;
        using config              = UniformConfig<dim>;
        using interval_t          = typename config::interval_t;
        constexpr std::size_t L   = 5;
        auto mesh                 = UniformMesh<config>(Box<double, dim>({0}, {1}), L);
        auto u                    = init_nonlinear(mesh);

        const int i = 8;
        for (std::size_t dl = 0; dl <= 4; ++dl)
        {
            const int n   = 1 << dl;
            auto column   = portion<1>(u, L, dl, std::make_tuple(interval_t{i, i + 1}), std::make_tuple(interval_t{0, n}));
            const double c = u(L, interval_t{i, i + 1})(0);
            EXPECT_NEAR(column[0], n * c, 1e-12) << "dl = " << dl;
        }
    }

    TEST(portion, mean_conservation_full_column_2d)
    {
        constexpr std::size_t dim = 2;
        using config              = UniformConfig<dim>;
        using interval_t          = typename config::interval_t;
        constexpr std::size_t L   = 5;
        auto mesh                 = UniformMesh<config>(Box<double, dim>({0, 0}, {1, 1}), L);
        auto u                    = init_nonlinear(mesh);

        const int i = 8, j = 8;
        for (std::size_t dl = 0; dl <= 3; ++dl)
        {
            const int n    = 1 << dl;
            auto column    = portion<1>(u, L, dl, std::make_tuple(interval_t{i, i + 1}, j), std::make_tuple(interval_t{0, n}, interval_t{0, n}));
            const double c = u(L, interval_t{i, i + 1}, j)(0);
            EXPECT_NEAR(column[0], (n * n) * c, 1e-12) << "dl = " << dl;
        }
    }

    // ------------------------------------------------------------------ slice == sum of scalar children
    // The interval form must equal summing the scalar form over the same children (the reconstruction
    // is linear). This directly guards the slice aggregation that makes the LBM stream cheap. The
    // boxes cover the corner cases the aggregation must handle:
    //   - the full column [0, 2^dl)               : the LBM projection;
    //   - a partial sub-box [a, b)                : arbitrary slice;
    //   - a box starting below 0                  : the LBM donor slice shifted by -c, which needs the
    //                                               signed accumulate_slice loop (a size_t loop would
    //                                               wrap and this is the case that once failed to build);
    //   - a single child [k, k+1)                 : degenerate slice, must equal the scalar form.

    TEST(portion, slice_equals_sum_of_children_1d)
    {
        constexpr std::size_t dim = 1;
        using config              = UniformConfig<dim>;
        using interval_t          = typename config::interval_t;
        constexpr std::size_t L   = 5;
        constexpr std::size_t dl  = 2; // children in [0, 4)
        auto mesh                 = UniformMesh<config>(Box<double, dim>({0}, {1}), L);
        auto u                    = init_nonlinear(mesh);

        const int i = 8;
        auto coarse = std::make_tuple(interval_t{i, i + 1});

        // {full column, partial box, box starting below 0, single child}
        for (auto box : {interval_t{0, 4}, interval_t{1, 3}, interval_t{-1, 3}, interval_t{2, 3}})
        {
            double ref = 0.;
            for (int k = box.start; k < box.end; ++k)
            {
                ref += portion<1>(u, L, dl, coarse, std::make_tuple(k))[0];
            }
            auto slice = portion<1>(u, L, dl, coarse, std::make_tuple(box));
            EXPECT_NEAR(slice[0], ref, 1e-12) << "box [" << box.start << ", " << box.end << ")";
        }
    }

    TEST(portion, slice_equals_sum_of_children_2d)
    {
        constexpr std::size_t dim = 2;
        using config              = UniformConfig<dim>;
        using interval_t          = typename config::interval_t;
        constexpr std::size_t L   = 5;
        constexpr std::size_t dl  = 2;
        auto mesh                 = UniformMesh<config>(Box<double, dim>({0, 0}, {1, 1}), L);
        auto u                    = init_nonlinear(mesh);

        const int i = 8, j = 8;
        auto coarse = std::make_tuple(interval_t{i, i + 1}, j);

        // A box negative in one direction and partial in the other (a diagonal-like LBM donor slice).
        const interval_t box_x{-1, 3};
        const interval_t box_y{0, 4};
        double ref = 0.;
        for (int kj = box_y.start; kj < box_y.end; ++kj)
        {
            for (int ki = box_x.start; ki < box_x.end; ++ki)
            {
                ref += portion<1>(u, L, dl, coarse, std::make_tuple(ki, kj))[0];
            }
        }
        auto slice = portion<1>(u, L, dl, coarse, std::make_tuple(box_x, box_y));
        EXPECT_NEAR(slice[0], ref, 1e-12);
    }

    // ------------------------------------------------------------------ vector fields
    // The stream is applied to vector fields (the LBM distributions); reconstruction must act on each
    // component independently. On an affine vector field with two distinct components, the whole-vector
    // reconstruction must return each component's exact value. (The per-component form portion(f, comp,
    // ...) used by the LBM shares the same get_prediction / portion_impl path; it always reads the
    // radius from the mesh config, which the single-level UniformMesh here does not carry, so it is
    // covered on the adapted meshes of test_projection_prediction_roundtrip instead.)

    TEST(portion, vector_reconstructs_each_component)
    {
        constexpr std::size_t dim   = 1;
        constexpr std::size_t ncomp = 2;
        using config                = UniformConfig<dim>;
        using interval_t            = typename config::interval_t;
        constexpr std::size_t L     = 5;
        constexpr std::size_t dl    = 4;
        using mesh_id_t             = typename UniformMesh<config>::mesh_id_t;

        auto mesh = UniformMesh<config>(Box<double, dim>({0}, {1}), L);
        auto u    = make_vector_field<double, ncomp>("u", mesh);
        // component 0 = x, component 1 = 2x: two distinct affine fields.
        for_each_cell(mesh[mesh_id_t::cells],
                      [&](const auto& cell)
                      {
                          const double x = cell.center()[0];
                          u[cell](0)     = x;
                          u[cell](1)     = 2. * x;
                      });

        const int i = 2, child = 3;
        auto pv     = portion<1>(u, L, dl, std::make_tuple(interval_t{i, i + 1}), std::make_tuple(child));
        const double x0 = ((i << dl) + child + .5) / (1 << (L + dl));
        EXPECT_DOUBLE_EQ(pv(0, 0), x0);
        EXPECT_DOUBLE_EQ(pv(0, 1), 2. * x0);
    }

    // Reconstruction across a refinement boundary must use the real finer leaves, not the projected
    // coarse value. A flat delta_l >= 2 portion reconstructs from level l only: where a level-l
    // stencil neighbour is refined, it reads that neighbour's projection (its exact average) and so
    // skips the real detail carried by the finer leaves at the intermediate levels. Descending one
    // level at a time and reading the real leaves gives a different - and more accurate - value.
    //
    // This test pins that difference down, so a real-aware reconstruction can be validated against
    // the descending answer (which it must reproduce) rather than the flat one (which it must not).
    TEST(portion, reconstruction_uses_real_cells_across_refinement)
    {
        constexpr std::size_t dim = 1;
        using config              = mesh_config<dim>;
        using Mesh                = MRMesh<config>;
        using interval_t          = typename Mesh::interval_t;
        using mesh_id_t           = typename Mesh::mesh_id_t;

        // Graded 1D mesh: level-1 leaves 0..3, level-1 cells 4-5 refined to level-2 leaves 8..11,
        // level-1 cells 6-7 refined to level-3 leaves 24..31. Cell 3 (level 1) is a coarse leaf whose
        // right neighbour (cell 4) is refined, so its level-2 children 8,9 are real leaves.
        CellList<dim> cl;
        cl[1][{}].add_interval({0, 4});
        cl[2][{}].add_interval({8, 12});
        cl[3][{}].add_interval({24, 32});
        auto cfg  = config().min_level(1).max_level(3).max_stencil_radius(2);
        auto mesh = mra::make_mesh(cl, cfg);

        auto u = make_scalar_field<double>("u", mesh);
        u.fill(0.);
        for_each_cell(mesh[mesh_id_t::cells],
                      [&](const auto& cell)
                      {
                          u[cell] = std::cos(3. * cell.center()[0]); // non-affine: carries real detail
                      });
        update_ghost_mr(u);

        // Reconstruct the same level-3 cell 15 (right grandchild of the coarse leaf cell 3) two ways.
        // flat: from level 1, delta_l = 2 -> reads level-1 cells only (neighbour via its projection).
        const double flat = portion<1>(u, 1, 2, std::make_tuple(interval_t{3, 4}), std::make_tuple(3))(0);
        // descending / real-aware: from level 2, delta_l = 1 -> its stencil reads level-2 cell 8, a
        // REAL leaf under the refined neighbour. After update_ghost_mr the level-2 strip blends real
        // leaves with prediction ghosts, so this is exactly the real-aware answer.
        const double descending = portion<1>(u, 2, 1, std::make_tuple(interval_t{7, 8}), std::make_tuple(1))(0);

        // The two disagree: the flat reconstruction ignored the neighbour's real level-2 leaf.
        EXPECT_GT(std::abs(flat - descending), 1e-4);

        // ... and the descending one is closer to the true field (both carry the coarse-cell
        // truncation error, but the flat one adds the projection error on top).
        const double truth = std::cos(3. * (15 + 0.5) / (1 << 3));
        EXPECT_LT(std::abs(descending - truth), std::abs(flat - truth));
    }
}
