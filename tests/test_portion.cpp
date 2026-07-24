#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <map>
#include <set>
#include <utility>

#include <samurai/algorithm.hpp>
#include <samurai/algorithm/update_ghost_mr.hpp>
#include <samurai/box.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/field.hpp>
#include <samurai/mesh_config.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/reconstruction.hpp>
#include <samurai/samurai.hpp>
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
            const int n    = 1 << dl;
            auto column    = portion<1>(u, L, dl, std::make_tuple(interval_t{i, i + 1}), std::make_tuple(interval_t{0, n}));
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
            const int n = 1 << dl;
            auto column = portion<1>(u, L, dl, std::make_tuple(interval_t{i, i + 1}, j), std::make_tuple(interval_t{0, n}, interval_t{0, n}));
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
        for (auto box : {
                 interval_t{0,  4},
                 interval_t{1,  3},
                 interval_t{-1, 3},
                 interval_t{2,  3}
        })
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
        auto pv         = portion<1>(u, L, dl, std::make_tuple(interval_t{i, i + 1}), std::make_tuple(child));
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

    // ------------------------------------------------------------------------------------
    //  exact-reconstruction reconstruction (reconstruct_exact): on an adapted mesh it must equal
    //  the multiresolution reconstruction capped at the donor's base level (finer real cells
    //  override, coarser-neighbour ghosts kept as the flat path sees them), reduce to the flat
    //  portion() when the cone stays clear of refinement, and actually differ from flat at a
    //  refinement interface.
    // ------------------------------------------------------------------------------------

    // Deterministic graded staircase L1..L5 (wide base) so a delta_l = 4 cone spans the L2, L3
    // and L4 real cells at once (nested-refinement / double-count case).
    TEST(portion_exact, graded_staircase_1D)
    {
        constexpr std::size_t dim = 1;
        CellList<dim> cl;
        cl[1][{}].add_interval({0, 8});     // [0,128)   in level-5 units
        cl[2][{}].add_interval({16, 18});   // [128,144)
        cl[3][{}].add_interval({36, 38});   // [144,152)
        cl[4][{}].add_interval({76, 78});   // [152,156)
        cl[5][{}].add_interval({156, 160}); // [156,160)

        auto cfg                = mesh_config<dim>().min_level(1).max_level(5).max_stencil_radius(2);
        auto mesh               = mra::make_mesh(cl, cfg);
        using mesh_t            = decltype(mesh);
        using mesh_id_t         = mesh_t::mesh_id_t;
        using interval_t        = mesh_t::interval_t;
        using value_t           = interval_t::value_t;
        constexpr std::size_t r = mesh_t::config_t::prediction_stencil_radius;

        auto u = make_scalar_field<double>("u", mesh);
        for_each_cell(mesh[mesh_id_t::cells],
                      [&](const auto& c)
                      {
                          u[c] = std::cos(3. * c.center()[0]);
                      });
        update_ghost_mr(u);

        auto stored = make_real_backed_lca(mesh);
        auto read   = [&](std::size_t l, const std::array<value_t, dim>& c) -> double
        {
            return u(l, interval_t{c[0], c[0] + 1})[0];
        };

        // Independent reference: capped recursion using a std::set membership (a different
        // mechanism than the find()-based test inside reconstruct_exact).
        std::vector<std::set<value_t>> in_set(mesh.max_level() + 1);
        for (std::size_t l = mesh.min_level(); l <= mesh.max_level(); ++l)
        {
            for (auto id : {mesh_id_t::cells, mesh_id_t::proj_cells})
            {
                for_each_interval(mesh[id][l],
                                  [&](std::size_t, const auto& i, const auto&)
                                  {
                                      for (value_t k = i.start; k < i.end; ++k)
                                      {
                                          in_set[l].insert(k);
                                      }
                                  });
            }
        }
        std::function<double(std::size_t, std::size_t, value_t)> ref = [&](std::size_t l0, std::size_t m, value_t i) -> double
        {
            if (m == l0 || in_set[m].count(i))
            {
                return u(m, interval_t{i, i + 1})[0];
            }
            double res = 0.;
            for (const auto& kv : prediction<r, value_t>(1, i & 1).coeff)
            {
                res += kv.second * ref(l0, m - 1, (i >> 1) + kv.first[0]);
            }
            return res;
        };

        auto flat_portion = [&](std::size_t l0, std::size_t delta_l, value_t g) -> double
        {
            value_t coarse = g >> delta_l;
            value_t local  = g - (coarse << delta_l);
            return portion<r>(u, l0, delta_l, std::make_tuple(interval_t{coarse, coarse + 1}), std::make_tuple(local))[0];
        };

        std::size_t checked       = 0;
        std::size_t nb_nontrivial = 0;
        for (std::size_t l0 = mesh.min_level(); l0 < mesh.max_level(); ++l0)
        {
            std::size_t delta_l = mesh.max_level() - l0;
            for (value_t g = 0; g < 160; ++g)
            {
                double flat = std::numeric_limits<double>::quiet_NaN();
                double expected;
                double got;
                try
                {
                    flat     = flat_portion(l0, delta_l, g);
                    expected = ref(l0, mesh.max_level(), g);
                    std::map<std::pair<std::size_t, std::array<value_t, dim>>, double> memo;
                    got = reconstruct_exact<r>(l0, delta_l, std::array<value_t, dim>{g}, read, stored, memo);
                }
                catch (...)
                {
                    continue; // g whose flat stencil leaves all_cells: not a case the stream hits
                }
                ++checked;
                // main property: the capped multiresolution reconstruction, to machine precision
                EXPECT_NEAR(got, expected, 1e-13) << "l0=" << l0 << " g=" << g;
                if (std::abs(expected - flat) > 1e-9)
                {
                    ++nb_nontrivial; // correction active (cone crosses a refinement)
                }
                else
                {
                    // no refinement in the cone => reduces to the flat prediction (up to the
                    // rounding of a cascade vs a single flattened stencil)
                    EXPECT_NEAR(got, flat, 1e-12) << "l0=" << l0 << " g=" << g;
                }
            }
        }
        EXPECT_GT(checked, 0u);
        EXPECT_GT(nb_nontrivial, 0u); // the correction is actually active at the interfaces
    }

    // Adapted 2D mesh (sharp front => graded refinement): exercises the dim-generic machinery
    // and nested refinement in 2D against the same capped-recursion reference.
    TEST(portion_exact, adapted_mesh_2D)
    {
        ::samurai::initialize();
        constexpr std::size_t dim = 2;
        using box_t               = Box<double, dim>;

        auto fill = [](auto& field)
        {
            auto& mesh = field.mesh();
            using mid  = std::decay_t<decltype(mesh)>::mesh_id_t;
            for_each_cell(mesh[mid::cells],
                          [&](const auto& c)
                          {
                              auto x   = c.center()[0];
                              auto y   = c.center()[1];
                              field[c] = std::tanh(20. * (x - 0.5)) + std::cos(4. * y);
                          });
        };

        auto cfg  = mesh_config<dim>().min_level(2).max_level(6);
        auto mesh = mra::make_mesh(
            box_t{
                {0., 0.},
                {1., 1.}
        },
            cfg);
        auto u = make_scalar_field<double>("u", mesh);
        fill(u);
        make_MRAdapt(u)(mra_config().epsilon(1e-3).regularity(2.));
        fill(u); // re-fill on the adapted topology
        update_ghost_mr(u);

        using mesh_t            = decltype(mesh);
        using mesh_id_t         = mesh_t::mesh_id_t;
        using interval_t        = mesh_t::interval_t;
        using value_t           = interval_t::value_t;
        constexpr std::size_t r = mesh_t::config_t::prediction_stencil_radius;
        const std::size_t L     = mesh.max_level();

        auto stored = make_real_backed_lca(mesh);
        auto read   = [&](std::size_t l, const std::array<value_t, dim>& c) -> double
        {
            return u(l, interval_t{c[0], c[0] + 1}, c[1])[0];
        };

        std::vector<std::set<std::pair<value_t, value_t>>> in_set(L + 1);
        for (std::size_t l = mesh.min_level(); l <= L; ++l)
        {
            for (auto id : {mesh_id_t::cells, mesh_id_t::proj_cells})
            {
                for_each_interval(mesh[id][l],
                                  [&](std::size_t, const auto& i, const auto& index)
                                  {
                                      for (value_t k = i.start; k < i.end; ++k)
                                      {
                                          in_set[l].insert({k, index[0]});
                                      }
                                  });
            }
        }
        std::function<double(std::size_t, std::size_t, value_t, value_t)> ref = [&](std::size_t l0, std::size_t m, value_t i, value_t j) -> double
        {
            if (m == l0 || in_set[m].count({i, j}))
            {
                return u(m, interval_t{i, i + 1}, j)[0];
            }
            double res = 0.;
            for (const auto& kv : prediction<r, value_t>(1, i & 1, j & 1).coeff)
            {
                res += kv.second * ref(l0, m - 1, (i >> 1) + kv.first[0], (j >> 1) + kv.first[1]);
            }
            return res;
        };

        std::size_t checked = 0, nb_nontrivial = 0;
        for (std::size_t l0 = mesh.min_level(); l0 < L; ++l0)
        {
            std::size_t delta_l = L - l0;
            value_t nsub        = value_t{1} << delta_l;
            for_each_interval(
                mesh[mesh_id_t::cells][l0],
                [&](std::size_t, const auto& ii, const auto& idx)
                {
                    for (value_t ci = ii.start; ci < ii.end; ++ci)
                    {
                        for (value_t sy = 0; sy < nsub; ++sy)
                        {
                            for (value_t sx = 0; sx < nsub; ++sx)
                            {
                                value_t gx = (ci << delta_l) + sx;
                                value_t gy = (idx[0] << delta_l) + sy;
                                double expected, got, flat;
                                try
                                {
                                    expected = ref(l0, L, gx, gy);
                                    std::map<std::pair<std::size_t, std::array<value_t, dim>>, double> memo;
                                    got  = reconstruct_exact<r>(l0, delta_l, std::array<value_t, dim>{gx, gy}, read, stored, memo);
                                    flat = portion<r>(u,
                                                      l0,
                                                      delta_l,
                                                      std::make_tuple(interval_t{ci, ci + 1}, idx[0]),
                                                      std::make_tuple(sx, sy))[0];
                                }
                                catch (...)
                                {
                                    continue;
                                }
                                ++checked;
                                EXPECT_NEAR(got, expected, 1e-12) << "l0=" << l0 << " g=(" << gx << "," << gy << ")";
                                if (std::abs(expected - flat) > 1e-9)
                                {
                                    ++nb_nontrivial;
                                }
                            }
                        }
                    }
                });
        }
        EXPECT_GT(checked, 0u);
        EXPECT_GT(nb_nontrivial, 0u);
        ::samurai::finalize();
    }

    // The vectorised reconstruct_exact_box (the fast path used by the LBM stream) must be
    // bit-identical to the scalar reconstruct_exact, cell for cell, over a whole sub-cell box.
    TEST(portion_exact, box_matches_scalar_1D)
    {
        constexpr std::size_t dim = 1;
        CellList<dim> cl;
        cl[1][{}].add_interval({0, 8});
        cl[2][{}].add_interval({16, 18});
        cl[3][{}].add_interval({36, 38});
        cl[4][{}].add_interval({76, 78});
        cl[5][{}].add_interval({156, 160});

        auto cfg                = mesh_config<dim>().min_level(1).max_level(5).max_stencil_radius(2);
        auto mesh               = mra::make_mesh(cl, cfg);
        using mesh_t            = decltype(mesh);
        using mesh_id_t         = mesh_t::mesh_id_t;
        using interval_t        = mesh_t::interval_t;
        using value_t           = interval_t::value_t;
        constexpr std::size_t r = mesh_t::config_t::prediction_stencil_radius;
        const std::size_t L     = mesh.max_level();

        auto u = make_scalar_field<double>("u", mesh);
        for_each_cell(mesh[mesh_id_t::cells],
                      [&](const auto& c)
                      {
                          u[c] = std::cos(3. * c.center()[0]);
                      });
        update_ghost_mr(u);

        auto stored = make_real_backed_lca(mesh);
        auto read   = [&](std::size_t l, const std::array<value_t, dim>& c) -> double
        {
            return u(l, interval_t{c[0], c[0] + 1})[0];
        };

        std::size_t checked = 0;
        for (std::size_t l0 = mesh.min_level(); l0 < L; ++l0)
        {
            std::size_t delta_l = L - l0;
            for_each_interval(mesh[mesh_id_t::cells][l0],
                              [&](std::size_t, const auto& i, const auto&)
                              {
                                  std::array<value_t, dim> lo{i.start << delta_l};
                                  std::array<value_t, dim> hi{i.end << delta_l};
                                  detail::ra_box<dim, value_t> box{lo, hi};
                                  std::vector<double> out;
                                  try
                                  {
                                      reconstruct_exact_box<r>(l0, L, lo, hi, read, stored, out);
                                  }
                                  catch (...)
                                  {
                                      return;
                                  }
                                  for (std::size_t f = 0; f < box.count(); ++f)
                                  {
                                      auto g = box.coord(f);
                                      exact_reconstruction_memo_t<dim, value_t> memo;
                                      double scalar;
                                      try
                                      {
                                          scalar = reconstruct_exact<r>(l0, delta_l, g, read, stored, memo);
                                      }
                                      catch (...)
                                      {
                                          continue;
                                      }
                                      EXPECT_DOUBLE_EQ(out[f], scalar) << "l0=" << l0 << " g=" << g[0];
                                      ++checked;
                                  }
                              });
        }
        EXPECT_GT(checked, 0u);
    }
}
