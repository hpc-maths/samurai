// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// The prediction kernels (prediction_op) against the tensor product written out
// cell by cell.
//
// The kernels contract the tensor product one direction at a time and look the stencil
// rows up once per interval, so they share no code with the reference below, which reads
// every stencil cell through get_index and multiplies the coefficients of every
// direction. The two must agree to round-off, for both kernels (the cells of an interval
// from their parents, and the children of every cell of an interval), in every dimension
// the kernels are written for, including the high ones no mesh reaches yet: the fields
// here are bare arrays over a CellArray.

#include <cmath>
#include <cstddef>
#include <functional>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <samurai/cell_array.hpp>
#include <samurai/level_cell_array.hpp>
#include <samurai/level_cell_list.hpp>
#include <samurai/numeric/prediction.hpp>

namespace samurai
{
    namespace
    {
        using interval_t = default_config::interval_t;
        using value_t    = interval_t::value_t;

        // What prediction_op needs from a mesh: the cell lookup and the reference cells.
        template <std::size_t D>
        struct ArrayMesh
        {
            static constexpr std::size_t dim = D;
            using interval_t                 = samurai::interval_t;

            enum class mesh_id_t
            {
                reference
            };

            CellArray<D, interval_t> cells;

            const CellArray<D, interval_t>& operator[](mesh_id_t) const
            {
                return cells;
            }

            template <class... T>
            auto get_index(std::size_t level, value_t i, T... index) const
            {
                return cells.get_index(level, i, index...);
            }
        };

        // What prediction_op needs from a field: its mesh and its values, cell-major.
        template <std::size_t D, std::size_t NC>
        struct ArrayField
        {
            using mesh_t                        = ArrayMesh<D>;
            static constexpr std::size_t n_comp = NC;

            const mesh_t* m;
            std::vector<double> values;

            const mesh_t& mesh() const
            {
                return *m;
            }

            const double* data() const
            {
                return values.data();
            }

            double* data()
            {
                return values.data();
            }
        };

        using index_t = std::vector<value_t>; // coordinates above x

        // Every row index of [lo, hi)^n, the outermost direction slowest.
        void for_each_row(std::size_t n, value_t lo, value_t hi, const std::function<void(const index_t&)>& f)
        {
            index_t idx(n, lo);
            if (n == 0)
            {
                f(idx);
                return;
            }
            while (true)
            {
                f(idx);
                std::size_t d = 0;
                while (d < n && ++idx[d] == hi)
                {
                    idx[d++] = lo;
                }
                if (d == n)
                {
                    return;
                }
            }
        }

        template <std::size_t D>
        auto to_fixed(const index_t& idx)
        {
            xt::xtensor_fixed<value_t, xt::xshape<D - 1>> out;
            for (std::size_t d = 0; d + 1 < D; ++d)
            {
                out[d] = idx[d];
            }
            return out;
        }

        // Coarse cells at `level`: in every direction above x, [0, m); along x, the two
        // runs [0, m) and [m + 2, 2m + 2), so that a row holds two intervals and the
        // lookup has to search. The level above holds the children of the coarse rows
        // [r, m - r), the ones whose stencil is complete.
        template <std::size_t D, std::size_t r>
        ArrayMesh<D> make_mesh(std::size_t level, value_t m)
        {
            LevelCellList<D, interval_t> coarse(level);
            LevelCellList<D, interval_t> fine(level + 1);
            for_each_row(D - 1,
                         0,
                         m,
                         [&](const index_t& idx)
                         {
                             coarse[to_fixed<D>(idx)].add_interval({0, m});
                             coarse[to_fixed<D>(idx)].add_interval({m + 2, 2 * m + 2});
                         });
            for_each_row(D - 1,
                         2 * static_cast<value_t>(r),
                         2 * (m - static_cast<value_t>(r)),
                         [&](const index_t& idx)
                         {
                             fine[to_fixed<D>(idx)].add_interval({0, 2 * m});
                             fine[to_fixed<D>(idx)].add_interval({2 * m + 4, 4 * m + 4});
                         });

            ArrayMesh<D> mesh;
            mesh.cells[level]     = LevelCellArray<D, interval_t>(coarse);
            mesh.cells[level + 1] = LevelCellArray<D, interval_t>(fine);
            mesh.cells.update_index();
            return mesh;
        }

        // The value of the fine cell (x, idx) at `level + 1`, as the full tensor product
        // over the stencil of its parent.
        template <std::size_t D, std::size_t r, std::size_t NC>
        double reference(const ArrayField<D, NC>& u, std::size_t level, value_t x, const index_t& idx, std::size_t n)
        {
            const auto& cells = u.mesh().cells;
            double v          = 0.;
            for_each_row(
                D,
                -static_cast<value_t>(r),
                static_cast<value_t>(r) + 1,
                [&](const index_t& k)
                {
                    double coeff = prediction_coefficients<r>(static_cast<std::size_t>(x & 1), 0).c[static_cast<std::size_t>(k[0]) + r];
                    xt::xtensor_fixed<value_t, xt::xshape<D>> coord;
                    coord[0] = (x >> 1) + k[0];
                    for (std::size_t d = 1; d < D; ++d)
                    {
                        coeff *= prediction_coefficients<r>(static_cast<std::size_t>(idx[d - 1] & 1), 0).c[static_cast<std::size_t>(k[d]) + r];
                        coord[d] = (idx[d - 1] >> 1) + k[d];
                    }
                    const auto cell = static_cast<std::size_t>(cells.get_index(level, coord));
                    v += coeff * u.values[cell * NC + n];
                });
            return v;
        }

        // Runs both kernels over every interval whose stencil is complete, and compares
        // each predicted value with the reference. `m` is the extent of the coarse block;
        // one predicted cell in `sample` is checked, since the reference costs 3^D lookups.
        template <std::size_t D, std::size_t r, std::size_t NC>
        void check_kernels(value_t m, std::size_t sample = 1)
        {
            constexpr std::size_t level = 1;
            const auto mesh             = make_mesh<D, r>(level, m);

            ArrayField<D, NC> src{&mesh, {}};
            src.values.resize(mesh.cells.nb_cells() * NC);
            std::mt19937 gen(42);
            std::uniform_real_distribution<double> dist(-1., 1.);
            for (auto& v : src.values)
            {
                v = dist(gen);
            }

            const auto lo = static_cast<value_t>(r);
            const auto hi = m - static_cast<value_t>(r);
            ASSERT_LT(lo, hi) << "the coarse block is too small for the stencil";

            std::size_t checked = 0;
            double max_error    = 0.;
            std::size_t seen    = 0;
            auto compare        = [&](const ArrayField<D, NC>& dest, value_t x, const index_t& fine_idx)
            {
                if (seen++ % sample != 0)
                {
                    return;
                }
                xt::xtensor_fixed<value_t, xt::xshape<D>> coord;
                coord[0] = x;
                for (std::size_t d = 1; d < D; ++d)
                {
                    coord[d] = fine_idx[d - 1];
                }
                const auto cell = static_cast<std::size_t>(mesh.cells.get_index(level + 1, coord));
                for (std::size_t n = 0; n < NC; ++n)
                {
                    const double expected = reference<D, r>(src, level, x, fine_idx, n);
                    max_error             = std::max(max_error, std::abs(dest.values[cell * NC + n] - expected));
                    ++checked;
                }
            };

            // The cells of a fine interval from their parents: every fine row of the
            // block, with intervals that start and end on either parity.
            {
                ArrayField<D, NC> dest = src;
                for_each_row(D - 1,
                             2 * lo,
                             2 * hi,
                             [&](const index_t& fine_idx)
                             {
                                 // Odd start and end when the block leaves room for them.
                                 const interval_t i = (2 * hi - 1 > 2 * lo + 1) ? interval_t{2 * lo + 1, 2 * hi - 1}
                                                                                : interval_t{2 * lo, 2 * hi};
                                 prediction_op<D, interval_t>(level + 1, i, to_fixed<D>(fine_idx))(Dim<D>{},
                                                                                                   dest,
                                                                                                   src,
                                                                                                   std::integral_constant<std::size_t, r>{},
                                                                                                   std::integral_constant<bool, false>{});
                                 for (value_t x = i.start; x < i.end; ++x)
                                 {
                                     compare(dest, x, fine_idx);
                                 }
                             });
            }

            // The children of every cell of a coarse interval.
            {
                ArrayField<D, NC> dest = src;
                for_each_row(D - 1,
                             lo,
                             hi,
                             [&](const index_t& idx)
                             {
                                 const interval_t i{lo, hi};
                                 prediction_op<D, interval_t>(level, i, to_fixed<D>(idx))(Dim<D>{},
                                                                                          dest,
                                                                                          src,
                                                                                          std::integral_constant<std::size_t, r>{},
                                                                                          std::integral_constant<bool, true>{});
                                 for_each_row(D - 1,
                                              0,
                                              2,
                                              [&](const index_t& q)
                                              {
                                                  index_t fine_idx(D - 1);
                                                  for (std::size_t d = 0; d + 1 < D; ++d)
                                                  {
                                                      fine_idx[d] = 2 * idx[d] + q[d];
                                                  }
                                                  for (value_t x = 2 * i.start; x < 2 * i.end; ++x)
                                                  {
                                                      compare(dest, x, fine_idx);
                                                  }
                                              });
                             });
            }

            EXPECT_GT(checked, 0U);
            EXPECT_LT(max_error, 1e-13) << "dim " << D << ", radius " << r << ", " << NC << " components";
        }
    }

    TEST(prediction_kernel, matches_tensor_product_1d)
    {
        check_kernels<1, 1, 1>(5);
        check_kernels<1, 2, 3>(7);
        check_kernels<1, 3, 1>(9);
    }

    TEST(prediction_kernel, matches_tensor_product_2d)
    {
        check_kernels<2, 1, 1>(5);
        check_kernels<2, 2, 2>(7);
        check_kernels<2, 3, 1>(8);
    }

    TEST(prediction_kernel, matches_tensor_product_3d)
    {
        check_kernels<3, 1, 1>(5);
        check_kernels<3, 1, 4>(4);
        check_kernels<3, 2, 1>(6);
    }

    // No mesh reaches these dimensions yet; the kernels must already be right there.
    TEST(prediction_kernel, matches_tensor_product_4d_to_6d)
    {
        check_kernels<4, 1, 1>(4);
        check_kernels<4, 2, 1>(5);
        check_kernels<5, 1, 2>(3);
        check_kernels<6, 1, 1>(3);
    }

    TEST(prediction_kernel, matches_tensor_product_10d)
    {
        check_kernels<10, 1, 1>(3, 64);
    }
}
