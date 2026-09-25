// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// The composed prediction maps - reconstruction(), portion() and what the LBM stream is built
// on - read only cells the domain holds, at the boundary too, and reproduce there what the
// operator reproduces everywhere else.
//
// The method is the one the rest of the boundary rewrite uses: fill a field with the cell
// averages of a polynomial the prediction operator reproduces, then **poison** every cell
// outside the domain with NaN and forbid the ghost update from repairing them. A single read
// of an outer ghost turns a reconstructed value into NaN; a wrong stencil turns it into a
// non-zero error. Exactness on the whole reconstructed field is then the whole statement.
//
// The interior is held to more than exactness: the maps must be bit for bit what the centred
// recursion gave before the stencils could shift, which is asserted against a copy of that
// recursion kept here, and the position class must be one rule whether it is read off a box
// or off the domain's rows.

#include <array>
#include <cmath>
#include <map>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include <samurai/algorithm/update.hpp>
#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/numeric/gauss_legendre.hpp>
#include <samurai/prediction_shifts.hpp>
#include <samurai/reconstruction.hpp>
#include <samurai/samurai.hpp>
#include <samurai/subset/node.hpp>

namespace samurai
{
    namespace
    {
        template <std::size_t dim, class TInterval, class F>
        double cell_average(const Cell<dim, TInterval>& cell, F&& f)
        {
            static GaussLegendre<10> gl;
            return gl.template quadrature<1>(cell, f) / std::pow(cell.length, static_cast<double>(dim));
        }

        // An adapted mesh whose refinement reaches the boundary: a sharp ball at the origin
        // corner refines the two '-' boundaries there, so coarse cells touching the boundary
        // sit below fine ones and the composed maps at every level gap are exercised.
        template <std::size_t dim>
        auto adapted_mesh(std::size_t max_level = 5)
        {
            auto cfg  = mesh_config<dim>().min_level(2).max_level(max_level);
            auto mesh = mra::make_mesh(Box<double, dim>{xt::zeros<double>({dim}), xt::ones<double>({dim})}, cfg);

            auto phi = make_scalar_field<double>("phi", mesh);
            for_each_cell(mesh,
                          [&](auto& cell)
                          {
                              const auto c = cell.center();
                              double r     = 0.;
                              for (std::size_t d = 0; d < dim; ++d)
                              {
                                  r += c[d] * c[d];
                              }
                              phi[cell] = (std::sqrt(r) < 0.3) ? 1. : 0.;
                          });
            make_bc<Dirichlet<1>>(phi, 0.);
            make_MRAdapt(phi)(mra_config().epsilon(1e-4));
            return mesh;
        }

        /**
         * The cell averages of @a f on every cell of the reference mesh that the domain holds,
         * NaN on every cell it does not, and the ghost update told it has nothing left to do -
         * so that a consumer reading an outer ghost produces a NaN rather than a plausible
         * number.
         */
        template <class Mesh, class F>
        auto poisoned_field(Mesh& mesh, F&& f)
        {
            using mesh_id_t = typename Mesh::mesh_id_t;

            auto u = make_scalar_field<double>("u", mesh);
            u.fill(std::nan(""));
            for (std::size_t level = mesh[mesh_id_t::reference].min_level(); level <= mesh[mesh_id_t::reference].max_level(); ++level)
            {
                auto inside = intersection(mesh[mesh_id_t::reference][level], mesh.domain(level));
                for_each_cell(mesh,
                              inside,
                              [&](const auto& cell)
                              {
                                  u[cell] = cell_average(cell, f);
                              });
            }
            u.ghosts_updated() = true;
            return u;
        }

        template <std::size_t dim>
        auto reproduced_polynomial()
        {
            constexpr double two_r = 2. * static_cast<double>(mesh_config<dim>::prediction_stencil_radius);
            return [](const auto& x)
            {
                double v = 1.;
                for (std::size_t d = 0; d < dim; ++d)
                {
                    v *= std::pow(x(d) + 0.3, two_r);
                }
                return v;
            };
        }
    }

    // ------------------------------------------------------------------------ reconstruction

    template <std::size_t dim>
    void expect_reconstruction_exact_with_poisoned_ghosts()
    {
        using mesh_id_t = MRMeshId;

        auto mesh    = adapted_mesh<dim>();
        const auto f = reproduced_polynomial<dim>();
        auto u       = poisoned_field(mesh, f);

        auto reconstructed = reconstruction(u);

        std::size_t nb     = 0;
        double max_error   = 0.;
        std::size_t nb_nan = 0;
        for_each_cell(reconstructed.mesh(),
                      [&](const auto& cell)
                      {
                          ++nb;
                          const double value = reconstructed[cell];
                          if (std::isnan(value))
                          {
                              ++nb_nan;
                              return;
                          }
                          max_error = std::max(max_error, std::abs(value - cell_average(cell, f)));
                      });
        EXPECT_GT(nb, 0u);
        EXPECT_EQ(nb_nan, 0u) << "an outer ghost was read";
        EXPECT_LT(max_error, 1e-12);

        // The mesh does reach the boundary at several levels, otherwise the test is vacuous.
        std::size_t boundary_coarse_cells = 0;
        for (std::size_t level = mesh.min_level(); level < mesh.max_level(); ++level)
        {
            auto at_boundary = difference(mesh[mesh_id_t::cells][level], contract(self(mesh.domain(level)), 1));
            for_each_cell(mesh,
                          at_boundary,
                          [&](const auto&)
                          {
                              ++boundary_coarse_cells;
                          });
        }
        EXPECT_GT(boundary_coarse_cells, 0u);
    }

    TEST(reconstruction_boundary, reconstruction_reads_no_outer_ghost_and_reproduces_the_polynomial_1d)
    {
        expect_reconstruction_exact_with_poisoned_ghosts<1>();
    }

    TEST(reconstruction_boundary, reconstruction_reads_no_outer_ghost_and_reproduces_the_polynomial_2d)
    {
        expect_reconstruction_exact_with_poisoned_ghosts<2>();
    }

    TEST(reconstruction_boundary, reconstruction_is_not_exact_one_degree_above)
    {
        // Exactness alone would also hold for an operator of higher order than the one claimed.
        constexpr std::size_t dim = 2;
        constexpr double two_r    = 2. * static_cast<double>(mesh_config<dim>::prediction_stencil_radius);

        auto mesh    = adapted_mesh<dim>();
        const auto f = [](const auto& x)
        {
            return std::pow(x(0) + 0.3, two_r + 1.);
        };
        auto u = poisoned_field(mesh, f);

        auto reconstructed = reconstruction(u);

        double max_error = 0.;
        for_each_cell(reconstructed.mesh(),
                      [&](const auto& cell)
                      {
                          const double value = reconstructed[cell];
                          ASSERT_FALSE(std::isnan(value));
                          max_error = std::max(max_error, std::abs(value - cell_average(cell, f)));
                      });
        EXPECT_GT(max_error, 1e-8);
    }

    // ------------------------------------------------------------------------------- portion

    TEST(reconstruction_boundary, portion_of_a_boundary_cell_reads_no_outer_ghost)
    {
        // Every child of every coarse real cell, scalar form and slice form, on a mesh whose
        // coarse cells touch the boundary.
        constexpr std::size_t dim = 2;
        using mesh_id_t           = MRMeshId;
        using interval_t          = typename MRMesh<mesh_config<dim>>::interval_t;
        using value_t             = typename interval_t::value_t;

        auto mesh    = adapted_mesh<dim>();
        const auto f = reproduced_polynomial<dim>();
        auto u       = poisoned_field(mesh, f);

        const std::size_t fine = mesh.max_level();
        std::size_t nb         = 0;
        for (std::size_t level = mesh.min_level(); level < fine; ++level)
        {
            const std::size_t delta_l = fine - level;
            const value_t width       = static_cast<value_t>(1) << delta_l;

            for_each_interval(
                mesh[mesh_id_t::cells][level],
                [&](std::size_t, const auto& i, const auto& index)
                {
                    const value_t j = index[0];

                    // Slice form: the sum over the whole box of children, at once.
                    auto whole = portion(u, level, delta_l, std::make_tuple(i, j), std::make_tuple(interval_t{0, width}, interval_t{0, width}));

                    for (value_t x = i.start; x < i.end; ++x)
                    {
                        double exact_sum = 0.;
                        for (value_t jj = 0; jj < width; ++jj)
                        {
                            for (value_t ii = 0; ii < width; ++ii)
                            {
                                auto value = portion(u, level, delta_l, std::make_tuple(interval_t{x, x + 1}, j), std::make_tuple(ii, jj));

                                typename Cell<dim, interval_t>::indices_t child_indices;
                                child_indices[0] = (x << delta_l) + ii;
                                child_indices[1] = (j << delta_l) + jj;
                                const Cell<dim, interval_t> child(mesh.origin_point(), mesh.scaling_factor(), fine, child_indices, 0);
                                const double exact = cell_average(child, f);
                                exact_sum += exact;

                                ++nb;
                                ASSERT_FALSE(std::isnan(value(0))) << "portion read an outer ghost";
                                EXPECT_NEAR(value(0), exact, 1e-12);
                            }
                        }
                        EXPECT_NEAR(whole(static_cast<std::size_t>(x - i.start)), exact_sum, 1e-11);
                    }
                });
        }
        EXPECT_GT(nb, 0u);
    }

    // ------------------------------------------------------ the interior maps are unchanged

    namespace
    {
        /**
         * The recursion as it stood before the stencils could shift: the parent's map plus the
         * centred tensor-product correction of the other nodes, in the same node order. Kept
         * here so that "bit for bit" is asserted against something, not remembered.
         */
        template <std::size_t order, class value_t, std::size_t dim>
        prediction_map<dim, value_t> centred_map(std::size_t level, const std::array<value_t, dim>& indices)
        {
            using map_t = prediction_map<dim, value_t>;
            using key_t = std::tuple<std::size_t, std::array<value_t, dim>>;
            static std::map<key_t, map_t> memo;

            const key_t key{level, indices};
            if (auto it = memo.find(key); it != memo.end())
            {
                return it->second;
            }
            if (level == 0)
            {
                return memo[key] = map_t{indices};
            }

            std::array<value_t, dim> parent;
            std::array<const std::array<double, 2 * order + 1>*, dim> coeffs;
            for (std::size_t d = 0; d < dim; ++d)
            {
                parent[d] = indices[d] >> 1;
                coeffs[d] = &prediction_coefficients<order>(static_cast<std::size_t>(indices[d] & 1), 0).c;
            }

            map_t out = centred_map<order, value_t, dim>(level - 1, parent);

            // Node order of detail::multi_dim_loop: the last direction outermost, x innermost.
            constexpr std::size_t size = 2 * order + 1;
            std::array<std::size_t, dim> node{};
            const auto visit = [&](auto& self, std::size_t d) -> void
            {
                if (d == dim)
                {
                    bool centre = true;
                    double c    = 1.;
                    std::array<value_t, dim> offsets;
                    for (std::size_t k = 0; k < dim; ++k)
                    {
                        centre     = centre && node[k] == order;
                        c          = c * (*coeffs[k])[node[k]];
                        offsets[k] = parent[k] + static_cast<value_t>(node[k]) - static_cast<value_t>(order);
                    }
                    if (!centre)
                    {
                        out += c * centred_map<order, value_t, dim>(level - 1, offsets);
                    }
                    return;
                }
                for (node[dim - 1 - d] = 0; node[dim - 1 - d] < size; ++node[dim - 1 - d])
                {
                    self(self, d + 1);
                }
            };
            visit(visit, 0);

            return memo[key] = out;
        }
    }

    TEST(reconstruction_boundary, the_interior_maps_are_bit_identical_to_the_centred_recursion)
    {
        using value_t = default_config::value_t;

        // 1D up to a gap of 5, 2D up to 3: every child of the reference cell, and the two
        // children a stream slice places in the neighbouring cells.
        for (std::size_t gap = 1; gap <= 5; ++gap)
        {
            for (value_t ii = -1; ii <= (value_t{1} << gap); ++ii)
            {
                const auto& got     = prediction<1>(gap, ii);
                const auto expected = centred_map<1, value_t, 1>(gap, {ii});
                ASSERT_EQ(got.coeff.size(), expected.coeff.size()) << "gap " << gap << " child " << ii;
                for (const auto& [offset, weight] : expected.coeff)
                {
                    const auto it = got.coeff.find(offset);
                    ASSERT_NE(it, got.coeff.end());
                    EXPECT_EQ(it->second, weight) << "gap " << gap << " child " << ii << " offset " << offset[0];
                }
            }
        }
        for (std::size_t gap = 1; gap <= 3; ++gap)
        {
            for (value_t jj = -1; jj <= (value_t{1} << gap); ++jj)
            {
                for (value_t ii = -1; ii <= (value_t{1} << gap); ++ii)
                {
                    const auto& got     = prediction<1>(gap, ii, jj);
                    const auto expected = centred_map<1, value_t, 2>(gap, {ii, jj});
                    ASSERT_EQ(got.coeff.size(), expected.coeff.size()) << "gap " << gap << " child (" << ii << "," << jj << ")";
                    for (const auto& [offset, weight] : expected.coeff)
                    {
                        const auto it = got.coeff.find(offset);
                        ASSERT_NE(it, got.coeff.end());
                        EXPECT_EQ(it->second, weight) << "gap " << gap << " child (" << ii << "," << jj << ")";
                    }
                }
            }
        }
    }

    TEST(reconstruction_boundary, a_boundary_map_differs_from_the_centred_one_and_reads_inward_only)
    {
        // A reference cell sitting on the low boundary in x: its maps must not reach the cell
        // at offset -1, which the centred maps do.
        using value_t = default_config::value_t;
        using class_t = PredictionPositionClass<prediction_class_reach<1>, 1>;

        class_t on_the_boundary = class_t::interior();
        on_the_boundary.low[0]  = 0;

        for (std::size_t gap = 1; gap <= 4; ++gap)
        {
            for (value_t ii = 0; ii < (value_t{1} << gap); ++ii)
            {
                const auto& shifted = prediction<1>(on_the_boundary, gap, ii);
                for (const auto& [offset, weight] : shifted.coeff)
                {
                    EXPECT_GE(offset[0], 0) << "gap " << gap << " child " << ii << " reads outside the domain";
                }
                // Conservative: the children's weights sum to 2^gap times the parent's, i.e. the
                // map of a child has total weight 1.
                double total = 0.;
                for (const auto& kv : shifted.coeff)
                {
                    total += kv.second;
                }
                EXPECT_NEAR(total, 1., 1e-14);
            }
            EXPECT_NE(prediction<1>(on_the_boundary, gap, value_t{0}).coeff, prediction<1>(gap, value_t{0}).coeff);
        }
    }

    // ----------------------------------------------------- the position class, both paths

    namespace
    {
        template <std::size_t dim>
        using lca_t = LevelCellArray<dim>;

        template <std::size_t dim>
        using interval_t = typename lca_t<dim>::interval_t;

        template <std::size_t dim>
        using value_t = typename interval_t<dim>::value_t;

        template <std::size_t dim>
        using index_t = xt::xtensor_fixed<value_t<dim>, xt::xshape<dim - 1>>;

        template <std::size_t dim>
        using box_t = std::array<std::pair<value_t<dim>, value_t<dim>>, dim>;

        template <int reach, std::size_t dim>
        struct ClassRun
        {
            value_t<dim> start;
            value_t<dim> end;
            PredictionPositionClass<reach, dim> cls;

            bool operator==(const ClassRun& o) const
            {
                return start == o.start && end == o.end && cls == o.cls;
            }
        };

        template <int reach, std::size_t dim>
        std::ostream& operator<<(std::ostream& os, const ClassRun<reach, dim>& r)
        {
            os << "[" << r.start << "," << r.end << ") rows {";
            for (std::size_t k = 0; k < r.cls.rows; ++k)
            {
                os << (k == 0 ? "" : " ") << int{r.cls.low[k]} << "/" << int{r.cls.high[k]};
            }
            return os << "}";
        }

        template <int reach, std::size_t dim>
        void expect_both_class_paths_agree(const box_t<dim>& box, const std::array<value_t<dim>, dim>& period)
        {
            xt::xtensor_fixed<value_t<dim>, xt::xshape<dim>> lo;
            xt::xtensor_fixed<value_t<dim>, xt::xshape<dim>> hi;
            for (std::size_t d = 0; d < dim; ++d)
            {
                lo[d] = box[d].first;
                hi[d] = box[d].second;
            }
            const lca_t<dim> domain{
                0,
                Box<value_t<dim>, dim>{lo, hi}
            };

            const PredictionDomain<dim, interval_t<dim>> boxed{domain, period, true, box};
            const PredictionDomain<dim, interval_t<dim>> scanned{domain, period, false, box};

            const interval_t<dim> i{box[0].first - reach - 1, box[0].second + reach + 1};

            const auto runs_of = [&](const auto& view, const index_t<dim>& index)
            {
                std::vector<ClassRun<reach, dim>> out;
                for_each_prediction_position_run<reach>(view,
                                                        i,
                                                        index,
                                                        [&](const auto& run, const auto& cls)
                                                        {
                                                            out.push_back({run.start, run.end, cls});
                                                        });
                return out;
            };

            const auto check = [&](const index_t<dim>& index)
            {
                EXPECT_EQ(runs_of(boxed, index), runs_of(scanned, index)) << "at transverse index " << index;
            };

            if constexpr (dim == 1)
            {
                check(index_t<dim>{});
            }
            else
            {
                for (auto j = box[1].first - reach - 1; j < box[1].second + reach + 1; ++j)
                {
                    check(index_t<dim>{j});
                }
            }
        }
    }

    TEST(reconstruction_boundary, the_position_class_is_one_rule_on_a_box)
    {
        expect_both_class_paths_agree<4, 1>({{{-2, 11}}}, {0});
        expect_both_class_paths_agree<4, 1>({{{-2, 11}}}, {13});
        expect_both_class_paths_agree<4, 2>(
            {
                {{0, 12}, {-1, 9}}
        },
            {0, 0});
        expect_both_class_paths_agree<4, 2>(
            {
                {{0, 12}, {-1, 9}}
        },
            {12, 0});
        expect_both_class_paths_agree<4, 2>(
            {
                {{0, 12}, {-1, 9}}
        },
            {0, 10});
        expect_both_class_paths_agree<7, 2>(
            {
                {{0, 12}, {-1, 9}}
        },
            {0, 0});
    }
}
