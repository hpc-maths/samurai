// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// The shift query has two paths - arithmetic on the bounding box where the domain is a box,
// the row scan otherwise - and they must be one rule. This file holds them to that, cell for
// cell, and pins the mesh-side cache the box path reads from.

#include <array>
#include <vector>

#include <gtest/gtest.h>

#include <samurai/amr/mesh.hpp>
#include <samurai/box.hpp>
#include <samurai/level_cell_array.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/prediction_shifts.hpp>
#include <samurai/subset/node.hpp>

namespace samurai
{
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
        using period_t = std::array<value_t<dim>, dim>;

        template <std::size_t dim>
        using box_t = std::array<std::pair<value_t<dim>, value_t<dim>>, dim>;

        /// The domain `prod_d [lo_d, hi_d)` at level 0.
        template <std::size_t dim>
        lca_t<dim> make_box(const box_t<dim>& box)
        {
            xt::xtensor_fixed<value_t<dim>, xt::xshape<dim>> lo;
            xt::xtensor_fixed<value_t<dim>, xt::xshape<dim>> hi;
            for (std::size_t d = 0; d < dim; ++d)
            {
                lo[d] = box[d].first;
                hi[d] = box[d].second;
            }
            return lca_t<dim>{
                0,
                Box<value_t<dim>, dim>{lo, hi}
            };
        }

        /// One run of a decomposition, flattened for comparison.
        template <std::size_t dim>
        struct ShiftRun
        {
            value_t<dim> start;
            value_t<dim> end;
            std::array<int, dim> shift;
            bool fits;

            bool operator==(const ShiftRun& o) const
            {
                return start == o.start && end == o.end && shift == o.shift && fits == o.fits;
            }
        };

        template <std::size_t dim>
        std::ostream& operator<<(std::ostream& os, const ShiftRun<dim>& r)
        {
            os << "[" << r.start << "," << r.end << ") shift {";
            for (std::size_t d = 0; d < dim; ++d)
            {
                os << (d == 0 ? "" : ",") << r.shift[d];
            }
            return os << "}" << (r.fits ? "" : " !fits");
        }

        /// The runs the row scan gives.
        template <std::size_t radius, std::size_t dim>
        std::vector<ShiftRun<dim>>
        scanned_runs(const lca_t<dim>& domain, const period_t<dim>& period, const interval_t<dim>& i, const index_t<dim>& index)
        {
            std::vector<ShiftRun<dim>> out;
            for_each_prediction_shift_run<radius>(domain,
                                                  period,
                                                  i,
                                                  index,
                                                  [&](const auto& run, const auto& shift)
                                                  {
                                                      out.push_back({run.start, run.end, shift.shift, shift.fits});
                                                  });
            return out;
        }

        /// The runs the box arithmetic gives, through the view the consumers use.
        template <std::size_t radius, std::size_t dim>
        std::vector<ShiftRun<dim>>
        box_runs(const lca_t<dim>& domain, const period_t<dim>& period, const box_t<dim>& box, const interval_t<dim>& i, const index_t<dim>& index)
        {
            const PredictionDomain<dim, interval_t<dim>> view{domain, period, true, box};

            std::vector<ShiftRun<dim>> out;
            for_each_prediction_shift_run<radius>(view,
                                                  i,
                                                  index,
                                                  [&](const auto& run, const auto& shift)
                                                  {
                                                      out.push_back({run.start, run.end, shift.shift, shift.fits});
                                                  });
            return out;
        }

        /**
         * Both paths on every row within reach of the box and beyond it, on an interval
         * overhanging the box at both ends, so that the cells outside the domain, the cells
         * within reach of an end and the bulk are all visited. The runs must be identical -
         * not only the shifts cell for cell, but where the runs break, since a consumer
         * launches one kernel per run.
         */
        template <std::size_t radius, std::size_t dim>
        void expect_both_paths_agree(const box_t<dim>& box, const period_t<dim>& period)
        {
            const auto domain   = make_box<dim>(box);
            constexpr int reach = 2 * static_cast<int>(radius);

            const interval_t<dim> i{box[0].first - reach - 1, box[0].second + reach + 1};

            const auto check = [&](const index_t<dim>& index)
            {
                const auto scanned = scanned_runs<radius, dim>(domain, period, i, index);
                const auto boxed   = box_runs<radius, dim>(domain, period, box, i, index);
                EXPECT_EQ(boxed, scanned) << "radius " << radius << " at transverse index " << index;
            };

            if constexpr (dim == 1)
            {
                check(index_t<dim>{});
            }
            else if constexpr (dim == 2)
            {
                for (auto j = box[1].first - reach - 1; j < box[1].second + reach + 1; ++j)
                {
                    check(index_t<dim>{j});
                }
            }
            else
            {
                for (auto k = box[2].first - reach - 1; k < box[2].second + reach + 1; ++k)
                {
                    for (auto j = box[1].first - reach - 1; j < box[1].second + reach + 1; ++j)
                    {
                        check(index_t<dim>{j, k});
                    }
                }
            }
        }
    }

    TEST(prediction_domain, the_box_path_agrees_with_the_row_scan_in_1d)
    {
        expect_both_paths_agree<1, 1>({{{-3, 9}}}, {0});
        expect_both_paths_agree<2, 1>({{{-3, 9}}}, {0});
        expect_both_paths_agree<1, 1>({{{-3, 9}}}, {12});
        expect_both_paths_agree<2, 1>({{{-3, 9}}}, {12});
    }

    TEST(prediction_domain, the_box_path_agrees_with_the_row_scan_in_2d)
    {
        const box_t<2> box{
            {{0, 12}, {-2, 8}}
        };
        expect_both_paths_agree<1, 2>(box, {0, 0});
        expect_both_paths_agree<2, 2>(box, {0, 0});
        expect_both_paths_agree<1, 2>(box, {12, 0});
        expect_both_paths_agree<1, 2>(box, {0, 10});
        expect_both_paths_agree<2, 2>(box, {12, 10});
    }

    TEST(prediction_domain, the_box_path_agrees_with_the_row_scan_in_3d)
    {
        const box_t<3> box{
            {{0, 7}, {0, 6}, {-1, 5}}
        };
        expect_both_paths_agree<1, 3>(box, {0, 0, 0});
        expect_both_paths_agree<1, 3>(box, {0, 6, 0});
        expect_both_paths_agree<1, 3>(box, {7, 6, 6});
    }

    TEST(prediction_domain, a_box_narrower_than_the_stencil_agrees_too)
    {
        // Nothing fits in a direction two cells wide at radius 1 - and a periodic direction
        // that narrow is where the box path hands over to the row scan.
        expect_both_paths_agree<1, 2>(
            {
                {{0, 2}, {0, 9}}
        },
            {0, 0});
        expect_both_paths_agree<1, 2>(
            {
                {{0, 9}, {0, 2}}
        },
            {0, 0});
        expect_both_paths_agree<1, 2>(
            {
                {{0, 2}, {0, 9}}
        },
            {2, 0});
        expect_both_paths_agree<1, 2>(
            {
                {{0, 9}, {0, 2}}
        },
            {0, 2});
    }

    TEST(prediction_domain, a_box_mesh_caches_its_extent_at_every_level)
    {
        constexpr std::size_t dim = 2;
        auto cfg                  = mesh_config<dim>().min_level(2).max_level(5);
        auto mesh                 = mra::make_mesh(
            Box<double, dim>{
                {0., 0.},
                {1., 1.}
        },
            cfg);

        for (std::size_t level = 0; level <= 5; ++level)
        {
            EXPECT_TRUE(mesh.domain_is_box(level)) << "level " << level;
            const auto& box = mesh.domain_bbox(level);
            for (std::size_t d = 0; d < dim; ++d)
            {
                EXPECT_EQ(box[d].first, 0) << "level " << level;
                EXPECT_EQ(box[d].second, 1 << level) << "level " << level;
            }
        }
        EXPECT_EQ(mesh.domain_bbox(), mesh.domain_bbox(mesh.max_level()));

        const auto view = prediction_domain(mesh, 3);
        EXPECT_TRUE(view.is_box);
        EXPECT_EQ(view.box, mesh.domain_bbox(3));
        EXPECT_EQ(view.period, (std::array<int, dim>{0, 0}));
        EXPECT_EQ(&view.cells, &mesh.domain(3));
    }

    TEST(prediction_domain, a_periodic_box_mesh_reports_its_wrap)
    {
        constexpr std::size_t dim = 2;
        auto cfg                  = mesh_config<dim>().min_level(2).max_level(5).periodic({true, false});
        auto mesh                 = mra::make_mesh(
            Box<double, dim>{
                {0., 0.},
                {1., 1.}
        },
            cfg);

        const auto view = prediction_domain(mesh, 3);
        EXPECT_TRUE(view.is_box);
        EXPECT_EQ(view.period, (std::array<int, dim>{8, 0}));
    }

    TEST(prediction_domain, a_holed_mesh_is_not_a_box)
    {
        // A 16x16 domain at level 4 minus the four cells [8,10)^2: not a box at level 4, nor
        // at level 3 where the hole is exactly one cell, but a box at level 2 and below, where
        // the hole rounds away.
        constexpr std::size_t dim = 2;
        using cl_t                = typename MRMesh<mesh_config<dim>>::cl_type;

        const std::size_t level = 4;
        auto cfg                = mesh_config<dim>().min_level(2).max_level(level);

        cl_t cl;
        for (int j = 0; j < 16; ++j)
        {
            if (j == 8 || j == 9)
            {
                cl[level][{j}].add_interval({0, 8});
                cl[level][{j}].add_interval({10, 16});
            }
            else
            {
                cl[level][{j}].add_interval({0, 16});
            }
        }
        auto mesh = mra::make_mesh(cl, cfg);

        EXPECT_FALSE(mesh.domain_is_box(4));
        EXPECT_EQ(mesh.domain_bbox(4)[0], std::make_pair(0, 16));
        EXPECT_EQ(mesh.domain_bbox(4)[1], std::make_pair(0, 16));
        EXPECT_FALSE(mesh.domain_is_box(3));
        EXPECT_TRUE(mesh.domain_is_box(2));
        EXPECT_TRUE(mesh.domain_is_box(0));

        // The consumers' view then takes the row scan, which sees the hole.
        const auto view   = prediction_domain(mesh, level);
        const auto shifts = prediction_shifts_at<1>(view, xt::xtensor_fixed<int, xt::xshape<dim>>{8, 7});
        EXPECT_TRUE(shifts.fits);
        EXPECT_EQ(shifts.shift[1], -1);
    }

    // The stencil shifts inward only on a mesh that guarantees the cells it then reads are held
    // and filled - MRMesh, through the inward reach of its sub-mesh update. Any other mesh keeps
    // the centred stencil at its boundary cells and reads its outer ghosts, as it always did:
    // an AMR mesh with a one-cell ghost layer holds the cell two in from the boundary under a
    // finer region without ever writing it.
    TEST(prediction_domain, only_a_mesh_with_the_inward_reach_shifts_its_stencil)
    {
        constexpr std::size_t dim = 2;
        const Box<double, dim> box{
            {0., 0.},
            {1., 1.}
        };
        const std::size_t level = 4;
        // The cell at the low x boundary, mid-height: the centred stencil of radius 1 reads x = -1.
        const xt::xtensor_fixed<int, xt::xshape<dim>> boundary_cell{0, 8};

        auto mr_mesh = mra::make_mesh(box, mesh_config<dim>().min_level(2).max_level(5));
        static_assert(holds_prediction_inward_reach<decltype(mr_mesh)>::value);
        const auto mr_domain = prediction_domain(mr_mesh, level);
        EXPECT_TRUE(mr_domain.clamp);
        const auto mr_shift = prediction_shifts_at<1>(mr_domain, boundary_cell);
        EXPECT_TRUE(mr_shift.fits);
        EXPECT_EQ(mr_shift.shift[0], 1);
        EXPECT_EQ(mr_shift.shift[1], 0);

        auto amr_mesh = amr::make_mesh(box, mesh_config<dim>().min_level(2).max_level(5));
        static_assert(!holds_prediction_inward_reach<decltype(amr_mesh)>::value);
        const auto amr_domain = prediction_domain(amr_mesh, level);
        EXPECT_FALSE(amr_domain.clamp);
        const auto amr_shift = prediction_shifts_at<1>(amr_domain, boundary_cell);
        EXPECT_TRUE(amr_shift.fits);
        EXPECT_EQ(amr_shift.shift[0], 0);
        EXPECT_EQ(amr_shift.shift[1], 0);

        // And the run decomposition is the single centred run over the whole interval.
        std::size_t runs = 0;
        for_each_prediction_shift_run<1>(amr_domain,
                                         typename decltype(amr_mesh)::interval_t{0, 16},
                                         xt::xtensor_fixed<int, xt::xshape<dim - 1>>{8},
                                         [&](const auto& run, const auto& shift)
                                         {
                                             ++runs;
                                             EXPECT_EQ(run.start, 0);
                                             EXPECT_EQ(run.end, 16);
                                             EXPECT_EQ(shift.shift[0], 0);
                                             EXPECT_TRUE(shift.fits);
                                         });
        EXPECT_EQ(runs, 1u);
    }
}
