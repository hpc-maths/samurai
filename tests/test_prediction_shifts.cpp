// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Where a prediction stencil sits, asserted on the domain alone.
//
// The query answers "what shift must the stencil take here" from the domain and nothing
// else, so these tests need no mesh, no field and no adaptation - they build a level cell
// array and ask.
//
// Asserted here:
//   1. The shift is the one the geometry forces, at every distance from a boundary, for
//      radius 1 and 2, in 1D, 2D and 3D.
//   2. The bulk of an interval comes back as ONE run with every shift zero. That is the
//      form of the interior bit-identity requirement: a consumer running its current
//      kernel on that run cannot move an interior value.
//   3. A holed domain is decomposed exactly, cell for cell, including an interval that
//      passes over the edge of a hole. Classifying such an interval by its worst cell
//      would move interior values, which is why the query returns runs at all.
//   4. Classifying against the cells one rank holds gives a DIFFERENT answer from
//      classifying against the whole domain. This is the executable form of the
//      partition-independence invariant: it says what going wrong would look like.
//   5. The runs tile the queried interval exactly, in order, with none empty.

#include <map>
#include <vector>

#include <gtest/gtest.h>

#include <samurai/box.hpp>
#include <samurai/level_cell_array.hpp>
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
        using box_t = Box<value_t<dim>, dim>;

        template <std::size_t dim>
        using period_t = std::array<value_t<dim>, dim>;

        /// One run of the decomposition, flattened for comparison.
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

        template <std::size_t radius, std::size_t dim>
        std::vector<ShiftRun<dim>>
        runs_of(const lca_t<dim>& domain, const interval_t<dim>& i, const index_t<dim>& index, const period_t<dim>& period = {})
        {
            std::vector<ShiftRun<dim>> out;
            for_each_prediction_shift_run<radius>(domain,
                                                  period,
                                                  i,
                                                  index,
                                                  [&](const auto& run, const auto& shifts)
                                                  {
                                                      out.push_back({run.start, run.end, shifts.shift, shifts.fits});
                                                  });

            // Property 5, checked everywhere rather than in a test of its own: the runs
            // tile the queried interval exactly, so a consumer looping over them visits
            // every cell once.
            EXPECT_FALSE(out.empty());
            EXPECT_EQ(out.front().start, i.start);
            EXPECT_EQ(out.back().end, i.end);
            for (std::size_t k = 0; k < out.size(); ++k)
            {
                EXPECT_LT(out[k].start, out[k].end) << "empty run " << out[k];
                if (k > 0)
                {
                    EXPECT_EQ(out[k].start, out[k - 1].end) << "gap or overlap before " << out[k];
                }
            }
            return out;
        }

        /**
         * The same answer, worked out the slow and obvious way: ask the domain about one
         * cell at a time. This is the definition the fast query has to agree with - it
         * knows nothing about rows, cursors or runs, so a disagreement is a bug in the
         * machinery rather than in both at once.
         */
        template <std::size_t radius, std::size_t dim>
        PredictionShifts<dim>
        reference_shifts(const lca_t<dim>& domain, xt::xtensor_fixed<value_t<dim>, xt::xshape<dim>> coord, const period_t<dim>& period = {})
        {
            const auto holds = [&](const auto& c)
            {
                return find(domain, c) >= 0;
            };

            if (!holds(coord))
            {
                return {{}, false};
            }

            PredictionShifts<dim> shifts;
            for (std::size_t d = 0; d < dim; ++d)
            {
                std::array<int, 2> available = {0, 0};
                for (std::size_t side = 0; side < 2; ++side)
                {
                    const auto step = (side == 0) ? -1 : 1;
                    for (std::size_t k = 1; k <= 2 * radius; ++k)
                    {
                        auto neighbour = coord;
                        neighbour[d] += static_cast<value_t<dim>>(step * static_cast<int>(k));
                        if (!holds(neighbour))
                        {
                            if (period[d] == 0)
                            {
                                break;
                            }
                            // Off the end of a periodic direction the stencil reads the
                            // cell one wrap away.
                            neighbour[d] -= static_cast<value_t<dim>>(step) * period[d];
                            if (!holds(neighbour))
                            {
                                break;
                            }
                        }
                        ++available[side];
                    }
                }

                const auto shift = prediction_shift(radius, available[0], available[1]);
                shifts.shift[d]  = shift.shift;
                shifts.fits      = shifts.fits && shift.fits;
            }
            return shifts;
        }

        /// The shift of every cell of an interval, one entry per cell.
        template <std::size_t radius, std::size_t dim>
        std::vector<std::array<int, dim>>
        shift_per_cell(const lca_t<dim>& domain, const interval_t<dim>& i, const index_t<dim>& index, const period_t<dim>& period = {})
        {
            std::vector<std::array<int, dim>> out;
            for (const auto& run : runs_of<radius>(domain, i, index, period))
            {
                for (auto x = run.start; x < run.end; ++x)
                {
                    out.push_back(run.shift);
                }
            }
            return out;
        }
    }

    TEST(prediction_shifts, radius1_1d_box)
    {
        constexpr std::size_t dim = 1;
        const lca_t<dim> domain{
            3,
            box_t<dim>{{0}, {8}}
        };

        // Only the cell touching the boundary is shifted: at radius 1 the stencil is three
        // cells wide, so one cell in is already enough to centre it.
        const std::vector<std::array<int, 1>> expected = {{1}, {0}, {0}, {0}, {0}, {0}, {0}, {-1}};
        EXPECT_EQ((shift_per_cell<1, dim>(domain, {0, 8}, {})), expected);
    }

    TEST(prediction_shifts, radius2_1d_box)
    {
        constexpr std::size_t dim = 1;
        const lca_t<dim> domain{
            4,
            box_t<dim>{{0}, {16}}
        };

        // Radius 2 reads five cells, so the two cells at each end are shifted, by 2 then 1.
        const std::vector<std::array<int, 1>> expected = {{2}, {1}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {-1}, {-2}};
        EXPECT_EQ((shift_per_cell<2, dim>(domain, {0, 16}, {})), expected);
    }

    TEST(prediction_shifts, the_bulk_is_one_run_with_no_shift)
    {
        constexpr std::size_t dim = 1;
        const lca_t<dim> domain{
            3,
            box_t<dim>{{0}, {8}}
        };

        // The interior comes back as a single run, so a consumer runs its current kernel
        // over it unchanged - the mechanism by which interior values stay bit-identical.
        const std::vector<ShiftRun<dim>> expected = {
            {0, 1, {1},  true},
            {1, 2, {0},  true},
            {2, 6, {0},  true},
            {6, 7, {0},  true},
            {7, 8, {-1}, true}
        };
        EXPECT_EQ((runs_of<1, dim>(domain, {0, 8}, {})), expected);
    }

    TEST(prediction_shifts, a_domain_too_narrow_to_hold_the_stencil_does_not_fit)
    {
        constexpr std::size_t dim = 1;

        // Three cells hold a radius-1 stencil however it is shifted; two do not.
        const lca_t<dim> wide{
            3,
            box_t<dim>{{0}, {3}}
        };
        for (auto& run : runs_of<1, dim>(wide, {0, 3}, {}))
        {
            EXPECT_TRUE(run.fits) << run;
        }

        const lca_t<dim> narrow{
            3,
            box_t<dim>{{0}, {2}}
        };
        for (auto& run : runs_of<1, dim>(narrow, {0, 2}, {}))
        {
            EXPECT_FALSE(run.fits) << run;
        }
    }

    TEST(prediction_shifts, cells_the_domain_does_not_hold_do_not_fit)
    {
        constexpr std::size_t dim = 1;
        const lca_t<dim> domain{
            3,
            box_t<dim>{{0}, {8}}
        };

        // Asked about cells outside the domain, the query says so rather than inventing a
        // shift: prediction has nothing to reach into there.
        const std::vector<ShiftRun<dim>> expected = {
            {-2, 0, {0}, false},
            {0,  1, {1}, true },
            {1,  2, {0}, true }
        };
        EXPECT_EQ((runs_of<1, dim>(domain, {-2, 2}, {})), expected);
    }

    TEST(prediction_shifts, radius1_2d_box_corner_shifts_in_both_directions)
    {
        constexpr std::size_t dim = 2;
        const lca_t<dim> domain{
            3,
            box_t<dim>{{0, 0}, {8, 8}}
        };

        // A corner is "clamp in x and clamp in y", one independent shift per direction,
        // with no notion of a diagonal or of an outward normal.
        EXPECT_EQ((prediction_shifts_at<1, dim>(domain, {}, {0, 0})).shift, (std::array<int, 2>{1, 1}));
        EXPECT_EQ((prediction_shifts_at<1, dim>(domain, {}, {7, 0})).shift, (std::array<int, 2>{-1, 1}));
        EXPECT_EQ((prediction_shifts_at<1, dim>(domain, {}, {0, 7})).shift, (std::array<int, 2>{1, -1}));
        EXPECT_EQ((prediction_shifts_at<1, dim>(domain, {}, {7, 7})).shift, (std::array<int, 2>{-1, -1}));

        // An edge shifts in one direction only, the interior in none.
        EXPECT_EQ((prediction_shifts_at<1, dim>(domain, {}, {4, 0})).shift, (std::array<int, 2>{0, 1}));
        EXPECT_EQ((prediction_shifts_at<1, dim>(domain, {}, {0, 4})).shift, (std::array<int, 2>{1, 0}));
        EXPECT_EQ((prediction_shifts_at<1, dim>(domain, {}, {4, 4})).shift, (std::array<int, 2>{0, 0}));
    }

    TEST(prediction_shifts, radius1_3d_box_corner_shifts_in_three_directions)
    {
        constexpr std::size_t dim = 3;
        const lca_t<dim> domain{
            2,
            box_t<dim>{{0, 0, 0}, {4, 4, 4}}
        };

        EXPECT_EQ((prediction_shifts_at<1, dim>(domain, {}, {0, 0, 0})).shift, (std::array<int, 3>{1, 1, 1}));
        EXPECT_EQ((prediction_shifts_at<1, dim>(domain, {}, {3, 0, 3})).shift, (std::array<int, 3>{-1, 1, -1}));
        EXPECT_EQ((prediction_shifts_at<1, dim>(domain, {}, {1, 2, 1})).shift, (std::array<int, 3>{0, 0, 0}));
    }

    TEST(prediction_shifts, a_hole_is_a_boundary_like_any_other)
    {
        constexpr std::size_t dim = 2;
        const std::size_t level   = 4;

        // A box with a square hole punched out of it.
        const lca_t<dim> full{
            level,
            box_t<dim>{{0, 0}, {12, 12}}
        };
        const lca_t<dim> hole{
            level,
            box_t<dim>{{4, 4}, {8, 8}}
        };
        const lca_t<dim> domain{difference(full, hole)};

        // The row just above the hole. Over the hole's span the stencil must shift up,
        // away from the cells the hole removed; on either side of it, it must not - and
        // that is one interval, so the query has to split it.
        const std::vector<ShiftRun<dim>> expected = {
            {0,  1,  {1, 0},  true},
            {1,  2,  {0, 0},  true},
            {2,  4,  {0, 0},  true},
            {4,  8,  {0, 1},  true},
            {8,  10, {0, 0},  true},
            {10, 11, {0, 0},  true},
            {11, 12, {-1, 0}, true}
        };
        EXPECT_EQ((runs_of<1, dim>(domain, {0, 12}, {8})), expected);

        // The row just below it shifts the other way, and the hole's own cells are not in
        // the domain at all.
        EXPECT_EQ((prediction_shifts_at<1, dim>(domain, {}, {6, 3})).shift, (std::array<int, 2>{0, -1}));
        EXPECT_FALSE((prediction_shifts_at<1, dim>(domain, {}, {6, 6})).fits);

        // A hole's side is a boundary in one direction only, exactly as the domain's own
        // side is, and the clamp is read per direction with no notion of a normal.
        EXPECT_EQ((prediction_shifts_at<1, dim>(domain, {}, {3, 4})).shift, (std::array<int, 2>{-1, 0}));
        EXPECT_EQ((prediction_shifts_at<1, dim>(domain, {}, {8, 5})).shift, (std::array<int, 2>{1, 0}));

        // A cell diagonally off the hole's corner sees no boundary at all: in each
        // direction separately the cells it reads are there. A rule that looked at the
        // hole rather than at one direction at a time would shift it, and move an interior
        // value for nothing.
        EXPECT_EQ((prediction_shifts_at<1, dim>(domain, {}, {3, 3})).shift, (std::array<int, 2>{0, 0}));
        EXPECT_EQ((prediction_shifts_at<1, dim>(domain, {}, {8, 8})).shift, (std::array<int, 2>{0, 0}));
    }

    TEST(prediction_shifts, two_different_boundaries_clamp_a_cell_independently)
    {
        constexpr std::size_t dim = 2;
        const std::size_t level   = 4;

        // A hole biting into the left edge of the domain, so that a cell can be clamped in
        // x by the domain's own side and in y by the hole.
        const lca_t<dim> full{
            level,
            box_t<dim>{{0, 0}, {12, 12}}
        };
        const lca_t<dim> hole{
            level,
            box_t<dim>{{0, 4}, {4, 8}}
        };
        const lca_t<dim> domain{difference(full, hole)};

        // Nothing in the query knows the two clamps come from different boundaries.
        EXPECT_EQ((prediction_shifts_at<1, dim>(domain, {}, {0, 3})).shift, (std::array<int, 2>{1, -1}));
        EXPECT_EQ((prediction_shifts_at<1, dim>(domain, {}, {0, 8})).shift, (std::array<int, 2>{1, 1}));

        // The row the hole ends on is cut in two by it: its left part starts at the hole's
        // edge rather than at the domain's.
        EXPECT_EQ((prediction_shifts_at<1, dim>(domain, {}, {4, 5})).shift, (std::array<int, 2>{1, 0}));
    }

    TEST(prediction_shifts, classifying_against_one_rank_s_cells_gives_the_wrong_answer)
    {
        constexpr std::size_t dim = 2;
        const std::size_t level   = 3;

        const lca_t<dim> domain{
            level,
            box_t<dim>{{0, 0}, {8, 8}}
        };
        // What a rank owning the left half would see if the classifier were handed its own
        // cells instead of the domain.
        const lca_t<dim> one_rank{
            level,
            box_t<dim>{{0, 0}, {4, 8}}
        };

        // Same cell, two answers. The interior answer is the right one, and it is the one
        // the domain gives on every rank without any communication; the subdomain invents a
        // boundary at the partition, which is how results would start depending on the rank
        // count.
        EXPECT_EQ((prediction_shifts_at<1, dim>(domain, {}, {3, 4})).shift, (std::array<int, 2>{0, 0}));
        EXPECT_EQ((prediction_shifts_at<1, dim>(one_rank, {}, {3, 4})).shift, (std::array<int, 2>{-1, 0}));
    }

    TEST(prediction_shifts, a_periodic_direction_has_no_boundary_to_clamp_against)
    {
        constexpr std::size_t dim = 1;
        const lca_t<dim> domain{
            3,
            box_t<dim>{{0}, {8}}
        };

        // The cells a stencil would want at the ends are the ones the periodic exchange
        // fills it from, so nothing is clamped anywhere.
        const std::vector<std::array<int, 1>> expected = {{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}};
        EXPECT_EQ((shift_per_cell<1, dim>(domain, {0, 8}, {}, {8})), expected);
        EXPECT_EQ((shift_per_cell<2, dim>(domain, {0, 8}, {}, {8})), expected);
    }

    TEST(prediction_shifts, periodicity_is_per_direction)
    {
        constexpr std::size_t dim = 2;
        const lca_t<dim> domain{
            3,
            box_t<dim>{{0, 0}, {8, 8}}
        };
        const period_t<dim> periodic_in_y = {0, 8};

        // A corner of a domain periodic in y only is clamped in x and free in y.
        EXPECT_EQ((prediction_shifts_at<1, dim>(domain, periodic_in_y, {0, 0})).shift, (std::array<int, 2>{1, 0}));
        EXPECT_EQ((prediction_shifts_at<1, dim>(domain, periodic_in_y, {7, 7})).shift, (std::array<int, 2>{-1, 0}));
        EXPECT_EQ((prediction_shifts_at<1, dim>(domain, periodic_in_y, {4, 0})).shift, (std::array<int, 2>{0, 0}));
    }

    TEST(prediction_shifts, a_hole_clamps_even_in_a_periodic_direction)
    {
        constexpr std::size_t dim = 2;
        const std::size_t level   = 4;

        const lca_t<dim> full{
            level,
            box_t<dim>{{0, 0}, {12, 12}}
        };
        const lca_t<dim> hole{
            level,
            box_t<dim>{{4, 4}, {8, 8}}
        };
        const lca_t<dim> domain{difference(full, hole)};
        const period_t<dim> both = {12, 12};

        // Periodicity says what happens off the end of the domain; it says nothing about a
        // hole in the middle of it, and the cells a hole removed are not filled by anyone.
        EXPECT_EQ((prediction_shifts_at<1, dim>(domain, both, {6, 3})).shift, (std::array<int, 2>{0, -1}));
        EXPECT_EQ((prediction_shifts_at<1, dim>(domain, both, {3, 6})).shift, (std::array<int, 2>{-1, 0}));

        // The domain's own edges, on the other hand, are not boundaries any more.
        EXPECT_EQ((prediction_shifts_at<1, dim>(domain, both, {0, 0})).shift, (std::array<int, 2>{0, 0}));
        EXPECT_EQ((prediction_shifts_at<1, dim>(domain, both, {11, 11})).shift, (std::array<int, 2>{0, 0}));
    }

    TEST(prediction_shifts, the_run_decomposition_agrees_cell_for_cell_when_periodic)
    {
        constexpr std::size_t dim = 2;
        const std::size_t level   = 4;

        const lca_t<dim> full{
            level,
            box_t<dim>{{0, 0}, {16, 16}}
        };
        const lca_t<dim> tall_hole{
            level,
            box_t<dim>{{3, 2}, {6, 9}}
        };
        const lca_t<dim> corner_hole{
            level,
            box_t<dim>{{14, 0}, {16, 2}}
        };
        const lca_t<dim> domain{difference(difference(full, tall_hole), corner_hole)};
        const period_t<dim> both = {16, 16};

        for (value_t<dim> y = 0; y < 16; ++y)
        {
            for (const auto& run : runs_of<1, dim>(domain, {0, 16}, {y}, both))
            {
                for (auto x = run.start; x < run.end; ++x)
                {
                    const auto expected = reference_shifts<1, dim>(domain, {x, y}, both);
                    EXPECT_EQ(run.shift, expected.shift) << "at (" << x << "," << y << ") in run " << run;
                    EXPECT_EQ(run.fits, expected.fits) << "at (" << x << "," << y << ") in run " << run;
                }
            }
        }
    }

    TEST(prediction_shifts, the_run_decomposition_agrees_cell_for_cell_with_the_slow_answer)
    {
        constexpr std::size_t dim = 2;
        const std::size_t level   = 4;

        // Deliberately awkward: two holes of different shapes, one of them one cell wide
        // (so a stencil cannot fit across it), one touching the domain's edge, plus a
        // block hanging off the side so that rows have different lengths.
        const lca_t<dim> full{
            level,
            box_t<dim>{{0, 0}, {16, 16}}
        };
        const lca_t<dim> tall_hole{
            level,
            box_t<dim>{{3, 2}, {6, 9}}
        };
        const lca_t<dim> thin_hole{
            level,
            box_t<dim>{{10, 7}, {12, 8}}
        };
        const lca_t<dim> edge_hole{
            level,
            box_t<dim>{{13, 0}, {16, 3}}
        };
        const lca_t<dim> wing{
            level,
            box_t<dim>{{16, 5}, {19, 11}}
        };
        const lca_t<dim> domain{union_(difference(difference(difference(full, tall_hole), thin_hole), edge_hole), wing)};

        // What the comparison actually met, so that the test cannot quietly become an
        // agreement about nothing if the domain above is ever simplified.
        std::map<std::array<int, dim>, int> seen;
        int outside = 0;
        int cramped = 0;

        // Queried past the domain on both sides, so cells the domain does not hold are
        // compared too.
        for (value_t<dim> y = -2; y < 18; ++y)
        {
            const auto runs = runs_of<1, dim>(domain, {-2, 21}, {y});
            for (const auto& run : runs)
            {
                for (auto x = run.start; x < run.end; ++x)
                {
                    const auto expected = reference_shifts<1, dim>(domain, {x, y});
                    EXPECT_EQ(run.shift, expected.shift) << "at (" << x << "," << y << ") in run " << run;
                    EXPECT_EQ(run.fits, expected.fits) << "at (" << x << "," << y << ") in run " << run;

                    ++seen[run.shift];
                    outside += (find(domain, {x, y}) < 0) ? 1 : 0;
                    cramped += (!run.fits && find(domain, {x, y}) >= 0) ? 1 : 0;
                }
            }
        }

        // Every shift a radius-1 stencil can take, in both directions and both signs.
        for (int sx = -1; sx <= 1; ++sx)
        {
            for (int sy = -1; sy <= 1; ++sy)
            {
                const std::array<int, dim> shift = {sx, sy};
                EXPECT_GT(seen[shift], 0) << "the domain never produced the shift {" << sx << "," << sy << "}";
            }
        }
        EXPECT_GT(outside, 0) << "no cell outside the domain was compared";
        EXPECT_GT(cramped, 0) << "the one-cell-wide hole never made a stencil not fit";

        // Radius 2 reads further, so it exercises rows the radius-1 pass never consults.
        for (value_t<dim> y = -2; y < 18; ++y)
        {
            for (const auto& run : runs_of<2, dim>(domain, {-2, 21}, {y}))
            {
                for (auto x = run.start; x < run.end; ++x)
                {
                    const auto expected = reference_shifts<2, dim>(domain, {x, y});
                    EXPECT_EQ(run.shift, expected.shift) << "at (" << x << "," << y << ") in run " << run;
                    EXPECT_EQ(run.fits, expected.fits) << "at (" << x << "," << y << ") in run " << run;
                }
            }
        }
    }

    TEST(prediction_shifts, the_run_decomposition_agrees_cell_for_cell_in_3d)
    {
        constexpr std::size_t dim = 3;
        const std::size_t level   = 3;

        const lca_t<dim> full{
            level,
            box_t<dim>{{0, 0, 0}, {8, 8, 8}}
        };
        const lca_t<dim> hole{
            level,
            box_t<dim>{{2, 3, 1}, {5, 6, 4}}
        };
        const lca_t<dim> domain{difference(full, hole)};

        for (value_t<dim> z = -1; z < 9; ++z)
        {
            for (value_t<dim> y = -1; y < 9; ++y)
            {
                for (const auto& run : runs_of<1, dim>(domain, {-1, 9}, {y, z}))
                {
                    for (auto x = run.start; x < run.end; ++x)
                    {
                        const auto expected = reference_shifts<1, dim>(domain, {x, y, z});
                        EXPECT_EQ(run.shift, expected.shift) << "at (" << x << "," << y << "," << z << ")";
                        EXPECT_EQ(run.fits, expected.fits) << "at (" << x << "," << y << "," << z << ")";
                    }
                }
            }
        }
    }

    TEST(prediction_shifts, radius2_needs_two_cells_of_clearance_across_a_hole)
    {
        constexpr std::size_t dim = 2;
        const std::size_t level   = 4;

        const lca_t<dim> full{
            level,
            box_t<dim>{{0, 0}, {16, 16}}
        };
        const lca_t<dim> hole{
            level,
            box_t<dim>{{6, 6}, {10, 10}}
        };
        const lca_t<dim> domain{difference(full, hole)};

        // Radius 2 reads two cells each side, so the shift persists one cell further from
        // the hole than radius 1 would put it.
        EXPECT_EQ((prediction_shifts_at<2, dim>(domain, {}, {8, 10})).shift, (std::array<int, 2>{0, 2}));
        EXPECT_EQ((prediction_shifts_at<2, dim>(domain, {}, {8, 11})).shift, (std::array<int, 2>{0, 1}));
        EXPECT_EQ((prediction_shifts_at<2, dim>(domain, {}, {8, 12})).shift, (std::array<int, 2>{0, 0}));

        // Between the hole and the domain edge there are six cells, which is more than the
        // five a radius-2 stencil needs, so everything there still fits.
        for (auto& run : runs_of<2, dim>(domain, {0, 6}, {8}))
        {
            EXPECT_TRUE(run.fits) << run;
        }
    }
}
