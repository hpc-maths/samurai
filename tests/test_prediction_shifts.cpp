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
//   3. The shift is chosen for the whole stencil BOX, not one direction at a time, so a
//      re-entrant corner - which no single direction can see - is clamped, and a shape no
//      shift can fit a box into reports that it does not fit.
//   4. A holed domain is decomposed exactly, cell for cell, including an interval that
//      passes over the edge of a hole. Classifying such an interval by its worst cell
//      would move interior values, which is why the query returns runs at all.
//   5. Classifying against the cells one rank holds gives a DIFFERENT answer from
//      classifying against the whole domain. This is the executable form of the
//      partition-independence invariant: it says what going wrong would look like.
//   6. The runs tile the queried interval exactly, in order, with none empty, and no two
//      neighbouring runs carry the same shift - they are maximal, so a consumer launches
//      one kernel per genuine change of shift.
//   7. The two seams the query is built on hold by themselves: the shift table and the
//      row array index rows the same way (checked at compile time), and one row's cover
//      reports exactly where it stops holding. That last property cannot be read off the
//      public decomposition - a cover cut too early only fragments the sweep before the
//      pieces are merged back together - yet the one-query-per-interval cost rests on it.

#include <algorithm>
#include <cstdlib>
#include <map>
#include <sstream>
#include <vector>

#include <gtest/gtest.h>

#include <samurai/box.hpp>
#include <samurai/level_cell_array.hpp>
#include <samurai/numeric/prediction_coefficients.hpp>
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
        using coord_t = xt::xtensor_fixed<value_t<dim>, xt::xshape<dim>>;

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
                                                  [&](const auto& run, const auto& shift)
                                                  {
                                                      out.push_back({run.start, run.end, shift.shift, shift.fits});
                                                  });

            // Property 6, checked everywhere rather than in a test of its own: the runs tile
            // the queried interval exactly, so a consumer looping over them visits every
            // cell once, and they are maximal, so it never launches two kernels where one
            // would do.
            EXPECT_FALSE(out.empty());
            EXPECT_EQ(out.front().start, i.start);
            EXPECT_EQ(out.back().end, i.end);
            for (std::size_t k = 0; k < out.size(); ++k)
            {
                EXPECT_LT(out[k].start, out[k].end) << "empty run " << out[k];
                if (k > 0)
                {
                    EXPECT_EQ(out[k].start, out[k - 1].end) << "gap or overlap before " << out[k];
                    EXPECT_FALSE(out[k].shift == out[k - 1].shift && out[k].fits == out[k - 1].fits)
                        << "run " << out[k] << " should have been merged into " << out[k - 1];
                }
            }
            return out;
        }

        /// The shifts a radius-@a radius stencil can take, in the order the rule prefers.
        template <std::size_t radius, std::size_t dim>
        std::vector<std::array<int, dim>> shifts_by_preference()
        {
            constexpr int r = static_cast<int>(radius);

            std::vector<std::array<int, dim>> out;
            std::size_t count = 1;
            for (std::size_t d = 0; d < dim; ++d)
            {
                count *= static_cast<std::size_t>(2 * r + 1);
            }
            for (std::size_t n = 0; n < count; ++n)
            {
                std::array<int, dim> shift{};
                auto rest = n;
                for (std::size_t d = 0; d < dim; ++d)
                {
                    shift[d] = static_cast<int>(rest % static_cast<std::size_t>(2 * r + 1)) - r;
                    rest /= static_cast<std::size_t>(2 * r + 1);
                }
                out.push_back(shift);
            }

            // Least shifted overall; then shifting x least, then y, and so on; then
            // negative before positive. Spelled out here rather than shared with the query,
            // so that the two say the same thing independently.
            std::sort(out.begin(),
                      out.end(),
                      [](const auto& a, const auto& b)
                      {
                          int total_a = 0;
                          int total_b = 0;
                          for (std::size_t d = 0; d < dim; ++d)
                          {
                              total_a += std::abs(a[d]);
                              total_b += std::abs(b[d]);
                          }
                          if (total_a != total_b)
                          {
                              return total_a < total_b;
                          }
                          for (std::size_t d = 0; d < dim; ++d)
                          {
                              if (std::abs(a[d]) != std::abs(b[d]))
                              {
                                  return std::abs(a[d]) < std::abs(b[d]);
                              }
                          }
                          return a < b;
                      });
            return out;
        }

        /**
         * The same answer, worked out the slow and obvious way: try the shifts in the order
         * the rule prefers them and keep the first one whose whole stencil box the domain
         * holds, asking about one cell at a time. This is the definition the fast query has
         * to agree with - it knows nothing about rows, cursors, runs or shift tables, so a
         * disagreement is a bug in the machinery rather than in both at once.
         */
        template <std::size_t radius, std::size_t dim>
        PredictionStencilShift<dim> reference_shift(const lca_t<dim>& domain, const coord_t<dim>& coord, const period_t<dim>& period = {})
        {
            constexpr int r = static_cast<int>(radius);

            std::size_t wraps = 1;
            for (std::size_t d = 0; d < dim; ++d)
            {
                wraps *= 3;
            }

            // Does the domain hold this cell, counting the cells a periodic exchange fills
            // it from - the same cell one wrap away, in any combination of directions?
            const auto holds = [&](const coord_t<dim>& c)
            {
                for (std::size_t n = 0; n < wraps; ++n)
                {
                    auto probe  = c;
                    auto rest   = n;
                    bool usable = true;
                    for (std::size_t d = 0; d < dim; ++d)
                    {
                        const auto choice = rest % 3;
                        rest /= 3;
                        if (choice == 0)
                        {
                            continue;
                        }
                        if (period[d] == 0)
                        {
                            usable = false;
                            break;
                        }
                        probe[d] += (choice == 1 ? -1 : 1) * period[d];
                    }
                    if (usable && find(domain, probe) >= 0)
                    {
                        return true;
                    }
                }
                return false;
            };

            std::size_t box = 1;
            for (std::size_t d = 0; d < dim; ++d)
            {
                box *= static_cast<std::size_t>(2 * r + 1);
            }

            for (const auto& shift : shifts_by_preference<radius, dim>())
            {
                bool admissible = true;
                for (std::size_t n = 0; n < box && admissible; ++n)
                {
                    auto probe = coord;
                    auto rest  = n;
                    for (std::size_t d = 0; d < dim; ++d)
                    {
                        const auto offset = static_cast<int>(rest % static_cast<std::size_t>(2 * r + 1)) - r;
                        rest /= static_cast<std::size_t>(2 * r + 1);
                        probe[d] += static_cast<value_t<dim>>(shift[d] + offset);
                    }
                    admissible = holds(probe);
                }
                if (admissible)
                {
                    return {shift, true};
                }
            }
            return {{}, false};
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

        /// The query against the slow answer, at every cell of a slab of the domain.
        template <std::size_t radius, std::size_t dim>
        void expect_agreement(const lca_t<dim>& domain, const interval_t<dim>& i, const index_t<dim>& index, const period_t<dim>& period = {})
        {
            for (const auto& run : runs_of<radius>(domain, i, index, period))
            {
                for (auto x = run.start; x < run.end; ++x)
                {
                    auto coord = coord_t<dim>{};
                    coord[0]   = x;
                    for (std::size_t d = 0; d + 1 < dim; ++d)
                    {
                        coord[d + 1] = index[d];
                    }

                    std::ostringstream where;
                    where << "at (";
                    for (std::size_t d = 0; d < dim; ++d)
                    {
                        where << (d == 0 ? "" : ",") << coord[d];
                    }
                    where << ") in run " << run;

                    const auto expected = reference_shift<radius, dim>(domain, coord, period);
                    EXPECT_EQ(run.shift, expected.shift) << where.str();
                    EXPECT_EQ(run.fits, expected.fits) << where.str();
                }
            }
        }

        /**
         * The row array is indexed through TransverseRows on both sides - the query when
         * it builds the rows, the shift table when it names them - so the one contract is
         * that index_of inverts offset. Both are constexpr: should one side's enumeration
         * order ever change alone, this stops compiling instead of misreading rows.
         */
        template <std::size_t radius, std::size_t dim>
        constexpr bool index_of_inverts_offset()
        {
            using rows_t = detail::TransverseRows<radius, dim>;
            for (std::size_t k = 0; k < rows_t::count; ++k)
            {
                if (rows_t::index_of(rows_t::offset(k)) != k)
                {
                    return false;
                }
            }
            return true;
        }

        static_assert(index_of_inverts_offset<1, 1>() && index_of_inverts_offset<1, 2>() && index_of_inverts_offset<1, 3>());
        static_assert(index_of_inverts_offset<2, 1>() && index_of_inverts_offset<2, 2>() && index_of_inverts_offset<2, 3>());

        /// One expected answer of DomainRow::around, for asserting a row point by point.
        struct CoverPoint
        {
            value_t<1> x;
            bool holds;
            int low;
            int high;
            value_t<1> until;
        };

        void expect_cover(detail::DomainRow<interval_t<1>>& row, const std::vector<CoverPoint>& expected)
        {
            for (const auto& e : expected)
            {
                const auto cover = row.around(e.x);
                EXPECT_EQ(cover.holds, e.holds) << "at x = " << e.x;
                EXPECT_EQ(cover.low, e.low) << "at x = " << e.x;
                EXPECT_EQ(cover.high, e.high) << "at x = " << e.x;
                EXPECT_EQ(cover.until, e.until) << "at x = " << e.x;
            }
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

    TEST(prediction_shifts, in_one_dimension_the_rule_is_the_shared_1d_classifier)
    {
        constexpr std::size_t dim = 1;
        const lca_t<dim> domain{
            4,
            box_t<dim>{{0}, {16}}
        };

        // With one direction there is no box and no mixed term, so the geometric rule has to
        // reduce to prediction_shift() applied to the two availability counts - the same
        // classifier the coefficients are keyed on. Anything else would mean two rules.
        for (value_t<dim> x = 0; x < 16; ++x)
        {
            const auto avail_low  = std::min(static_cast<int>(x), 4);
            const auto avail_high = std::min(static_cast<int>(15 - x), 4);
            const auto expected   = prediction_shift(2, avail_low, avail_high);

            const auto got = prediction_shifts_at<2, dim>(domain, {}, {x});
            EXPECT_EQ(got.shift[0], expected.shift) << "at x = " << x;
            EXPECT_EQ(got.fits, expected.fits) << "at x = " << x;
        }
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
        // The run reaches right up to the shifted cells: the decomposition breaks the
        // interval where the geometry changes, but neighbouring runs carrying the same
        // shift are merged, so what comes out is one run per change of shift.
        const std::vector<ShiftRun<dim>> expected = {
            {0, 1, {1},  true},
            {1, 7, {0},  true},
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

        // On a box the box rule and a per-direction clamp agree at every cell, corners
        // included: a convex corner is "clamp in x and clamp in y", and no other shift fits
        // the stencil box in.
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

        // The row just above the hole. Over the hole's span the stencil must shift up, away
        // from the cells the hole removed; away from it, it must not - and that is one
        // interval, so the query has to split it. The shifted run reaches one cell past the
        // hole at each end: those cells have the hole's corner in their stencil box.
        const std::vector<ShiftRun<dim>> expected = {
            {0,  1,  {1, 0},  true},
            {1,  3,  {0, 0},  true},
            {3,  9,  {0, 1},  true},
            {9,  11, {0, 0},  true},
            {11, 12, {-1, 0}, true}
        };
        EXPECT_EQ((runs_of<1, dim>(domain, {0, 12}, {8})), expected);

        // The row just below it shifts the other way, and the hole's own cells are not in
        // the domain at all.
        EXPECT_EQ((prediction_shifts_at<1, dim>(domain, {}, {6, 3})).shift, (std::array<int, 2>{0, -1}));
        EXPECT_FALSE((prediction_shifts_at<1, dim>(domain, {}, {6, 6})).fits);

        // A hole's side is a boundary like the domain's own side, and the shift it forces is
        // read off the geometry with no notion of a normal.
        EXPECT_EQ((prediction_shifts_at<1, dim>(domain, {}, {3, 4})).shift, (std::array<int, 2>{-1, 0}));
        EXPECT_EQ((prediction_shifts_at<1, dim>(domain, {}, {8, 5})).shift, (std::array<int, 2>{1, 0}));
    }

    TEST(prediction_shifts, a_re_entrant_corner_is_clamped_though_no_direction_sees_it)
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

        // The cell diagonally off the hole's corner. In each direction separately every cell
        // it reads is there, so a per-direction clamp leaves it alone - and its stencil box
        // then reads the hole's corner cell (4,4), which the domain does not hold and which
        // nothing fills. The box rule sees it and shifts.
        EXPECT_FALSE(find(domain, coord_t<dim>{4, 4}) >= 0);
        const auto diagonal = prediction_shifts_at<1, dim>(domain, {}, {3, 3});
        EXPECT_TRUE(diagonal.fits);
        EXPECT_EQ(diagonal.shift, (std::array<int, 2>{0, -1}));

        // One direction is enough to clear the corner, so it shifts by one cell and not by
        // one cell in each direction, and it takes it transversally: a shift along x would
        // move the reads of the innermost loop for nothing.
        EXPECT_EQ((prediction_shifts_at<1, dim>(domain, {}, {8, 8})).shift, (std::array<int, 2>{0, 1}));
        EXPECT_EQ((prediction_shifts_at<1, dim>(domain, {}, {3, 8})).shift, (std::array<int, 2>{0, 1}));
        EXPECT_EQ((prediction_shifts_at<1, dim>(domain, {}, {8, 3})).shift, (std::array<int, 2>{0, -1}));
    }

    TEST(prediction_shifts, a_shape_no_box_fits_into_does_not_fit)
    {
        constexpr std::size_t dim = 2;
        const std::size_t level   = 4;

        // A plus: three cells wide in x through the middle row, three cells tall in y
        // through the middle column, and nothing at the four corners.
        const lca_t<dim> across{
            level,
            box_t<dim>{{0, 1}, {3, 2}}
        };
        const lca_t<dim> down{
            level,
            box_t<dim>{{1, 0}, {2, 3}}
        };
        const lca_t<dim> domain{union_(across, down)};

        // The centre cell has one cell available either side of it in x AND in y, so every
        // direction taken on its own says a radius-1 stencil fits. The box does not: its
        // corners are the plus's missing corners, and no shift moves a 3x3 box inside a
        // plus. fits is the joint condition, which is what the mesh has to guarantee.
        EXPECT_EQ(prediction_shift(1, 1, 1).fits, true);
        EXPECT_FALSE((prediction_shifts_at<1, dim>(domain, {}, {1, 1})).fits);
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

        // The domain's own edges, on the other hand, are not boundaries any more - and the
        // wrap reaches diagonally too, so a corner cell of a doubly periodic domain reads
        // the opposite corner and stays centred.
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
            expect_agreement<1, dim>(domain, {0, 16}, {y}, both);
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
            expect_agreement<1, dim>(domain, {-2, 21}, {y});

            for (const auto& run : runs_of<1, dim>(domain, {-2, 21}, {y}))
            {
                for (auto x = run.start; x < run.end; ++x)
                {
                    ++seen[run.shift];
                    outside += (find(domain, coord_t<dim>{x, y}) < 0) ? 1 : 0;
                    cramped += (!run.fits && find(domain, coord_t<dim>{x, y}) >= 0) ? 1 : 0;
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
            expect_agreement<2, dim>(domain, {-2, 21}, {y});
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
                expect_agreement<1, dim>(domain, {-1, 9}, {y, z});
            }
        }

        // The same domain read as periodic in all three directions. Two transverse
        // directions then step off the end at once at an edge of the box, which no 2D sweep
        // and no non-periodic sweep reaches: the row the stencil wants exists only after
        // wrapping both of them.
        const period_t<dim> all = {8, 8, 8};
        for (value_t<dim> z = 0; z < 8; ++z)
        {
            for (value_t<dim> y = 0; y < 8; ++y)
            {
                expect_agreement<1, dim>(domain, {0, 8}, {y, z}, all);
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

    TEST(prediction_shifts, a_row_cover_reports_exactly_where_it_stops_holding)
    {
        // One row holding [0, 8) and [12, 20) - a hole in the middle, an end each side -
        // read at reach 2 (radius 1) with no wrap.
        const std::vector<interval_t<1>> cells = {
            {0,  8 },
            {12, 20}
        };
        detail::DomainRow<interval_t<1>> row{
            detail::RowScan<interval_t<1>>{cells.data(), cells.size()},
            0,
            2
        };

        // What until buys: the driver asks once per breakpoint, not once per cell, so
        // until must be exact. Too late would misclassify cells - the sweeps of test 4
        // would see it - but too early only fragments the sweep before the merge glues
        // the pieces back together, so that side of the contract is only visible here:
        // in the bulk of a run the cover holds all the way to run.end - reach, and it
        // moves cell by cell only within reach of an end or off the row.
        expect_cover(row,
                     {
                         {-1, false, 0, 0, 0                                     }, // before everything: covered from 0
                         {0,  true,  0, 2, 1                                     }, // within reach of the run's start
                         {1,  true,  1, 2, 2                                     },
                         {2,  true,  2, 2, 6                                     }, // the bulk: constant until end - reach
                         {5,  true,  2, 2, 6                                     },
                         {6,  true,  2, 1, 7                                     }, // within reach of its end
                         {7,  true,  2, 0, 8                                     },
                         {8,  false, 0, 0, 12                                    }, // the hole: covered again from 12
                         {12, true,  0, 2, 13                                    },
                         {14, true,  2, 2, 18                                    },
                         {19, true,  2, 0, 20                                    },
                         {20, false, 0, 0, std::numeric_limits<value_t<1>>::max()},
        });
    }

    TEST(prediction_shifts, a_wrap_tops_the_cover_up_but_does_not_move_its_breakpoints)
    {
        // The same row, read with wrap 20: the row's own ends now count through the
        // cells the periodic exchange fills them from, while the hole's edges do not -
        // a hole cell's image one wrap away is outside the domain.
        const std::vector<interval_t<1>> cells = {
            {0,  8 },
            {12, 20}
        };
        detail::DomainRow<interval_t<1>> row{
            detail::RowScan<interval_t<1>>{cells.data(), cells.size()},
            20,
            2
        };

        // low and high are topped up, until is not: it still breaks the sweep at the
        // geometric ends of the run. A wrap makes the end cells read the same as the
        // bulk - the merge is what makes them one run with it, and only when the shift
        // agrees. Cutting a run the wrap has evened out is the cheap mistake; gluing
        // one the geometry still splits would be the wrong one.
        expect_cover(row,
                     {
                         {0,  true, 2, 2, 1 }, // low reads 19, 18 through the wrap; until stays x + 1
                         {7,  true, 2, 0, 8 }, // the hole's edge: images of 8, 9 are outside, nothing tops up
                         {12, true, 0, 2, 13}, // same on the hole's other side
                         {19, true, 2, 2, 20}, // the domain's end: high reads 0, 1 through the wrap
        });
    }
}
