// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <array>
#include <cstddef>

#include <gtest/gtest.h>

#include <samurai/algorithm/update.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/schemes/lbm.hpp>

namespace samurai
{
    namespace
    {
        constexpr std::size_t dim   = 2;
        constexpr std::size_t q9    = 9;
        constexpr std::size_t level = 4;

        // D2Q9, same order as demos/lbm/new_D2Q9_von_karman.cpp
        auto d2q9_velocities()
        {
            return std::array<std::array<int, dim>, q9>{
                {{0, 0}, {1, 0}, {0, 1}, {-1, 0}, {0, -1}, {1, 1}, {-1, 1}, {-1, -1}, {1, -1}}
            };
        }

        // D2Q4, axial only: no diagonal velocity, so no corner is claimed
        auto d2q4_velocities()
        {
            return std::array<std::array<int, dim>, 4>{
                {{1, 0}, {0, 1}, {-1, 0}, {0, -1}}
            };
        }

        template <class Vel>
        auto opposite_of(const Vel& velocities)
        {
            std::array<std::size_t, std::tuple_size_v<Vel>> opposite{};
            for (std::size_t a = 0; a < opposite.size(); ++a)
            {
                opposite[a] = a;
                for (std::size_t b = 0; b < opposite.size(); ++b)
                {
                    if (velocities[b][0] == -velocities[a][0] && velocities[b][1] == -velocities[a][1])
                    {
                        opposite[a] = b;
                    }
                }
            }
            return opposite;
        }

        auto make_test_mesh()
        {
            const Box<double, dim> box({0., 0.}, {1., 1.});
            auto config = mesh_config<dim>().min_level(level).max_level(level).periodic(false);
            return mra::make_mesh(box, config);
        }

        // Interior filled with f(a) = a, uniform in space: the reflection then gives
        // ghost(a) = opposite[a] and a plain copy gives ghost(a) = a, whatever the
        // extrapolation order, so the two are trivially distinguishable.
        template <class Field>
        void fill_by_component(Field& f)
        {
            f.fill(0.);
            for_each_cell(f.mesh(),
                          [&](const auto& cell)
                          {
                              for (std::size_t a = 0; a < Field::n_comp; ++a)
                              {
                                  f[cell](a) = static_cast<double>(a);
                              }
                          });
        }
    }

    // A D2Q9 wall streams across the domain corners, so the corner ghost must carry the
    // reflection. Before the fix it held a polynomial extrapolation, i.e. a plain copy of the
    // diagonal inner cell, and the stream consumed it.
    TEST(lbm_corner_bc, d2q9_corner_ghost_holds_the_reflection)
    {
        auto mesh           = make_test_mesh();
        auto velocities     = d2q9_velocities();
        const auto opposite = opposite_of(velocities);

        auto f = make_vector_field<double, q9>("f", mesh);
        fill_by_component(f);
        make_bc<BounceBack>(f, velocities);
        update_ghost_mr(f);

        const int n = 1 << level;

        // Control: a face ghost must show the reflection (it always did)
        for (std::size_t a = 0; a < q9; ++a)
        {
            EXPECT_DOUBLE_EQ(f(level, {-1, 0}, n / 2)(0, a), static_cast<double>(opposite[a])) << "face ghost, component " << a;
        }

        // The four domain corners must show the reflection too
        const std::array<std::array<int, dim>, 4> corners{
            {{-1, -1}, {n, -1}, {-1, n}, {n, n}}
        };
        for (const auto& c : corners)
        {
            for (std::size_t a = 0; a < q9; ++a)
            {
                EXPECT_DOUBLE_EQ(f(level, {c[0], c[0] + 1}, c[1])(0, a), static_cast<double>(opposite[a]))
                    << "corner ghost (" << c[0] << "," << c[1] << "), component " << a;
            }
        }
    }

    // The corner value is consumed by the stream, so getting it wrong reaches the solution.
    //
    // The assertion has to be on the concrete expected value, not on "cell (0,0) equals the corner
    // ghost": the buggy corner fill is a plain COPY of the diagonal inner cell, so that comparison
    // holds whatever the ghost contains and discriminates nothing (verified - it passes with the fix
    // disabled). With the reflection in place the ghost holds opposite[a], so after a pure stream
    // cell (0,0) must hold opposite[5] = 7, where the buggy fill would deliver 5.
    TEST(lbm_corner_bc, d2q9_corner_ghost_is_read_by_the_stream)
    {
        auto mesh                       = make_test_mesh();
        auto velocities                 = d2q9_velocities();
        const auto opposite             = opposite_of(velocities);
        constexpr std::size_t diag_comp = 5; // velocity {1,1}: cell (0,0) reads ghost (-1,-1)

        auto f = make_vector_field<double, q9>("f", mesh);
        auto m = make_vector_field<double, q9>("m", mesh);
        fill_by_component(f);
        m.fill(0.);
        make_bc<BounceBack>(f, velocities);

        // M = invM = I and every relaxation rate zero, so the MRT collision is the identity and one
        // step is a pure stream. operator() updates the ghosts itself first, which is what we want:
        // the corner value under test must be the one the boundary condition produces.
        std::array<std::array<double, q9>, q9> M{};
        std::array<std::array<double, q9>, q9> invM{};
        for (std::size_t a = 0; a < q9; ++a)
        {
            M[a][a]    = 1.;
            invM[a][a] = 1.;
        }
        std::array<double, q9> s{};
        s.fill(0.);
        auto eq = [](std::array<double, q9>& meq, std::span<const double> mm)
        {
            for (std::size_t a = 0; a < q9; ++a)
            {
                meq[a] = mm[a];
            }
        };

        using field_t = decltype(f);
        auto scheme   = make_lbm_scheme<field_t>("d2q9_identity_collision", 1., velocity_scheme<dim, q9>(velocities, M, invM, s, eq));

        update_ghost_mr(f);
        // Guard against the whole test becoming vacuous: the reflection and the copy must differ
        // for this component, otherwise the assertion below cannot discriminate.
        ASSERT_NE(opposite[diag_comp], diag_comp);
        EXPECT_DOUBLE_EQ(f(level, {-1, 0}, -1)(0, diag_comp), static_cast<double>(opposite[diag_comp]));

        scheme(f, m);

        // After a pure stream, cell (0,0) must hold the corner ghost's value: the reflection
        // (opposite[5] = 7), where the buggy corner fill would deliver a copy (5).
        EXPECT_DOUBLE_EQ(f(level, {0, 1}, 0)(0, diag_comp), static_cast<double>(opposite[diag_comp]));
    }

    // A scheme with no diagonal velocity must not claim the corners: those ghosts keep the
    // polynomial extrapolation, i.e. a copy of the diagonal inner cell (ghost(a) == a here).
    TEST(lbm_corner_bc, axial_scheme_leaves_the_corner_to_the_extrapolation)
    {
        auto mesh       = make_test_mesh();
        auto velocities = d2q4_velocities();

        auto f = make_vector_field<double, 4>("f", mesh);
        fill_by_component(f);
        make_bc<BounceBack>(f, velocities);
        update_ghost_mr(f);

        for (std::size_t a = 0; a < 4; ++a)
        {
            EXPECT_DOUBLE_EQ(f(level, {-1, 0}, -1)(0, a), static_cast<double>(a)) << "corner ghost, component " << a;
        }
    }

    // A corner between two DIFFERENT boundary conditions is deliberately left alone: neither wall
    // declares both of its Cartesian components, so the extrapolation keeps it.
    TEST(lbm_corner_bc, corner_between_two_different_bcs_is_left_to_the_extrapolation)
    {
        auto mesh           = make_test_mesh();
        auto velocities     = d2q9_velocities();
        const auto opposite = opposite_of(velocities);

        auto f = make_vector_field<double, q9>("f", mesh);
        fill_by_component(f);

        const DirectionVector<dim> left{-1, 0};
        const DirectionVector<dim> right{1, 0};
        const DirectionVector<dim> top{0, 1};
        const DirectionVector<dim> bottom{0, -1};
        make_bc<BounceBack>(f, velocities)->on(left, top, bottom);
        make_bc<Neumann<1>>(f)->on(right);

        update_ghost_mr(f);

        const int n = 1 << level;

        // top-left and bottom-left: both components belong to the BounceBack wall -> reflection
        for (std::size_t a = 0; a < q9; ++a)
        {
            EXPECT_DOUBLE_EQ(f(level, {-1, 0}, -1)(0, a), static_cast<double>(opposite[a])) << "bottom-left, component " << a;
            EXPECT_DOUBLE_EQ(f(level, {-1, 0}, n)(0, a), static_cast<double>(opposite[a])) << "top-left, component " << a;
        }

        // bottom-right: 'right' carries a different condition -> extrapolation (a plain copy)
        for (std::size_t a = 0; a < q9; ++a)
        {
            EXPECT_DOUBLE_EQ(f(level, {n, n + 1}, -1)(0, a), static_cast<double>(a)) << "bottom-right, component " << a;
        }
    }
}
