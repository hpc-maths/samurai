#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#ifdef SAMURAI_WITH_OPENMP
#include <omp.h>
#endif

#include <samurai/interval.hpp>
#include <samurai/numeric/prediction_coefficients.hpp>
#include <samurai/reconstruction.hpp>

// The prediction stencils are memoised in function-local caches that the FV schemes fill from
// inside OpenMP loops. These tests fill caches that nothing else in the binary has touched yet
// from several threads at once, and check that every thread gets the stencils a serial
// computation gives.

namespace samurai
{
    namespace
    {
#ifdef SAMURAI_WITH_OPENMP
        int n_threads()
        {
            return std::max(omp_get_max_threads(), 4);
        }
#endif

        // A prediction_map in a form that compares across index types and ignores insertion order.
        template <std::size_t dim, class index_t>
        std::map<std::array<long long, dim>, double> sorted(const prediction_map<dim, index_t>& p)
        {
            std::map<std::array<long long, dim>, double> out;
            for (const auto& [k, v] : p.coeff)
            {
                std::array<long long, dim> key{};
                for (std::size_t d = 0; d < dim; ++d)
                {
                    key[d] = static_cast<long long>(k[d]);
                }
                out[key] = v;
            }
            return out;
        }

        // Only Field::dim and Field::interval_t are read by detail::get_prediction. A type of its
        // own gives the slice cache an instantiation no other test has filled.
        struct SliceField
        {
            static constexpr std::size_t dim = 1;
            using interval_t                 = Interval<int, long long>;
        };
    }

#ifdef SAMURAI_WITH_OPENMP
    namespace
    {
        // Every thread asks for every (parity, shift) entry, each starting at a different one
        // so that the threads insert distinct entries at the same time.
        template <std::size_t radius>
        int coefficient_mismatches()
        {
            static constexpr int r                 = static_cast<int>(radius);
            static constexpr std::size_t n_entries = 2 * (2 * radius + 1);

            auto parity_of = [](std::size_t e)
            {
                return e % 2;
            };
            auto shift_of = [](std::size_t e)
            {
                return static_cast<int>(e / 2) - r;
            };

            std::vector<std::array<double, 2 * radius + 1>> expected;
            for (std::size_t e = 0; e < n_entries; ++e)
            {
                expected.push_back(detail::solve_moment_system<radius>(parity_of(e), shift_of(e)));
            }

            int mismatches = 0;
#pragma omp parallel num_threads(n_threads()) reduction(+ : mismatches)
            {
                const auto first = static_cast<std::size_t>(omp_get_thread_num()) * 5;
                for (std::size_t k = 0; k < n_entries; ++k)
                {
                    const std::size_t e = (first + k) % n_entries;
                    const auto& p       = prediction_coefficients<radius>(parity_of(e), shift_of(e));
                    mismatches += (p.start != -r + shift_of(e) || p.c != expected[e]) ? 1 : 0;
                }
            }
            return mismatches;
        }
    }
#endif

    TEST(prediction_cache_threads, coefficients_filled_concurrently)
    {
#ifndef SAMURAI_WITH_OPENMP
        GTEST_SKIP() << "needs OpenMP";
#else
        // Radii 4 and 5 are used nowhere else in the test binary, so their caches start empty.
        EXPECT_EQ(coefficient_mismatches<4>(), 0);
        EXPECT_EQ(coefficient_mismatches<5>(), 0);
#endif
    }

    TEST(prediction_cache_threads, prediction_filled_concurrently)
    {
#ifndef SAMURAI_WITH_OPENMP
        GTEST_SKIP() << "needs OpenMP";
#else
        // No other test predicts in 2D with radius 3, so this cache starts empty. The reference
        // does not go through it: the 2D stencil is the tensor product of two 1D ones.
        static constexpr std::size_t radius    = 3;
        static constexpr std::size_t max_level = 4;

        using stencil_t = std::map<std::array<long long, 2>, double>;
        std::vector<stencil_t> expected;
        for (std::size_t level = 1; level <= max_level; ++level)
        {
            for (int i = 0; i < (1 << level); ++i)
            {
                for (int j = 0; j < (1 << level); ++j)
                {
                    stencil_t product;
                    for (const auto& [a, ca] : sorted(prediction<radius, int>(level, i)))
                    {
                        for (const auto& [b, cb] : sorted(prediction<radius, int>(level, j)))
                        {
                            product[{a[0], b[0]}] = ca * cb;
                        }
                    }
                    expected.push_back(product);
                }
            }
        }

        // Same stencil up to round-off: the product and the 2D recursion sum in different orders.
        auto same = [](const stencil_t& actual, const stencil_t& reference)
        {
            if (actual.size() != reference.size())
            {
                return false;
            }
            return std::ranges::all_of(actual,
                                       [&](const auto& entry)
                                       {
                                           auto it = reference.find(entry.first);
                                           return it != reference.end() && std::abs(it->second - entry.second) < 1e-13;
                                       });
        };

        int mismatches = 0;
#pragma omp parallel num_threads(n_threads()) reduction(+ : mismatches)
        {
            std::size_t n = 0;
            for (std::size_t level = 1; level <= max_level; ++level)
            {
                for (int i = 0; i < (1 << level); ++i)
                {
                    for (int j = 0; j < (1 << level); ++j, ++n)
                    {
                        mismatches += same(sorted(prediction<radius, int>(level, i, j)), expected[n]) ? 0 : 1;
                    }
                }
            }
        }
        EXPECT_EQ(mismatches, 0);
#endif
    }

    TEST(prediction_cache_threads, slice_prediction_filled_concurrently)
    {
#ifndef SAMURAI_WITH_OPENMP
        GTEST_SKIP() << "needs OpenMP";
#else
        static constexpr std::size_t radius    = 1;
        static constexpr std::size_t max_level = 6;
        using interval_t                       = SliceField::interval_t;

        // A slice stencil is the sum of the stencils of its children.
        auto slice_reference = [](std::size_t delta_l, int start, int end)
        {
            prediction_map<1, int> sum;
            for (int k = start; k < end; ++k)
            {
                sum += prediction<radius, int>(delta_l, k);
            }
            return sorted(sum);
        };

        std::vector<std::map<std::array<long long, 1>, double>> expected;
        for (std::size_t delta_l = 1; delta_l <= max_level; ++delta_l)
        {
            for (int start = 0; start < (1 << delta_l); ++start)
            {
                expected.push_back(slice_reference(delta_l, start, 1 << delta_l));
            }
        }

        int mismatches = 0;
#pragma omp parallel num_threads(n_threads()) reduction(+ : mismatches)
        {
            std::size_t n = 0;
            for (std::size_t delta_l = 1; delta_l <= max_level; ++delta_l)
            {
                for (int start = 0; start < (1 << delta_l); ++start, ++n)
                {
                    const std::tuple<interval_t> slice{
                        interval_t{start, 1 << delta_l}
                    };
                    const auto& map = detail::get_prediction<radius, SliceField>(0, delta_l, slice);
                    mismatches += (sorted(map) != expected[n]) ? 1 : 0;
                }
            }
        }
        EXPECT_EQ(mismatches, 0);
#endif
    }
}
