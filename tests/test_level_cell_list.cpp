#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <xtensor/containers/xarray.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/level_cell_array.hpp>
#include <samurai/level_cell_list.hpp>

namespace samurai
{
    TEST(level_cell_list, add_interval)
    {
        constexpr size_t dim = 2;
        LevelCellList<dim> lcl;
        lcl[{0}].add_interval({-3, 3});
    }

    namespace
    {
        // Deterministic 64-bit linear congruential generator: the test only
        // needs arbitrary, reproducible values.
        class Lcg
        {
          public:

            /// Integer in [lo, hi]
            int next(int lo, int hi)
            {
                m_state         = m_state * 6364136223846793005ULL + 1442695040888963407ULL;
                const auto span = static_cast<std::uint64_t>(hi - lo + 1);
                return lo + static_cast<int>((m_state >> 33) % span);
            }

          private:

            std::uint64_t m_state = 42;
        };

        // Arbitrary intervals (some empty, overlapping or touching) added to rows
        // visited in arbitrary order: the LevelCellArray built from the list must
        // hold, row by row in increasing order, the union of the intervals.
        template <std::size_t dim>
        void check_random_rows()
        {
            using lcl_t    = LevelCellList<dim>;
            using lca_t    = LevelCellArray<dim>;
            using row_t    = std::array<int, dim - 1>;
            using index_yz = typename lcl_t::index_yz_t;

            Lcg gen;

            lcl_t lcl(5);
            std::map<row_t, std::set<int>> expected; // ordered like the cell array: last coordinate first
            for (int k = 0; k < 3000; ++k)
            {
                row_t row;
                index_yz index;
                for (std::size_t d = 0; d < dim - 1; ++d)
                {
                    index[d]         = gen.next(-6, 6);
                    row[dim - 2 - d] = index[d];
                }
                const int s = gen.next(-40, 40);
                const int e = s + gen.next(-1, 6);
                lcl[index].add_interval({s, e});
                auto& points = expected[row];
                for (int x = s; x < e; ++x)
                {
                    points.insert(x);
                }
            }

            std::vector<std::tuple<row_t, int, int>> want;
            for (const auto& [row, points] : expected)
            {
                for (auto it = points.begin(); it != points.end();)
                {
                    const int s = *it;
                    int e       = s + 1;
                    for (++it; it != points.end() && *it == e; ++it)
                    {
                        ++e;
                    }
                    want.emplace_back(row, s, e);
                }
            }

            std::vector<std::tuple<row_t, int, int>> got;
            lca_t lca(lcl);
            for_each_interval(lca,
                              [&](auto, const auto& i, const auto& index)
                              {
                                  row_t row;
                                  for (std::size_t d = 0; d < dim - 1; ++d)
                                  {
                                      row[dim - 2 - d] = index[d];
                                  }
                                  got.emplace_back(row, i.start, i.end);
                              });
            EXPECT_EQ(got, want);
        }
    }

    TEST(level_cell_list, rows_in_any_order_2d)
    {
        check_random_rows<2>();
    }

    TEST(level_cell_list, rows_in_any_order_3d)
    {
        check_random_rows<3>();
    }
}
