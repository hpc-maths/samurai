#pragma once

#include <iostream>
#include <functional>

#include "interval.hpp"
#include "level_cell_array.hpp"

namespace mure
{
    template<class MRConfig>
    void merge(const LevelCellArray<MRConfig>& a,
               const LevelCellArray<MRConfig>& b,
               std::function<bool(bool, bool)> op)
    {
        auto ia = 0;
        auto ib = 0;
        auto a_end = a.size() - 1;
        auto b_end = b.size() - 1;
        auto a_endpoints = a[ia].start;
        auto b_endpoints = b[ib].start;
        auto scan = std::min(a_endpoints, b_endpoints);
        auto sentinel = std::max(a[a_end].end, b[b_end].end) + 1;

        std::size_t a_index = 0, b_index = 0, r_index = 0;
        typename LevelCellArray<MRConfig>::interval_t result;

        auto new_endpoint = [sentinel](auto& array, auto& it, auto& index, auto& endpoints)
                            {
                                if (index == 1)
                                {
                                    it += 1;
                                    endpoints = (it == array.size())?sentinel:array[it].start;
                                    index = 0;
                                }
                                else
                                {
                                    endpoints = array[it].end;
                                    index = 1;
                                }
                            };

        std::cout << "scan " << scan << " sentinel " << sentinel << "\n";
        while (scan < sentinel)
        {
            auto in_a = !((scan < a_endpoints) ^ a_index);
            auto in_b = !((scan < b_endpoints) ^ b_index);
            auto in_res = op(in_a, in_b);

            if (in_res ^ (r_index & 1))
            {
                if (r_index == 0)
                {
                    result.start = scan;
                    r_index = 1;
                }
                else
                {
                    result.end = scan;
                    r_index = 0;
                    std::cout << result << "\n";
                }
            }

            if (scan == a_endpoints)
            {
                new_endpoint(a, ia, a_index, a_endpoints);
            }

            if (scan == b_endpoints)
            {
                new_endpoint(b, ib, b_index, b_endpoints);
            }

            scan = std::min(a_endpoints, b_endpoints);
        }
    }

    template<class MRConfig>
    void intersection(const LevelCellArray<MRConfig>& a,
                      const LevelCellArray<MRConfig>& b)
    {
        merge(a, b, [](bool in_a, bool in_b){return (in_a && in_b);});
    }

    template<class MRConfig>
    void union_(const LevelCellArray<MRConfig>& a,
               const LevelCellArray<MRConfig>& b)
    {
        merge(a, b, [](bool in_a, bool in_b){return (in_a || in_b);});
    }

    template<class MRConfig>
    void difference(const LevelCellArray<MRConfig>& a,
                    const LevelCellArray<MRConfig>& b)
    {
        merge(a, b, [](bool in_a, bool in_b){return (in_a && !in_b);});
    }
}