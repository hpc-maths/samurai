// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <string>

#if defined(WITH_STATS)
#include <algorithm>
#include <cstddef>
#include <fstream>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include <nlohmann/json.hpp>
#endif

namespace samurai
{
#if defined(WITH_STATS)
    /**
     * Records, for each call, the number of cells and intervals of every level
     * of a mesh, and writes them as JSON (read by python/read_stats.py).
     *
     * Each call to operator()(test_case, mesh) appends one entry to the array
     * `test_case` of the file. The file is written every `save_every` calls
     * (never if `save_every` is 0) and when the recorder is destroyed, as long
     * as something was recorded.
     *
     * With MPI, each rank records its own subdomain and writes the same file.
     */
    class Statistics
    {
        using json = nlohmann::json;

      public:

        explicit Statistics(std::string filename, std::size_t save_every = 10)
            : m_filename(std::move(filename))
            , m_save_every(save_every)
        {
        }

        Statistics(const Statistics&)            = delete;
        Statistics& operator=(const Statistics&) = delete;
        Statistics(Statistics&&)                 = delete;
        Statistics& operator=(Statistics&&)      = delete;

        ~Statistics()
        {
            save();
        }

        template <class Mesh>
        void operator()(const std::string& test_case, const Mesh& mesh)
        {
            using mesh_id_t = typename Mesh::mesh_id_t;
            const auto& ca  = mesh[mesh_id_t::cells];

            const std::size_t min_level = ca.min_level();
            const std::size_t max_level = ca.max_level();

            json by_level = json::object();
            for (std::size_t l = min_level; l <= max_level; ++l)
            {
                const auto& lca = ca[l];
                json result;
                result["cells"] = lca.nb_cells();

                for (std::size_t d = 0; d < Mesh::dim; ++d)
                {
                    std::vector<std::size_t> cells_per_interval;
                    cells_per_interval.reserve(lca[d].size());
                    for (const auto& interval : lca[d])
                    {
                        cells_per_interval.push_back(interval.size());
                    }

                    auto& axis                  = result[fmt::format("axis-{}", d)];
                    axis["number of intervals"] = lca[d].size();
                    axis["cells per interval"]  = min_max(cells_per_interval);
                }

                for (std::size_t d = 1; d < Mesh::dim; ++d)
                {
                    // offsets(d) gives, for each coordinate covered by an
                    // interval of axis d, where its intervals of axis d - 1
                    // start: consecutive differences count them.
                    const auto& offsets = lca.offsets(d);
                    std::vector<std::size_t> intervals_per_component;
                    for (std::size_t i = 1; i < offsets.size(); ++i)
                    {
                        intervals_per_component.push_back(offsets[i] - offsets[i - 1]);
                    }

                    result[fmt::format("axis-{}", d)]["number of intervals per component"] = min_max(intervals_per_component);
                }

                by_level[fmt::format("{:02}", l)] = std::move(result);
            }

            m_stats[test_case].push_back({
                {"min_level", min_level},
                {"max_level", max_level},
                {"by_level",  by_level }
            });

            ++m_ncalls;
            if (m_save_every != 0 && m_ncalls % m_save_every == 0)
            {
                save();
            }
        }

      private:

        /// Minimum and maximum of `values`, both null when there is none (e.g. on an empty level).
        static json min_max(const std::vector<std::size_t>& values)
        {
            if (values.empty())
            {
                return {
                    {"min", nullptr},
                    {"max", nullptr}
                };
            }
            const auto [min_it, max_it] = std::minmax_element(values.cbegin(), values.cend());
            return {
                {"min", *min_it},
                {"max", *max_it}
            };
        }

        void save() const
        {
            if (m_stats.empty())
            {
                return;
            }
            std::ofstream file(m_filename);
            // A test case name that is not valid UTF-8 must not throw from the destructor.
            file << m_stats.dump(4, ' ', false, json::error_handler_t::replace) << '\n';
        }

        std::string m_filename;
        std::size_t m_save_every;
        std::size_t m_ncalls = 0;
        json m_stats         = json::object();
    };

    /// The recorder samurai::statistics(test_case, mesh) writes to stats.json in the working directory.
    inline Statistics statistics("stats.json");
#else
    /// Does nothing: configure samurai with -DWITH_STATS=ON to record mesh statistics.
    template <class Mesh>
    void statistics(const std::string& /* test_case */, const Mesh& /* mesh */)
    {
    }
#endif
} // namespace samurai
