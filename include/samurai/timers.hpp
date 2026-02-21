// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include <fmt/color.h>
#include <fmt/format.h>

#include "assert_log_trace.hpp"

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#else
#include <chrono>
#endif

namespace samurai
{
    // =========================================================================
    // Internal clock abstractions (MPI vs. std::chrono)
    // =========================================================================

#ifdef SAMURAI_WITH_MPI
    using clock_value_t    = double;
    using duration_value_t = double;
#else
    using clock_value_t    = std::chrono::time_point<std::chrono::high_resolution_clock>;
    using duration_value_t = std::chrono::microseconds;
#endif

    // =========================================================================
    // TimerEntry – per-context accumulator
    // =========================================================================

    /**
     * @struct TimerEntry
     * @brief Accumulates timing data and optional cell counts for one (parent, name) context.
     *
     * - `elapsed`      : total wall-clock time accumulated across all `stop()` calls.
     * - `ntimes`       : number of completed start/stop cycles.
     * - `total_cells`  : sum of cell counts passed to `stop(name, nb_cells)`.
     *                    Stays at 0 if no cell count is ever provided, which causes
     *                    the Mcells/s column to be suppressed in the output.
     */
    struct TimerEntry
    {
        clock_value_t start{};
        duration_value_t elapsed{};
        uint32_t ntimes{0};
        uint64_t total_cells{0};
    };

    // =========================================================================
    // Timers – central registry
    // =========================================================================

    /**
     * @class Timers
     * @brief Registry of named timers with automatic call-context hierarchy.
     *
     * ### Per-context tracking
     * The same timer name started from different call contexts (different parent
     * timers) is tracked as a **separate entry**. For example, if `"ghost update"`
     * is called both inside `"mesh adaptation"` and at the top level, both
     * instances appear as distinct rows in the output, each with their own
     * elapsed time and call count.
     *
     * The internal key is `(parent_context, name)` — determined at runtime by
     * the thread-local active stack.
     *
     * ### Cell throughput
     * The overload `stop(name, nb_cells)` accumulates cell counts. When at least one
     * `stop()` call provides a non-zero count, the output shows a `Mcells/s` column.
     *
     * ### Usage
     * @code
     * // Preferred: RAII guard – hierarchy built automatically
     * {
     *     samurai::ScopedTimer t("mesh adaptation");
     *     // ...
     * } // stop() called here
     *
     * // With cell count
     * {
     *     samurai::ScopedTimer t("ghost update");
     *     // ...
     *     t.set_cells(mesh.nb_cells());
     * }
     *
     * // Explicit start/stop still fully supported
     * samurai::times::timers.start("my timer");
     * samurai::times::timers.stop("my timer");
     * or
     * samurai::times::timers.stop("my timer", nb_cells);
     * @endcode
     */
    class Timers
    {
      public:

        // -----------------------------------------------------------------
        // Enable / disable
        // -----------------------------------------------------------------

        /**
         * @brief Enable timing instrumentation.
         *
         * When disabled (the default), `start()` and `stop()` are no-ops and
         * impose zero overhead. Call `enable()` before starting any timers.
         */
        void enable()
        {
            _enabled = true;
        }

        /** @brief Disable timing instrumentation and clear all accumulated data. */
        void disable()
        {
            _enabled = false;
            _times.clear();
            _active_stack().clear();
        }

        [[nodiscard]] bool is_enabled() const noexcept
        {
            return _enabled;
        }

        // -----------------------------------------------------------------
        // Start / stop
        // -----------------------------------------------------------------

        /**
         * @brief Start a named timer in the current call context.
         *
         * The current top of the thread-local stack defines the parent context.
         * Each unique `(parent, name)` pair gets its own `TimerEntry`.
         * If the same pair has been seen before, its accumulated elapsed and
         * total_cells are preserved and only the start timestamp is refreshed.
         *
         * @param tname  Human-readable timer name.
         */
        void start(const std::string& tname)
        {
            if (!_enabled)
            {
                return;
            }
            const std::string ctx_key = _make_key(tname);

            if (_times.find(ctx_key) == _times.end())
            {
                _times.emplace(ctx_key, TimerEntry{_getTime(), _zero_duration(), 0, 0});
            }
            else
            {
                _times.at(ctx_key).start = _getTime();
            }
            // Push the context key so children can find their parent
            _active_stack().push_back(ctx_key);
        }

        /**
         * @brief Stop a named timer (no cell count).
         * @param tname  Must match a previously started timer in the current context.
         */
        void stop(const std::string& tname)
        {
            stop(tname, 0ULL);
        }

        /**
         * @brief Stop a named timer and record the number of cells processed.
         *
         * @param tname     Must match a previously started timer in the current context.
         * @param nb_cells  Number of cells processed during this invocation.
         */
        void stop(const std::string& tname, uint64_t nb_cells)
        {
            if (!_enabled)
            {
                return;
            }
            // Find the last stack entry whose display name matches tname
            auto& stack = _active_stack();
            std::string ctx_key;
            for (auto it = stack.rbegin(); it != stack.rend(); ++it)
            {
                if (_display_name(*it) == tname)
                {
                    ctx_key = *it;
                    stack.erase(std::next(it).base());
                    break;
                }
            }

            SAMURAI_ASSERT(!ctx_key.empty(), "[Timers::stop] No active timer named '" + tname + "' on the stack!");

            auto& entry = _times.at(ctx_key);

#ifdef SAMURAI_WITH_MPI
            entry.elapsed += (_getTime() - entry.start);
#else
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(_getTime() - entry.start);
            entry.elapsed += duration;
#endif
            entry.ntimes++;
            entry.total_cells += nb_cells;
        }

        // -----------------------------------------------------------------
        // Query
        // -----------------------------------------------------------------

        /**
         * @brief Return the total elapsed duration for a timer name, summed across
         *        all call contexts in which it was used.
         */
        [[nodiscard]] auto getElapsedTime(const std::string& tname) const
        {
            duration_value_t sum = _zero_duration();
            for (const auto& [key, entry] : _times)
            {
                if (_display_name(key) == tname)
                {
                    sum += entry.elapsed;
                }
            }
            return sum;
        }

        // -----------------------------------------------------------------
        // Print (MPI path) — same colored tree as sequential, with min/max/ave columns
        // -----------------------------------------------------------------

#ifdef SAMURAI_WITH_MPI
        void print() const
        {
            if (_times.empty())
            {
                return;
            }

            boost::mpi::communicator world;

            // ---- Gather all elapsed values from every rank in key order ----
            // Keys are iterated in std::map order (lexicographic), which is deterministic
            // and identical on all ranks for a symmetric run.
            std::vector<std::string> all_keys;
            std::vector<double> local_elapsed;
            all_keys.reserve(_times.size());
            local_elapsed.reserve(_times.size());
            for (const auto& [key, entry] : _times)
            {
                all_keys.push_back(key);
                local_elapsed.push_back(entry.elapsed);
            }

            // all_gather: each rank sends its vector; root receives nranks × nkeys matrix
            std::vector<std::vector<double>> gathered; // filled only on rank 0
            boost::mpi::gather(world, local_elapsed, gathered, 0);

            // ---- Compute per-key stats (rank 0 only) ----
            struct KeyStats
            {
                double min_s{0}, max_s{0}, ave_s{0};
                int minrank{0}, maxrank{0};
            };

            std::map<std::string, KeyStats> stats;

            if (world.rank() == 0)
            {
                const int nranks = world.size();
                const auto nkeys = static_cast<int>(all_keys.size());
                for (int k = 0; k < nkeys; ++k)
                {
                    double mn  = std::numeric_limits<double>::max();
                    double mx  = std::numeric_limits<double>::lowest();
                    double sum = 0.0;
                    int mnr = 0, mxr = 0;
                    for (int r = 0; r < nranks; ++r)
                    {
                        const double v = gathered[static_cast<std::size_t>(r)][static_cast<std::size_t>(k)];
                        sum += v;
                        if (v < mn)
                        {
                            mn  = v;
                            mnr = r;
                        }
                        if (v > mx)
                        {
                            mx  = v;
                            mxr = r;
                        }
                    }
                    stats[all_keys[static_cast<std::size_t>(k)]] = {mn, mx, sum / nranks, mnr, mxr};
                }
            }

            if (world.rank() != 0)
            {
                return;
            }

            // ---- Everything below runs only on rank 0 ----

            // Build child map from key strings (parent = everything before last "::")
            const auto children = _build_children();

            // Find "total runtime"
            double total_runtime_s = 0.0;
            bool has_total_runtime = false;
            std::string total_runtime_key;
            for (const auto& [key, entry] : _times)
            {
                if (_display_name(key) == "total runtime")
                {
                    has_total_runtime = true;
                    total_runtime_s   = stats.at(key).ave_s; // use average as grand total
                    total_runtime_key = key;
                }
            }

            // Sum direct children of "total runtime" (by ave) for (untimed)
            auto _sum_top_level_s = [&]() -> double
            {
                double sum = 0.0;
                if (has_total_runtime)
                {
                    auto it = children.find(total_runtime_key);
                    if (it != children.end())
                    {
                        for (const auto& child_key : it->second)
                        {
                            if (stats.count(child_key))
                            {
                                sum += stats.at(child_key).ave_s;
                            }
                        }
                    }
                }
                else
                {
                    for (const auto& [key, ignored] : _times)
                    {
                        if (_is_root(key))
                        {
                            sum += stats.at(key).ave_s;
                        }
                    }
                }
                return sum;
            };

            const double total_measured_s = _sum_top_level_s();
            const double grand_total_s    = has_total_runtime ? total_runtime_s : total_measured_s;

            // ---- Column widths ----
            const int nameWidth  = _compute_name_width_with_tree(20);
            const int timeWidth  = 10; // "  6.219"
            const int rankWidth  = 5;  // "[12]"
            const int percWidth  = 8;  // "  63.5%"
            const int parWidth   = 13; // "(63.5% par)"
            const int callsWidth = 7;

            const int total_width = nameWidth + 1 + timeWidth + 1 + rankWidth + 1 // min [r]
                                  + timeWidth + 1 + rankWidth + 1                 // max [r]
                                  + timeWidth + 1                                 // ave
                                  + percWidth + 1 + parWidth + 1 + callsWidth;

            // ---- Styles ----
            const auto hdr_style  = fmt::emphasis::bold | fmt::fg(fmt::terminal_color::white);
            const auto bold_style = fmt::emphasis::bold;
            const auto dim_style  = fmt::emphasis::faint;

            // ---- Title + header ----
            fmt::print("\n");
            fmt::print(bold_style, " Timers\n");
            fmt::print(hdr_style,
                       "{:<{}} {:>{}} {:>{}} {:>{}} {:>{}} {:>{}} {:>{}} {:>{}} {:>{}}\n",
                       "Timer",
                       nameWidth,
                       "Min (s)",
                       timeWidth,
                       "[r]",
                       rankWidth,
                       "Max (s)",
                       timeWidth,
                       "[r]",
                       rankWidth,
                       "Ave (s)",
                       timeWidth,
                       "% total",
                       percWidth,
                       "% parent",
                       parWidth,
                       "Calls",
                       callsWidth);
            fmt::print(dim_style, "{}\n", std::string(static_cast<std::size_t>(total_width), '-'));

            // ---- Helpers to sort keys by average elapsed ----
            auto sort_by_ave = [&](std::vector<std::string>& keys)
            {
                std::sort(keys.begin(),
                          keys.end(),
                          [&](const std::string& a, const std::string& b)
                          {
                              const double ea = stats.count(a) ? stats.at(a).ave_s : 0.0;
                              const double eb = stats.count(b) ? stats.at(b).ave_s : 0.0;
                              return ea > eb;
                          });
            };

            // ---- Per-entry print lambda ----
            auto print_entry =
                [&](const std::string& ctx_key, const std::string& prefix, const std::string& connector, bool is_root, double parent_s)
            {
                if (!stats.count(ctx_key))
                {
                    return;
                }
                const KeyStats& s           = stats.at(ctx_key);
                const auto& entry           = _times.at(ctx_key);
                const std::string full_name = prefix + connector + _display_name(ctx_key);

                const double perc_total  = (grand_total_s > 0.0) ? s.ave_s / grand_total_s * 100.0 : 0.0;
                const double perc_parent = (parent_s > 0.0) ? s.ave_s / parent_s * 100.0 : 0.0;

                const std::string perc_col = fmt::format("{:.1f}%", perc_total);
                const std::string par_col  = is_root ? std::string{} : fmt::format("({:.1f}% par)", perc_parent);

                const fmt::text_style name_style = is_root ? bold_style : _heat_style(perc_total);
                const fmt::text_style num_style  = is_root ? bold_style : _heat_style(perc_total);

                fmt::print(name_style, "{:<{}}", full_name, nameWidth);
                fmt::print(num_style, " {:>{}.3f}", s.min_s, timeWidth);
                fmt::print(dim_style, " {:>{}}", fmt::format("[{}]", s.minrank), rankWidth);
                fmt::print(num_style, " {:>{}.3f}", s.max_s, timeWidth);
                fmt::print(dim_style, " {:>{}}", fmt::format("[{}]", s.maxrank), rankWidth);
                fmt::print(num_style, " {:>{}.3f}", s.ave_s, timeWidth);
                fmt::print(num_style, " {:>{}}", perc_col, percWidth);
                fmt::print(dim_style, " {:>{}}", par_col, parWidth);
                fmt::print(dim_style, " {:>{}}", entry.ntimes, callsWidth);
                fmt::print("\n");
            };

            // ---- Recursive tree walk ----
            std::function<void(const std::string&, const std::string&, double)> print_children;
            print_children = [&](const std::string& parent_key, const std::string& prefix, double parent_s)
            {
                auto it = children.find(parent_key);
                if (it == children.end())
                {
                    return;
                }

                std::vector<std::string> kids = it->second;
                sort_by_ave(kids);

                for (std::size_t i = 0; i < kids.size(); ++i)
                {
                    const bool is_last             = (i + 1 == kids.size());
                    const std::string connector    = is_last ? "`-- " : "+-- ";
                    const std::string child_prefix = prefix + (is_last ? "    " : "|   ");
                    print_entry(kids[i], prefix, connector, /*is_root=*/false, parent_s);
                    print_children(kids[i], child_prefix, stats.count(kids[i]) ? stats.at(kids[i]).ave_s : 0.0);
                }
            };

            // Print roots
            std::vector<std::string> roots;
            for (const auto& [key, ignored] : _times)
            {
                if (_is_root(key))
                {
                    roots.push_back(key);
                }
            }
            sort_by_ave(roots);

            auto it_total = std::find(roots.begin(), roots.end(), total_runtime_key);
            if (it_total != roots.end())
            {
                roots.erase(it_total);
                print_entry(total_runtime_key, "", "", /*is_root=*/true, grand_total_s);
                print_children(total_runtime_key, "", grand_total_s);
            }
            for (const auto& root_key : roots)
            {
                print_entry(root_key, "", "", /*is_root=*/true, grand_total_s);
                print_children(root_key, "", stats.count(root_key) ? stats.at(root_key).ave_s : 0.0);
            }

            // ---- Separator + untimed + footer ----
            const std::string sep(static_cast<std::size_t>(total_width), '-');
            if (has_total_runtime)
            {
                const double untimed_s   = total_runtime_s - total_measured_s;
                const bool overcommitted = untimed_s < 0.0;

                fmt::print(dim_style, "{}\n", sep);
                if (!overcommitted)
                {
                    const double perc = (grand_total_s > 0.0) ? untimed_s / grand_total_s * 100.0 : 0.0;
                    fmt::print(dim_style,
                               "{:<{}} {:>{}.3f} {:>{}.1f}%\n",
                               "(untimed)",
                               nameWidth,
                               untimed_s,
                               timeWidth + 1 + rankWidth + 1 + timeWidth + 1 + rankWidth,
                               perc,
                               percWidth - 1);
                }
                else
                {
                    fmt::print(dim_style,
                               "{:<{}} {:>{}}\n",
                               "(overlap / pause-resume timers)",
                               nameWidth,
                               "(children exceed parent \u2014 timer overlap detected)",
                               timeWidth + 1 + rankWidth + 1 + timeWidth + 1 + rankWidth + 1 + timeWidth + 1 + percWidth);
                }
                fmt::print(dim_style, "{}\n", sep);
                fmt::print(bold_style,
                           "{:<{}} {:>{}.3f} {:>{}.1f}%\n",
                           "total runtime (ave)",
                           nameWidth,
                           total_runtime_s,
                           timeWidth + 1 + rankWidth + 1 + timeWidth + 1 + rankWidth,
                           100.0,
                           percWidth - 1);
            }
            else
            {
                fmt::print(dim_style, "{}\n", sep);
                fmt::print(bold_style,
                           "{:<{}} {:>{}.3f} {:>{}.1f}%\n",
                           "Total (ave)",
                           nameWidth,
                           total_measured_s,
                           timeWidth + 1 + rankWidth + 1 + timeWidth + 1 + rankWidth,
                           100.0,
                           percWidth - 1);
            }

            fmt::print("\n");
        }

#else
        // -----------------------------------------------------------------
        // Print (sequential path) — colored indented tree
        // -----------------------------------------------------------------

        /**
         * @brief Print a hierarchical, sorted timing report to stdout with colors and tree connectors.
         *
         * Each unique (parent, name) call context is shown as a separate row,
         * indented according to its depth in the call tree using box-drawing connectors.
         * Rows are heat-colored (green → yellow → red) by their fraction of total runtime.
         */
        void print() const
        {
            if (_times.empty())
            {
                return;
            }

            // ---- Find "total runtime" ----
            duration_value_t total_runtime = _zero_duration();
            bool has_total_runtime         = false;
            std::string total_runtime_key;

            for (const auto& [key, entry] : _times)
            {
                if (_display_name(key) == "total runtime")
                {
                    has_total_runtime = true;
                    total_runtime     = entry.elapsed;
                    total_runtime_key = key;
                }
            }

            // Build child map from key strings (parent = everything before last "::")
            const auto children = _build_children();

            // ---- Sum direct children of "total runtime" for (untimed) ----
            auto _sum_top_level = [&]()
            {
                duration_value_t sum = _zero_duration();
                if (has_total_runtime)
                {
                    auto it = children.find(total_runtime_key);
                    if (it != children.end())
                    {
                        for (const auto& child_key : it->second)
                        {
                            if (auto cit = _times.find(child_key); cit != _times.end())
                            {
                                sum += cit->second.elapsed;
                            }
                        }
                    }
                }
                else
                {
                    for (const auto& [key, entry] : _times)
                    {
                        if (_is_root(key))
                        {
                            sum += entry.elapsed;
                        }
                    }
                }
                return sum;
            };

            const duration_value_t total_measured = _sum_top_level();
            const duration_value_t grand_total    = has_total_runtime ? total_runtime : total_measured;

            // ---- Column widths ----
            const bool show_mcells = _any_has_cells();
            // Name column: account for tree connector prefix ("├── " = 4 chars per level) + display name
            const int nameWidth    = _compute_name_width_with_tree(20);
            const int elapsedWidth = 12;
            const int percWidth    = 8;  // "63.5%"
            const int parWidth     = 10; // "63.5%" or empty
            const int callsWidth   = 10;
            const int mcellsWidth  = 10;

            // ---- Colors / styles ----
            // Header style: bold + dim grey
            const auto hdr_style                   = fmt::emphasis::bold | fmt::fg(fmt::terminal_color::white);
            const auto dim_style                   = fmt::emphasis::faint;
            const auto bold_style                  = fmt::emphasis::bold;
            [[maybe_unused]] const auto cyan_style = fmt::fg(fmt::terminal_color::cyan);
            const auto dim_sep                     = fmt::emphasis::faint;

            const int total_width = nameWidth + 1 + elapsedWidth + 1 + percWidth + 1 + parWidth + 1 + callsWidth
                                  + (show_mcells ? 1 + mcellsWidth : 0);

            // ---- Column headers ----
            if (show_mcells)
            {
                fmt::print(hdr_style,
                           "{:<{}} {:>{}} {:>{}} {:>{}} {:>{}} {:>{}}\n",
                           "Timer",
                           nameWidth,
                           "Elapsed (s)",
                           elapsedWidth,
                           "% total",
                           percWidth,
                           "% parent",
                           parWidth,
                           "Calls",
                           callsWidth,
                           "Mcells/s",
                           mcellsWidth);
            }
            else
            {
                fmt::print(hdr_style,
                           "{:<{}} {:>{}} {:>{}} {:>{}} {:>{}}\n",
                           "Timer",
                           nameWidth,
                           "Elapsed (s)",
                           elapsedWidth,
                           "% total",
                           percWidth,
                           "% parent",
                           parWidth,
                           "Calls",
                           callsWidth);
            }

            // Thin separator under headers
            fmt::print(dim_sep, "{}\n", std::string(static_cast<std::size_t>(total_width), '-'));

            // ---- Tree walk ----
            // Collect root keys (no "::" in key = no parent)
            std::vector<std::string> roots;
            for (const auto& [key, ignored] : _times)
            {
                if (_is_root(key))
                {
                    roots.push_back(key);
                }
            }
            _sort_by_elapsed(roots);

            // Print "total runtime" root first
            auto it_total = std::find(roots.begin(), roots.end(), total_runtime_key);
            if (it_total != roots.end())
            {
                roots.erase(it_total);
                _print_entry(total_runtime_key,
                             /*prefix=*/"",
                             /*connector=*/"",
                             /*is_root=*/true,
                             grand_total,
                             grand_total,
                             show_mcells,
                             nameWidth,
                             elapsedWidth,
                             percWidth,
                             parWidth,
                             callsWidth,
                             mcellsWidth);
                _print_children(total_runtime_key,
                                /*prefix=*/"",
                                grand_total,
                                children,
                                show_mcells,
                                nameWidth,
                                elapsedWidth,
                                percWidth,
                                parWidth,
                                callsWidth,
                                mcellsWidth);
            }

            // Remaining roots (rare — only if something was started without "total runtime" as parent)
            for (const auto& root_key : roots)
            {
                _print_entry(root_key,
                             "",
                             "",
                             /*is_root=*/true,
                             grand_total,
                             grand_total,
                             show_mcells,
                             nameWidth,
                             elapsedWidth,
                             percWidth,
                             parWidth,
                             callsWidth,
                             mcellsWidth);
                _print_children(root_key, "", grand_total, children, show_mcells, nameWidth, elapsedWidth, percWidth, parWidth, callsWidth, mcellsWidth);
            }

            // ---- Separator + untimed + bold total ----
            const std::string sep(static_cast<std::size_t>(total_width), '-');
            if (has_total_runtime)
            {
                const auto untimed          = total_runtime - total_measured;
                const auto untimedInSeconds = _to_seconds(untimed);
                const bool overcommitted    = untimedInSeconds < 0.0;

                fmt::print(dim_sep, "{}\n", sep);

                if (!overcommitted)
                {
                    fmt::print(dim_style,
                               "{:<{}} {:>{}.3f} {:>{}.1f}%\n",
                               "(untimed)",
                               nameWidth,
                               untimedInSeconds,
                               elapsedWidth,
                               _percent(untimed, grand_total),
                               percWidth - 1);
                }
                else
                {
                    fmt::print(dim_style,
                               "{:<{}} {:>{}}\n",
                               "(overlap / pause-resume timers)",
                               nameWidth,
                               "(children exceed parent \u2014 timer overlap detected)",
                               elapsedWidth + 1 + percWidth + 1 + parWidth + 1 + callsWidth);
                }

                fmt::print(dim_sep, "{}\n", sep);
                fmt::print(bold_style,
                           "{:<{}} {:>{}.3f} {:>{}.1f}%\n",
                           "total runtime",
                           nameWidth,
                           _to_seconds(total_runtime),
                           elapsedWidth,
                           100.0,
                           percWidth - 1);
            }
            else
            {
                fmt::print(dim_sep, "{}\n", sep);
                fmt::print(bold_style,
                           "{:<{}} {:>{}.3f} {:>{}.1f}%\n",
                           "Total",
                           nameWidth,
                           _to_seconds(total_measured),
                           elapsedWidth,
                           100.0,
                           percWidth - 1);
            }

            fmt::print("\n");
        }
#endif

      private:

        // -----------------------------------------------------------------
        // Data
        // -----------------------------------------------------------------

        /// When false, start() and stop() are no-ops. Default: false (disabled).
        bool _enabled{false};

        /// Internal key: "parent_ctx_key::display_name" (or just "display_name" for roots)
        std::map<std::string, TimerEntry> _times;

        // -----------------------------------------------------------------
        // Thread-local active stack — stores ctx_keys
        // -----------------------------------------------------------------

        static std::vector<std::string>& _active_stack()
        {
            thread_local std::vector<std::string> stack;
            return stack;
        }

        // -----------------------------------------------------------------
        // Key helpers
        // -----------------------------------------------------------------

        /**
         * @brief Build the context key for `tname` given the current stack top.
         *
         * The key encodes the full parent path so that the same display name
         * called from two different parents produces two distinct keys.
         */
        [[nodiscard]] static std::string _make_key(const std::string& tname)
        {
            auto& stack = _active_stack();
            if (stack.empty())
            {
                return tname;
            }
            return stack.back() + "::" + tname;
        }

        /**
         * @brief Extract the display name from a context key.
         *
         * The display name is everything after the last "::".
         */
        [[nodiscard]] static std::string _display_name(const std::string& ctx_key)
        {
            const auto pos = ctx_key.rfind("::");
            return (pos == std::string::npos) ? ctx_key : ctx_key.substr(pos + 2);
        }

        /**
         * @brief Return true if `ctx_key` is a root (has no parent, i.e. no "::" separator).
         */
        [[nodiscard]] static bool _is_root(const std::string& ctx_key)
        {
            return ctx_key.find("::") == std::string::npos;
        }

        /**
         * @brief Return the parent key of `ctx_key`, or an empty string for roots.
         *
         * The parent key is everything up to (but not including) the last "::".
         */
        [[nodiscard]] static std::string _parent_key(const std::string& ctx_key)
        {
            const auto pos = ctx_key.rfind("::");
            return (pos == std::string::npos) ? std::string{} : ctx_key.substr(0, pos);
        }

        /**
         * @brief Build a parent → [children] map from the keys in `_times`.
         *
         * This replaces the old `_children` data member. Called once at the start
         * of `print()` so it is an O(n) operation performed only at output time.
         */
        [[nodiscard]] std::map<std::string, std::vector<std::string>> _build_children() const
        {
            std::map<std::string, std::vector<std::string>> ch;
            for (const auto& [key, ignored] : _times)
            {
                if (!_is_root(key))
                {
                    auto& siblings = ch[_parent_key(key)];
                    if (std::find(siblings.begin(), siblings.end(), key) == siblings.end())
                    {
                        siblings.push_back(key);
                    }
                }
            }
            return ch;
        }

        // -----------------------------------------------------------------
        // Clock helpers
        // -----------------------------------------------------------------

#ifdef SAMURAI_WITH_MPI
        static double _getTime()
        {
            return MPI_Wtime();
        }

        static double _zero_duration()
        {
            return 0.0;
        }
#else
        static std::chrono::time_point<std::chrono::high_resolution_clock> _getTime()
        {
            return std::chrono::high_resolution_clock::now();
        }

        static std::chrono::microseconds _zero_duration()
        {
            return std::chrono::microseconds(0);
        }
#endif

        // -----------------------------------------------------------------
        // Output helpers (sequential path only — use chrono duration_value_t)
        // -----------------------------------------------------------------

#ifndef SAMURAI_WITH_MPI

        [[nodiscard]] static double _to_seconds(const duration_value_t& d)
        {
            return std::chrono::duration_cast<std::chrono::duration<double>>(d).count();
        }

        [[nodiscard]] static double _percent(const duration_value_t& value, const duration_value_t& total)
        {
            if (total.count() == 0)
            {
                return 0.0;
            }
            return static_cast<double>(value.count() * 100) / static_cast<double>(total.count());
        }

        [[nodiscard]] static std::string _mcells_per_sec_str(uint64_t total_cells, const duration_value_t& elapsed)
        {
            if (total_cells == 0)
            {
                return {};
            }
            const double secs = _to_seconds(elapsed);
            if (secs <= 0.0)
            {
                return {};
            }
            return fmt::format("{:.1f}", static_cast<double>(total_cells) / secs / 1.0e6);
        }

        /** @brief Print one timer row with color and tree-connector prefix. */
        void _print_entry(const std::string& ctx_key,
                          const std::string& prefix,
                          const std::string& connector,
                          bool is_root,
                          const duration_value_t& grand_total,
                          const duration_value_t& parent_total,
                          bool show_mcells,
                          int nameWidth,
                          int elapsedWidth,
                          int percWidth,
                          int parWidth,
                          int callsWidth,
                          int mcellsWidth) const
        {
            auto it = _times.find(ctx_key);
            if (it == _times.end())
            {
                return;
            }
            const auto& entry = it->second;

            // Build the visible name: prefix (tree lines) + connector + display name
            const std::string display   = _display_name(ctx_key);
            const std::string full_name = prefix + connector + display;

            const double elapsed_s   = _to_seconds(entry.elapsed);
            const double perc_total  = _percent(entry.elapsed, grand_total);
            const double perc_parent = _percent(entry.elapsed, parent_total);

            const std::string perc_col = fmt::format("{:.1f}%", perc_total);
            const std::string par_col  = (!is_root) ? fmt::format("{:.1f}%", perc_parent) : std::string{};

            // Choose row style
            const fmt::text_style name_style = is_root ? fmt::emphasis::bold : _heat_style(perc_total);
            const fmt::text_style num_style  = is_root ? fmt::emphasis::bold : _heat_style(perc_total);
            const fmt::text_style dim_style  = fmt::emphasis::faint;

            // Print: name (styled) + numeric columns (styled) + calls (dim) + mcells (cyan)
            fmt::print(name_style, "{:<{}}", full_name, nameWidth);
            fmt::print(num_style, " {:>{}.3f}", elapsed_s, elapsedWidth);
            fmt::print(num_style, " {:>{}}", perc_col, percWidth);
            fmt::print(dim_style, " {:>{}}", par_col, parWidth);
            fmt::print(dim_style, " {:>{}}", entry.ntimes, callsWidth);
            if (show_mcells)
            {
                const std::string mcells_str = _mcells_per_sec_str(entry.total_cells, entry.elapsed);
                fmt::print(fmt::fg(fmt::terminal_color::cyan), " {:>{}}", mcells_str, mcellsWidth);
            }
            fmt::print("\n");
        }

        /** @brief Recursively print children of a ctx_key with tree connectors, sorted by elapsed. */
        void _print_children(const std::string& parent_key,
                             const std::string& prefix,
                             const duration_value_t& grand_total,
                             const std::map<std::string, std::vector<std::string>>& children,
                             bool show_mcells,
                             int nameWidth,
                             int elapsedWidth,
                             int percWidth,
                             int parWidth,
                             int callsWidth,
                             int mcellsWidth) const
        {
            auto it = children.find(parent_key);
            if (it == children.end())
            {
                return;
            }

            std::vector<std::string> sorted_children = it->second;
            _sort_by_elapsed(sorted_children);

            duration_value_t parent_elapsed = _zero_duration();
            if (auto pit = _times.find(parent_key); pit != _times.end())
            {
                parent_elapsed = pit->second.elapsed;
            }

            for (std::size_t i = 0; i < sorted_children.size(); ++i)
            {
                const bool is_last             = (i + 1 == sorted_children.size());
                const std::string connector    = is_last ? "`-- " : "+-- ";
                const std::string child_prefix = prefix + (is_last ? "    " : "|   ");

                _print_entry(sorted_children[i],
                             prefix,
                             connector,
                             /*is_root=*/false,
                             grand_total,
                             parent_elapsed,
                             show_mcells,
                             nameWidth,
                             elapsedWidth,
                             percWidth,
                             parWidth,
                             callsWidth,
                             mcellsWidth);
                _print_children(sorted_children[i],
                                child_prefix,
                                grand_total,
                                children,
                                show_mcells,
                                nameWidth,
                                elapsedWidth,
                                percWidth,
                                parWidth,
                                callsWidth,
                                mcellsWidth);
            }
        }

#endif // !SAMURAI_WITH_MPI

        // -----------------------------------------------------------------
        // Shared output helpers (sequential and MPI paths)
        // -----------------------------------------------------------------

        [[nodiscard]] bool _any_has_cells() const
        {
            return std::any_of(_times.begin(),
                               _times.end(),
                               [](const auto& kv)
                               {
                                   return kv.second.total_cells > 0;
                               });
        }

        [[nodiscard]] static std::size_t _depth_of(const std::string& ctx_key)
        {
            std::size_t depth = 0;
            std::size_t pos   = 0;
            while ((pos = ctx_key.find("::", pos)) != std::string::npos)
            {
                ++depth;
                pos += 2;
            }
            return depth;
        }

        /**
         * @brief Compute the name column width accounting for tree connector prefix.
         *
         * Each depth level adds 4 characters (prefix + connector).
         */
        [[nodiscard]] int _compute_name_width_with_tree(std::size_t min_width) const
        {
            std::size_t max_w = 0;
            for (const auto& [key, ignored] : _times)
            {
                const std::size_t depth = _depth_of(key);
                const std::string dname = _display_name(key);
                max_w                   = std::max(max_w, dname.size() + depth * 4);
            }
            return static_cast<int>(std::max(min_width, max_w + 2));
        }

        /** @brief Sort a list of ctx_keys by descending elapsed time (sequential) or descending ave (MPI via lambda). */
        void _sort_by_elapsed(std::vector<std::string>& keys) const
        {
            std::sort(keys.begin(),
                      keys.end(),
                      [this](const std::string& a, const std::string& b)
                      {
                          auto ia = _times.find(a);
                          auto ib = _times.find(b);
                          if (ia == _times.end())
                          {
                              return false;
                          }
                          if (ib == _times.end())
                          {
                              return true;
                          }
                          return ia->second.elapsed > ib->second.elapsed;
                      });
        }

        /**
         * @brief Return a fmt text_style heat-colored by fraction of total (green→yellow→red).
         */
        [[nodiscard]] static fmt::text_style _heat_style(double perc_total)
        {
            if (perc_total >= 20.0)
            {
                return fmt::fg(fmt::terminal_color::red);
            }
            if (perc_total >= 5.0)
            {
                return fmt::fg(fmt::terminal_color::yellow);
            }
            return fmt::fg(fmt::terminal_color::green);
        }
    };

    // =========================================================================
    // ScopedTimer – RAII guard
    // =========================================================================

    /**
     * @class ScopedTimer
     * @brief RAII timer guard that automatically builds the call-context hierarchy.
     *
     * Construct a `ScopedTimer` at the beginning of a scope; it calls
     * `Timers::start()` immediately and `Timers::stop()` at destruction.
     * Because `start()` pushes onto the thread-local stack, any timer started
     * while this one is alive becomes a child in the call tree.
     *
     * The same display name started from different call contexts is tracked
     * as a separate entry — so `"ghost update"` called from inside
     * `"mesh adaptation"` and from the top level both appear in the output.
     *
     * @code
     * {
     *     samurai::ScopedTimer t("mesh adaptation");
     *     // ...
     *     t.set_cells(mesh.nb_cells());
     * }
     * @endcode
     *
     * @note Non-copyable and non-movable.
     */
    class ScopedTimer
    {
      public:

        explicit ScopedTimer(std::string name, Timers& timers);
        explicit ScopedTimer(std::string name);

        ScopedTimer(const ScopedTimer&)            = delete;
        ScopedTimer& operator=(const ScopedTimer&) = delete;
        ScopedTimer(ScopedTimer&&)                 = delete;
        ScopedTimer& operator=(ScopedTimer&&)      = delete;

        void set_cells(uint64_t nb_cells) noexcept
        {
            m_cells = nb_cells;
        }

        ~ScopedTimer();

      private:

        std::string m_name;
        Timers& m_timers;
        uint64_t m_cells{0};
    };

    // =========================================================================
    // Global timer instance
    // =========================================================================

    namespace times
    {
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
        static Timers timers;
    } // namespace times

    // =========================================================================
    // ScopedTimer – out-of-line definitions (after times::timers is declared)
    // =========================================================================

    inline ScopedTimer::ScopedTimer(std::string name, Timers& timers)
        : m_name(std::move(name))
        , m_timers(timers)
    {
        m_timers.start(m_name);
    }

    inline ScopedTimer::ScopedTimer(std::string name)
        : ScopedTimer(std::move(name), times::timers)
    {
    }

    inline ScopedTimer::~ScopedTimer()
    {
        m_timers.stop(m_name, m_cells);
    }

} // namespace samurai
