// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <fmt/format.h>
#include <fmt/ostream.h>

#ifdef SAMURAI_WITH_MPI
#include <mpi.h>
#endif

#include <samurai/arguments.hpp>

namespace samurai
{
    namespace io
    {
        // Tags & helpers
        struct all_t
        {
        };

        struct root_t
        {
        };

        struct rank_t
        {
            int value;
        };

        inline constexpr all_t all{};
        inline constexpr root_t root{};

        inline constexpr rank_t rank(int v)
        {
            return {v};
        }

        inline int current_rank()
        {
#ifdef SAMURAI_WITH_MPI
            int r = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &r);
            return r;
#else
            return 0;
#endif
        }

        namespace detail
        {
            inline bool allow_default_print()
            {
#ifdef SAMURAI_WITH_MPI
                return !(samurai::args::print_root_only && current_rank() != 0);
#else
                return true;
#endif
            }

            // Default (non-scoped) printing helper
            template <class... Args>
            inline void do_print(FILE* s, fmt::format_string<Args...> f, Args&&... a)
            {
                if (allow_default_print())
                {
                    fmt::print(s, f, std::forward<Args>(a)...);
                }
            }

            inline bool should_print(all_t)
            {
                return true;
            }

            inline bool should_print(root_t)
            {
                return current_rank() == 0;
            }

            inline bool should_print(rank_t r)
            {
                return current_rank() == r.value;
            }

            // Scoped printing helper
            template <class Scope, class... Args>
            inline void do_print(FILE* s, Scope sc, fmt::format_string<Args...> f, Args&&... a)
            {
                if (should_print(sc))
                {
                    fmt::print(s, f, std::forward<Args>(a)...);
                }
            }
        }

        // stdout
        template <class... Args>
        inline void print(fmt::format_string<Args...> f, Args&&... a)
        {
            detail::do_print(stdout, f, std::forward<Args>(a)...);
        }

        template <class Scope, class... Args>
        inline void print(Scope sc, fmt::format_string<Args...> f, Args&&... a)
        {
            detail::do_print(stdout, sc, f, std::forward<Args>(a)...);
        }

        // stderr
        template <class... Args>
        inline void eprint(fmt::format_string<Args...> f, Args&&... a)
        {
            detail::do_print(stderr, f, std::forward<Args>(a)...);
        }

        template <class Scope, class... Args>
        inline void eprint(Scope sc, fmt::format_string<Args...> f, Args&&... a)
        {
            detail::do_print(stderr, sc, f, std::forward<Args>(a)...);
        }
    } // namespace io
} // namespace samurai
