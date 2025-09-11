// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <fmt/format.h>
#include <fmt/ostream.h>

#ifdef SAMURAI_WITH_MPI
#  include <mpi.h>
#endif

#include "arguments.hpp"

namespace samurai
{
    namespace io
    {
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
        inline constexpr rank_t rank(int value)
        {
            return rank_t{value};
        }

#ifdef SAMURAI_WITH_MPI
        inline int current_rank()
        {
            int r = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &r);
            return r;
        }
#else
        inline int current_rank()
        {
            return 0;
        }
#endif

        // stdout printers
        template <class... Args>
        inline void print(all_t, fmt::format_string<Args...> fmt, Args&&... args)
        {
            fmt::print(fmt, std::forward<Args>(args)...);
        }

        template <class... Args>
        inline void print(root_t, fmt::format_string<Args...> fmt, Args&&... args)
        {
            if (current_rank() == 0)
            {
                fmt::print(fmt, std::forward<Args>(args)...);
            }
        }

        template <class... Args>
        inline void print(rank_t r, fmt::format_string<Args...> fmt, Args&&... args)
        {
            if (current_rank() == r.value)
            {
                fmt::print(fmt, std::forward<Args>(args)...);
            }
        }

        template <class... Args>
        inline void print(fmt::format_string<Args...> fmt, Args&&... args)
        {
#ifdef SAMURAI_WITH_MPI
            if (samurai::args::print_root_only && current_rank() != 0)
            {
                return;
            }
#endif
            fmt::print(fmt, std::forward<Args>(args)...);
        }

        // stderr printers
        template <class... Args>
        inline void eprint(all_t, fmt::format_string<Args...> fmt, Args&&... args)
        {
            fmt::print(stderr, fmt, std::forward<Args>(args)...);
        }

        template <class... Args>
        inline void eprint(root_t, fmt::format_string<Args...> fmt, Args&&... args)
        {
            if (current_rank() == 0)
            {
                fmt::print(stderr, fmt, std::forward<Args>(args)...);
            }
        }

        template <class... Args>
        inline void eprint(rank_t r, fmt::format_string<Args...> fmt, Args&&... args)
        {
            if (current_rank() == r.value)
            {
                fmt::print(stderr, fmt, std::forward<Args>(args)...);
            }
        }

        template <class... Args>
        inline void eprint(fmt::format_string<Args...> fmt, Args&&... args)
        {
#ifdef SAMURAI_WITH_MPI
            if (samurai::args::print_root_only && current_rank() != 0)
            {
                return;
            }
#endif
            fmt::print(stderr, fmt, std::forward<Args>(args)...);
        }
    } // namespace io
} // namespace samurai

