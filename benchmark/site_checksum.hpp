// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// What a call site computes, reduced to a checksum per level, and the files
// of reference checksums generate_set_algebra_inputs writes with the static
// algebra. Each benchmark of a site computes the checksum of its backend once,
// outside the timed loop, and compares it with the reference
// (call_site_benchmarks.hpp), so that a backend that computes other sets
// does not go unnoticed.
//
// The checksum is computed from the traversal of the sets, not from their
// storage, so it is the same for every backend that computes the same cells.
// The sets a site evaluates or builds at one level give:
//   sets       how many sets the site passed at this level
//   intervals  their intervals, adjacent intervals of a row merged
//   cells      their cells
//   hash       FNV-1a over the 64-bit words: for each set in the order the
//              site passes them, a marker, then for each merged interval in
//              traversal order its start, its end and its (y, z)
// For a site that only asks whether its sets are empty, `intervals` counts
// the sets that are not, `cells` is 0, and the hash takes one word per set:
// 1 if it is empty, 0 if not.
//
// File format, text, one file per mesh or partition (checksum_file):
//   samurai-set-algebra-checksums <format version> <dim> <number of sites>
// then, per site:
//   site <name> <number of levels>
// followed by one line per level: level sets intervals cells hash (hex).

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "set_algebra_backend.hpp"
#include "site_inputs.hpp"

namespace bench
{
    struct LevelChecksum
    {
        static constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
        static constexpr std::uint64_t fnv_prime  = 1099511628211ULL;

        std::size_t sets      = 0;
        std::size_t intervals = 0;
        std::size_t cells     = 0;
        std::uint64_t hash    = fnv_offset;

        void mix(std::int64_t word)
        {
            auto bits = static_cast<std::uint64_t>(word);
            for (int byte = 0; byte < 8; ++byte)
            {
                hash ^= bits & 0xffU;
                hash *= fnv_prime;
                bits >>= 8U;
            }
        }

        bool operator==(const LevelChecksum&) const = default;
    };

    // By level.
    using SiteChecksum = std::map<std::size_t, LevelChecksum>;

    // Adds one set, interval by interval in traversal order, to the checksum
    // of its level. Merges an interval with the previous one when they are
    // adjacent in the same row, and skips the empty ones, so that a backend
    // that splits a row differently gives the same checksum.
    template <std::size_t dim>
    class SetHasher
    {
      public:

        static constexpr std::int64_t set_marker = -0x5e7a19e6;

        explicit SetHasher(LevelChecksum& level)
            : m_level(level)
        {
            ++m_level.sets;
            m_level.mix(set_marker);
        }

        SetHasher(const SetHasher&)            = delete;
        SetHasher& operator=(const SetHasher&) = delete;

        ~SetHasher()
        {
            flush();
        }

        template <class Interval, class Index>
        void operator()(const Interval& i, const Index& index)
        {
            if (i.start == i.end)
            {
                return;
            }
            std::array<std::int64_t, dim - 1> yz{};
            for (std::size_t k = 0; k + 1 < dim; ++k)
            {
                yz[k] = static_cast<std::int64_t>(index[k]);
            }
            if (m_pending && yz == m_yz && i.start == m_end)
            {
                m_end = i.end;
                return;
            }
            flush();
            m_pending = true;
            m_start   = i.start;
            m_end     = i.end;
            m_yz      = yz;
        }

      private:

        void flush()
        {
            if (!m_pending)
            {
                return;
            }
            ++m_level.intervals;
            m_level.cells += static_cast<std::size_t>(m_end - m_start);
            m_level.mix(m_start);
            m_level.mix(m_end);
            for (const auto y : m_yz)
            {
                m_level.mix(y);
            }
            m_pending = false;
        }

        LevelChecksum& m_level;
        bool m_pending       = false;
        std::int64_t m_start = 0;
        std::int64_t m_end   = 0;
        std::array<std::int64_t, dim - 1> m_yz{};
    };

    // Adds the set to the checksum of its level and returns its number of
    // intervals, as the traversal emits them.
    template <class B, std::size_t dim, class Set>
    std::size_t add_set(SiteChecksum& checksum, std::size_t level, const Set& set)
    {
        std::size_t intervals = 0;
        SetHasher<dim> hasher(checksum[level]);
        Algebra<B>::traverse(set,
                             [&](const auto& i, const auto& index)
                             {
                                 hasher(i, index);
                                 ++intervals;
                             });
        return intervals;
    }

    // The checksum of a site run by bench::run_site: site(f) passes each set
    // it evaluates to f(level, set), which returns its number of intervals.
    template <class B, std::size_t dim, class Site>
    SiteChecksum checksum_traverse_site(Site&& site)
    {
        SiteChecksum checksum;
        site(
            [&](std::size_t level, const auto& set)
            {
                return add_set<B, dim>(checksum, level, set);
            });
        return checksum;
    }

    // The checksum of a site run by bench::run_build_site: site(emit) passes
    // each level cell array it keeps to emit(level, lca).
    template <class B, std::size_t dim, class Site>
    SiteChecksum checksum_build_site(Site&& site)
    {
        SiteChecksum checksum;
        site(
            [&](std::size_t level, const auto& lca)
            {
                add_set<B, dim>(checksum, level, lca);
            });
        return checksum;
    }

    // The checksum of a site run by bench::run_empty_site: site(f) passes
    // each set it evaluates to f(level, set), which returns 1 if it is not
    // empty, 0 if it is.
    template <class B, class Site>
    SiteChecksum checksum_empty_site(Site&& site)
    {
        SiteChecksum checksum;
        site(
            [&](std::size_t level, const auto& set)
            {
                const bool is_empty = Algebra<B>::empty(set);
                auto& c             = checksum[level];
                ++c.sets;
                c.intervals += is_empty ? 0 : 1;
                c.mix(is_empty ? 1 : 0);
                return is_empty ? std::size_t{0} : std::size_t{1};
            });
        return checksum;
    }

    inline std::string describe(const LevelChecksum& c)
    {
        std::ostringstream out;
        out << c.sets << " sets, " << c.intervals << " intervals, " << c.cells << " cells, hash " << std::hex << c.hash;
        return out.str();
    }

    // Empty if `actual` equals `reference`, otherwise the first level where
    // they differ.
    inline std::string checksum_mismatch(const SiteChecksum& actual, const SiteChecksum& reference)
    {
        auto a = actual.begin();
        auto r = reference.begin();
        while (a != actual.end() || r != reference.end())
        {
            if (r == reference.end() || (a != actual.end() && a->first < r->first))
            {
                return "level " + std::to_string(a->first) + ": " + describe(a->second) + ", none in the reference";
            }
            if (a == actual.end() || r->first < a->first)
            {
                return "level " + std::to_string(r->first) + ": none, " + describe(r->second) + " in the reference";
            }
            if (!(a->second == r->second))
            {
                return "level " + std::to_string(a->first) + ": " + describe(a->second) + ", " + describe(r->second) + " in the reference";
            }
            ++a;
            ++r;
        }
        return {};
    }

    //////////////////////////////////////////////////////////////////////
    // Files
    //////////////////////////////////////////////////////////////////////

    inline constexpr const char* checksums_magic = "samurai-set-algebra-checksums";

    // The checksums of the sites on a whole mesh, or on a partition when
    // c.ranks > 0.
    template <std::size_t dim>
    std::filesystem::path checksum_file(const MeshCase& c)
    {
        auto path = (c.ranks > 0) ? partition_file<dim>(c) : inputs_file<dim>(c, false);
        path.replace_extension();
        path += "_checksums.txt";
        return path;
    }

    template <std::size_t dim>
    void write_checksums(const std::map<std::string, SiteChecksum>& sites, const std::filesystem::path& path)
    {
        std::ofstream out(path);
        out << checksums_magic << ' ' << inputs_format_version << ' ' << dim << ' ' << sites.size() << '\n';
        for (const auto& [name, checksum] : sites)
        {
            out << "site " << name << ' ' << checksum.size() << '\n';
            for (const auto& [level, c] : checksum)
            {
                out << level << ' ' << c.sets << ' ' << c.intervals << ' ' << c.cells << ' ' << std::hex << c.hash << std::dec << '\n';
            }
        }
        if (!out)
        {
            throw std::runtime_error("cannot write " + path.string());
        }
    }

    template <std::size_t dim>
    std::map<std::string, SiteChecksum> read_checksums(const std::filesystem::path& path)
    {
        auto file       = open_inputs(path);
        const auto fail = read_failure(path);
        std::string magic;
        int version         = 0;
        std::size_t d       = 0;
        std::size_t n_sites = 0;
        file >> magic >> version >> d >> n_sites;
        if (magic != checksums_magic || version != inputs_format_version || d != dim)
        {
            throw fail("not a set algebra checksums file of version " + std::to_string(inputs_format_version) + " in " + std::to_string(dim)
                       + "D");
        }
        std::map<std::string, SiteChecksum> sites;
        for (std::size_t s = 0; s < n_sites; ++s)
        {
            std::string key;
            std::string name;
            std::size_t n_levels = 0;
            file >> key >> name >> n_levels;
            if (key != "site")
            {
                throw fail("expected a site, read " + key);
            }
            auto& checksum = sites[name];
            for (std::size_t l = 0; l < n_levels; ++l)
            {
                std::size_t level = 0;
                LevelChecksum c;
                file >> level >> c.sets >> c.intervals >> c.cells >> std::hex >> c.hash >> std::dec;
                checksum[level] = c;
            }
            if (!file)
            {
                throw fail("truncated site " + name);
            }
        }
        return sites;
    }

    // The reference checksum of `site` on the mesh or partition c, from its
    // file, read once per binary. Throws if there is none.
    template <std::size_t dim>
    const SiteChecksum& reference_checksum(const MeshCase& c, const std::string& site)
    {
        static std::map<mesh_key_t, std::map<std::string, SiteChecksum>> cache;
        auto it = cache.find(c.key());
        if (it == cache.end())
        {
            it = cache.emplace(c.key(), read_checksums<dim>(checksum_file<dim>(c))).first;
        }
        const auto found = it->second.find(site);
        if (found == it->second.end())
        {
            throw std::runtime_error(checksum_file<dim>(c).string() + " has no checksum of " + site
                                     + ": run generate_set_algebra_inputs again");
        }
        return found->second;
    }
} // namespace bench
