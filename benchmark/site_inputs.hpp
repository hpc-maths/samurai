// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// The sets that the library call sites read from an MR-adapted mesh, held in
// plain CellArrays. The benchmarks of those sites read them from files, so
// they need neither the mesh, nor the MR machinery, nor the fields.
// generate_set_algebra_inputs adapts the meshes of mesh_cases() and writes
// the files; their content does not change when the set algebra does.
//
// File format, text, one record per set:
//   samurai-set-algebra-inputs <format version> <dim>
//   levels <min_level> <max_level>
//   config <graduation_width> <max_stencil_radius>
//   ca <name> <number of intervals>, then per interval: level start end [y [z]]
//   lca corner <level> <number of intervals>, then per interval: start end [y [z]]
// Each interval is the row [start, end) of cells at (y, z) of its level.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/level_cell_array.hpp>
#include <samurai/level_cell_list.hpp>
#include <samurai/stencil.hpp>

#include "set_algebra_fixtures.hpp"

#ifndef SAMURAI_SET_ALGEBRA_INPUTS_DIR
#define SAMURAI_SET_ALGEBRA_INPUTS_DIR "set_algebra_inputs"
#endif

namespace bench
{
    //////////////////////////////////////////////////////////////////////
    // The adapted meshes
    //////////////////////////////////////////////////////////////////////

    template <std::size_t dim>
    constexpr std::size_t default_max_level()
    {
        return (dim == 1) ? 16 : ((dim == 2) ? 10 : 7);
    }

    // One adapted mesh, refined to max_level along the surfaces of nballs
    // balls: more balls means more, smaller fronts. stencil_radius is the max
    // stencil radius of the mesh configuration, which sets the ghost width.
    // The benchmark arguments, in this order.
    struct MeshCase
    {
        std::size_t max_level;
        std::size_t nballs;
        int stencil_radius;

        auto key() const
        {
            return std::tuple{max_level, nballs, stencil_radius};
        }
    };

    using mesh_key_t = decltype(std::declval<MeshCase>().key());

    inline MeshCase mesh_case(const benchmark::State& state)
    {
        return {static_cast<std::size_t>(state.range(0)), static_cast<std::size_t>(state.range(1)), static_cast<int>(state.range(2))};
    }

    // Stencil radius 1 is the upwind scheme. The case of radius 3, the
    // stencil of a WENO5 scheme, triples the ghost width.
    template <std::size_t dim>
    std::vector<MeshCase> mesh_cases()
    {
        constexpr std::size_t max_level = default_max_level<dim>();
        if constexpr (dim == 1)
        {
            // a 1D front is a point: many of them to get a non-trivial mesh
            return {
                {max_level, 16,  1},
                {max_level, 256, 1},
                {max_level, 256, 3}
            };
        }
        else if constexpr (dim == 2)
        {
            return {
                {max_level,     1,  1},
                {max_level,     16, 1},
                {max_level + 2, 16, 1},
                {max_level,     16, 3}
            };
        }
        else
        {
            return {
                {max_level, 1, 1},
                {max_level, 8, 1},
                {max_level, 8, 3}
            };
        }
    }

    // The intervals of a CellArray, level by level.
    template <std::size_t dim, class TInterval, std::size_t max_size, class F>
    void for_each_interval(const samurai::CellArray<dim, TInterval, max_size>& ca, F&& f)
    {
        for (std::size_t level = ca.min_level(); level <= ca.max_level(); ++level)
        {
            bench::for_each_interval(ca[level], f);
        }
    }

    // Registers b on each case of `cases`.
    inline void mesh_args(benchmark::internal::Benchmark* b, const std::vector<MeshCase>& cases)
    {
        for (const auto& c : cases)
        {
            b->Args({static_cast<int64_t>(c.max_level), static_cast<int64_t>(c.nballs), c.stencil_radius});
        }
        b->Unit(benchmark::kMicrosecond);
    }

    template <std::size_t dim>
    void mesh_args(benchmark::internal::Benchmark* b)
    {
        mesh_args(b, mesh_cases<dim>());
    }

    // So that every report says what it was measured on.
    template <std::size_t dim>
    void report_mesh(benchmark::State& state, const samurai::CellArray<dim>& cells)
    {
        std::size_t intervals = 0;
        bench::for_each_interval(cells,
                                 [&](std::size_t, const auto&, const auto&)
                                 {
                                     ++intervals;
                                 });
        state.counters["leaves"]   = static_cast<double>(cells.nb_cells());
        state.counters["leaf_itv"] = static_cast<double>(intervals);
        state.counters["levels"]   = static_cast<double>(cells.max_level() - cells.min_level() + 1);
    }

    //////////////////////////////////////////////////////////////////////
    // The sets of one mesh
    //////////////////////////////////////////////////////////////////////

    // MRMesh::config_t::prediction_stencil_radius of the adapted meshes.
    inline constexpr int prediction_stencil_radius = 1;

    // Named after the MRMesh members they are copied from (make_site_inputs
    // in mr_fixtures.hpp), with accessors where the mesh has one. CA holds one
    // level cell array per level: samurai's CellArray in the files, a
    // BackendCellArray for a backend with level cell arrays of its own.
    template <std::size_t dim, class CA = samurai::CellArray<dim>>
    struct SiteInputs
    {
        using ca_t  = CA;
        using lca_t = std::decay_t<decltype(std::declval<const ca_t&>()[0])>;

        // mesh.min_level() and mesh.max_level(): the bounds of the mesh
        // configuration, not of the cells.
        std::size_t min_level = 0;
        std::size_t max_level = 0;
        // mesh.graduation_width() and mesh.max_stencil_radius(), which
        // make_graduation reads.
        std::size_t graduation_width = 1;
        int max_stencil_radius       = 1;

        ca_t cells;
        ca_t cells_and_ghosts;
        ca_t proj_cells;
        ca_t reference;
        ca_t union_; // mesh.get_union()
        ca_t domain_pyramid;
        ca_t subdomain_pyramid;
        // mesh.corner(d) for each d, in the order of for_each_diagonal_direction
        std::vector<lca_t> corners;
        // The cells of the next adaptation step before their graduation: what
        // Adapt::harten (mr/adapt.hpp) passes to make_graduation at its first
        // iteration, update_cell_array_from_tag(cells, tag), when the fronts
        // have moved by one step. Not graded.
        ca_t ungraded_cells;
        // The four sets update_cell_array_from_tag (algorithm/graduation.hpp)
        // builds from the tags before it computes ungraded_cells, the members
        // of the samurai::TagChanges that cells_changed_by_tag returns: the
        // parents of the coarsened cells (add_m), the coarsened cells
        // (remove_m), the children of the refined cells (add_p) and the
        // refined cells (remove_p).
        ca_t tag_add_m;
        ca_t tag_remove_m;
        ca_t tag_add_p;
        ca_t tag_remove_p;

        // Member by member: what the generator checks after it reads back a
        // file it wrote.
        bool operator==(const SiteInputs&) const = default;

        // MRMeshId::all_cells is MRMeshId::reference.
        const ca_t& all_cells() const
        {
            return reference;
        }

        const lca_t& domain() const
        {
            return domain_pyramid[max_level];
        }

        const lca_t& domain(std::size_t level) const
        {
            return domain_pyramid[level];
        }

        const lca_t& subdomain() const
        {
            return subdomain_pyramid[max_level];
        }

        const lca_t& subdomain(std::size_t level) const
        {
            return subdomain_pyramid[level];
        }

        // The lookup of Mesh_base::corner.
        const lca_t& corner(const samurai::DirectionVector<dim>& direction) const
        {
            std::size_t i           = 0;
            std::size_t i_direction = 0;
            samurai::for_each_diagonal_direction<dim>(
                [&](const auto& dir)
                {
                    if (dir == direction)
                    {
                        i_direction = i;
                    }
                    ++i;
                });
            return corners[i_direction];
        }
    };

    //////////////////////////////////////////////////////////////////////
    // Files
    //////////////////////////////////////////////////////////////////////

    inline constexpr int inputs_format_version = 3;

    // The CellArrays of the file, by record name. Inputs is SiteInputs<dim>,
    // const or not.
    template <class Inputs>
    auto named_cell_arrays(Inputs& in)
    {
        return std::array{
            std::pair{"cells",             &in.cells            },
            std::pair{"cells_and_ghosts",  &in.cells_and_ghosts },
            std::pair{"proj_cells",        &in.proj_cells       },
            std::pair{"reference",         &in.reference        },
            std::pair{"union",             &in.union_           },
            std::pair{"domain_pyramid",    &in.domain_pyramid   },
            std::pair{"subdomain_pyramid", &in.subdomain_pyramid},
            std::pair{"ungraded_cells",    &in.ungraded_cells   },
            std::pair{"tag_add_m",         &in.tag_add_m        },
            std::pair{"tag_remove_m",      &in.tag_remove_m     },
            std::pair{"tag_add_p",         &in.tag_add_p        },
            std::pair{"tag_remove_p",      &in.tag_remove_p     },
        };
    }

    // SAMURAI_SET_ALGEBRA_INPUTS in the environment overrides the directory
    // set at configure time, to share one set of files between builds.
    inline std::filesystem::path inputs_dir()
    {
        const char* dir = std::getenv("SAMURAI_SET_ALGEBRA_INPUTS"); // NOLINT(concurrency-mt-unsafe)
        return (dir != nullptr) ? dir : SAMURAI_SET_ALGEBRA_INPUTS_DIR;
    }

    // The name of the files of the mesh c, without the suffix of the file.
    template <std::size_t dim>
    std::string mesh_stem(const MeshCase& c)
    {
        return std::to_string(dim) + "d_level" + std::to_string(c.max_level) + "_balls" + std::to_string(c.nballs) + "_radius"
             + std::to_string(c.stencil_radius);
    }

    template <std::size_t dim>
    std::filesystem::path inputs_file(const MeshCase& c, bool moved)
    {
        return inputs_dir() / (mesh_stem<dim>(c) + (moved ? "_moved" : "") + ".txt");
    }

    // Set is a LevelCellArray or a CellArray.
    template <class Set>
    void write_intervals(std::ostream& out, const Set& set, bool with_level)
    {
        bench::for_each_interval(set,
                                 [&](std::size_t level, const auto& i, const auto& index)
                                 {
                                     if (with_level)
                                     {
                                         out << level << ' ';
                                     }
                                     out << i.start << ' ' << i.end;
                                     for (const auto& y : index)
                                     {
                                         out << ' ' << y;
                                     }
                                     out << '\n';
                                 });
    }

    inline constexpr const char* inputs_magic = "samurai-set-algebra-inputs";

    template <std::size_t dim>
    void write_site_inputs(const SiteInputs<dim>& in, const std::filesystem::path& path)
    {
        std::ofstream out(path);
        out << inputs_magic << ' ' << inputs_format_version << ' ' << dim << '\n';
        out << "levels " << in.min_level << ' ' << in.max_level << '\n';
        out << "config " << in.graduation_width << ' ' << in.max_stencil_radius << '\n';
        for (const auto& [name, ca] : named_cell_arrays(in))
        {
            std::size_t n = 0;
            bench::for_each_interval(*ca,
                                     [&n](std::size_t, const auto&, const auto&)
                                     {
                                         ++n;
                                     });
            out << "ca " << name << ' ' << n << '\n';
            write_intervals(out, *ca, true);
        }
        for (const auto& corner : in.corners)
        {
            out << "lca corner " << corner.level() << ' ' << corner[0].size() << '\n';
            write_intervals(out, corner, false);
        }
        if (!out)
        {
            throw std::runtime_error("cannot write " + path.string());
        }
    }

    // Reads the interval of a record line that follows its level.
    template <class LCL>
    void read_interval(std::istream& file, LCL& lcl)
    {
        typename LCL::coord_index_t start = 0;
        typename LCL::coord_index_t end   = 0;
        typename LCL::index_yz_t index;
        file >> start >> end;
        for (auto& y : index)
        {
            file >> y;
        }
        lcl[index].add_interval({start, end});
    }

    // Reads the first three lines of a file.
    template <std::size_t dim, class Fail>
    void read_header(std::istream& file, SiteInputs<dim>& in, const Fail& fail)
    {
        std::string magic;
        int version   = 0;
        std::size_t d = 0;
        std::string levels_key;
        std::string config_key;
        file >> magic >> version >> d >> levels_key >> in.min_level >> in.max_level >> config_key >> in.graduation_width
            >> in.max_stencil_radius;
        if (magic != inputs_magic || version != inputs_format_version || d != dim || levels_key != "levels" || config_key != "config")
        {
            throw fail("not a set algebra inputs file of version " + std::to_string(inputs_format_version) + " in " + std::to_string(dim)
                       + "D");
        }
    }

    // Reads the record "ca <name> <n>" from <n> on: the n intervals of the
    // CellArray `name`, one per line, each after its level.
    template <std::size_t dim, class Fail>
    void read_cell_array(std::istream& file, const std::string& name, SiteInputs<dim>& in, const Fail& fail)
    {
        std::size_t n = 0;
        file >> n;
        samurai::CellList<dim> cl;
        for (std::size_t k = 0; k < n; ++k)
        {
            std::size_t level = 0;
            file >> level;
            read_interval(file, cl[level]);
        }
        bool found = false;
        for (const auto& [ca_name, ca] : named_cell_arrays(in))
        {
            if (name == ca_name)
            {
                *ca   = typename SiteInputs<dim>::ca_t(cl);
                found = true;
            }
        }
        if (!found)
        {
            throw fail("unknown set " + name);
        }
    }

    // Reads the record "lca corner <level> <n>" from <level> on: the n
    // intervals of the next corner, one per line.
    template <std::size_t dim>
    void read_corner(std::istream& file, SiteInputs<dim>& in)
    {
        std::size_t level = 0;
        std::size_t n     = 0;
        file >> level >> n;
        samurai::LevelCellList<dim> lcl(level);
        for (std::size_t k = 0; k < n; ++k)
        {
            read_interval(file, lcl);
        }
        in.corners.emplace_back(lcl);
    }

    // Reads the rest of the record whose first two words are `key` and
    // `name`.
    template <std::size_t dim, class Fail>
    void read_record(std::istream& file, const std::string& key, const std::string& name, SiteInputs<dim>& in, const Fail& fail)
    {
        if (key == "ca")
        {
            read_cell_array(file, name, in, fail);
        }
        else if (key == "lca" && name == "corner")
        {
            read_corner(file, in);
        }
        else
        {
            throw fail("unknown record " + key + " " + name);
        }
        if (!file)
        {
            throw fail("truncated record " + key + " " + name);
        }
    }

    inline std::ifstream open_inputs(const std::filesystem::path& path)
    {
        std::ifstream file(path);
        if (!file)
        {
            throw std::runtime_error("cannot open " + path.string() + ": run generate_set_algebra_inputs first");
        }
        return file;
    }

    inline auto read_failure(const std::filesystem::path& path)
    {
        return [path](const std::string& what)
        {
            return std::runtime_error(path.string() + ": " + what);
        };
    }

    template <std::size_t dim>
    SiteInputs<dim> read_site_inputs(const std::filesystem::path& path)
    {
        auto file       = open_inputs(path);
        const auto fail = read_failure(path);
        SiteInputs<dim> in;
        read_header(file, in, fail);

        std::string key;
        std::string name;
        while (file >> key)
        {
            file >> name;
            read_record(file, key, name, in, fail);
        }
        return in;
    }

    // Read once per binary. Reports the mesh it returns: a benchmark that
    // reads two meshes reports the last one.
    template <std::size_t dim>
    const SiteInputs<dim>& site_inputs(benchmark::State& state, bool moved = false)
    {
        static std::map<std::tuple<mesh_key_t, bool>, SiteInputs<dim>> cache;
        const auto c   = mesh_case(state);
        const auto key = std::tuple{c.key(), moved};
        auto it        = cache.find(key);
        if (it == cache.end())
        {
            try
            {
                it = cache.emplace(key, read_site_inputs<dim>(inputs_file<dim>(c, moved))).first;
            }
            catch (const std::exception& e)
            {
                std::cerr << e.what() << std::endl;
                std::exit(EXIT_FAILURE); // NOLINT(concurrency-mt-unsafe)
            }
        }
        report_mesh(state, it->second.cells);
        return it->second;
    }

    //////////////////////////////////////////////////////////////////////
    // The sets of one mesh in the level cell arrays of a backend
    //////////////////////////////////////////////////////////////////////

    // The counterpart of samurai's CellArray for a backend B
    // (set_algebra_backend.hpp): one B::lca_t<dim> per level, with the same
    // levels and bounds.
    template <class LCA>
    class BackendCellArray
    {
      public:

        BackendCellArray() = default;

        BackendCellArray(std::vector<LCA> levels, std::size_t min_level, std::size_t max_level)
            : m_levels(std::move(levels))
            , m_min_level(min_level)
            , m_max_level(max_level)
        {
        }

        const LCA& operator[](std::size_t level) const
        {
            return m_levels[level];
        }

        std::size_t min_level() const
        {
            return m_min_level;
        }

        std::size_t max_level() const
        {
            return m_max_level;
        }

      private:

        std::vector<LCA> m_levels;
        std::size_t m_min_level = 0;
        std::size_t m_max_level = 0;
    };

    // The inputs a backend reads: the samurai sets themselves when its level
    // cell arrays are samurai's, a copy in its own otherwise.
    template <class B, std::size_t dim>
    using backend_inputs_t = std::conditional_t<std::is_same_v<typename B::template lca_t<dim>, typename SiteInputs<dim>::lca_t>,
                                                SiteInputs<dim>,
                                                SiteInputs<dim, BackendCellArray<typename B::template lca_t<dim>>>>;

    // Copies lca into a level cell array of B, through its builder.
    template <class B, std::size_t dim, class TInterval>
    auto to_backend(const samurai::LevelCellArray<dim, TInterval>& lca)
    {
        auto builder = B::template builder<dim>(lca.level());
        bench::for_each_interval(lca,
                                 [&](std::size_t, const auto& i, const auto& index)
                                 {
                                     builder.add_interval(i, index);
                                 });
        return builder.build();
    }

    template <class B, std::size_t dim, class TInterval, std::size_t max_size>
    auto to_backend(const samurai::CellArray<dim, TInterval, max_size>& ca)
    {
        std::vector<typename B::template lca_t<dim>> levels;
        levels.reserve(max_size + 1);
        for (std::size_t level = 0; level <= max_size; ++level)
        {
            levels.push_back(to_backend<B>(ca[level]));
        }
        return BackendCellArray<typename B::template lca_t<dim>>(std::move(levels), ca.min_level(), ca.max_level());
    }

    template <class B, std::size_t dim>
    backend_inputs_t<B, dim> to_backend(const SiteInputs<dim>& in)
    {
        backend_inputs_t<B, dim> out;
        out.min_level          = in.min_level;
        out.max_level          = in.max_level;
        out.graduation_width   = in.graduation_width;
        out.max_stencil_radius = in.max_stencil_radius;
        const auto cell_arrays = named_cell_arrays(in);
        const auto out_arrays  = named_cell_arrays(out);
        for (std::size_t k = 0; k < cell_arrays.size(); ++k)
        {
            *out_arrays[k].second = to_backend<B>(*cell_arrays[k].second);
        }
        for (const auto& corner : in.corners)
        {
            out.corners.push_back(to_backend<B>(corner));
        }
        return out;
    }

    // site_inputs() for backend B. The copy, if B needs one, is made once,
    // outside the timed loop.
    template <class B, std::size_t dim>
    const backend_inputs_t<B, dim>& backend_inputs(benchmark::State& state, bool moved = false)
    {
        const auto& in = site_inputs<dim>(state, moved);
        if constexpr (std::is_same_v<backend_inputs_t<B, dim>, SiteInputs<dim>>)
        {
            return in;
        }
        else
        {
            static std::map<const SiteInputs<dim>*, backend_inputs_t<B, dim>> cache;
            auto it = cache.find(&in);
            if (it == cache.end())
            {
                it = cache.emplace(&in, to_backend<B>(in)).first;
            }
            return it->second;
        }
    }
} // namespace bench
