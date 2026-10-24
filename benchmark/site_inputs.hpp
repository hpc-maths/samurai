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
//   geometry <domain is a box: 0 or 1> <periodic along x: 0 or 1> [y [z]]
//   ca <name> <number of intervals>, then per interval: level start end [y [z]]
//   lca corner <level> <number of intervals>, then per interval: start end [y [z]]
// Each interval is the row [start, end) of cells at (y, z) of its level.
//
// A partition file holds a mesh split into subdomains, one block per rank:
//   samurai-set-algebra-partition <format version> <dim> <number of ranks>
// then, for each rank, the records of an inputs file (from its first line),
// with the sets of the mesh of that rank, followed by
//   ranks candidates <n> <rank>...
//   ranks neighbours <n> <rank>...
//
// Next to the files of each mesh and partition, a file of the same name
// ending in _checksums.txt holds what each call site computes on it with the
// static algebra (format in site_checksum.hpp).

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
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

    // The domain of a mesh (mr_fixtures.hpp builds it).
    enum class Geometry : int
    {
        box      = 0, // the unit box
        hole     = 1, // the unit box without the box [0.5, 0.75)^dim, built by DomainBuilder::remove
        periodic = 2, // the unit box, periodic along every axis
    };

    // One adapted mesh, refined to max_level along the surfaces of nballs
    // balls: more balls means more, smaller fronts. stencil_radius is the max
    // stencil radius of the mesh configuration, which sets the ghost width.
    // The benchmark arguments, in this order.
    struct MeshCase
    {
        std::size_t max_level;
        std::size_t nballs;
        int stencil_radius;
        Geometry geometry = Geometry::box;
        // The number of subdomains the mesh is split into, 0 for the whole
        // mesh (see partition_cases).
        std::size_t ranks = 0;

        auto key() const
        {
            return std::tuple{max_level, nballs, stencil_radius, geometry, ranks};
        }

        // The case of the whole mesh.
        MeshCase whole() const
        {
            return {max_level, nballs, stencil_radius, geometry};
        }
    };

    using mesh_key_t = decltype(std::declval<MeshCase>().key());

    inline MeshCase mesh_case(const benchmark::State& state)
    {
        return {static_cast<std::size_t>(state.range(0)),
                static_cast<std::size_t>(state.range(1)),
                static_cast<int>(state.range(2)),
                static_cast<Geometry>(state.range(3))};
    }

    // The case of a partition benchmark, whose fifth argument is the number
    // of subdomains.
    inline MeshCase partition_case(const benchmark::State& state)
    {
        auto c  = mesh_case(state);
        c.ranks = static_cast<std::size_t>(state.range(4));
        return c;
    }

    // The sweep over the max level, at the number of balls of the main mesh
    // of each dimension and stencil radius 1: for curves of the time against
    // the number of intervals. Only with the sweep (sweep_enabled): its 3D
    // meshes are the largest inputs. The max level stays below the
    // max_refinement_level of the mesh configuration (20): an MR mesh puts
    // prediction ghosts one level above its max level, and make_graduation
    // keeps one LevelCellArray per level below max_refinement_level only.
    template <std::size_t dim>
    std::vector<MeshCase> sweep_cases()
    {
        std::vector<std::size_t> levels;
        std::size_t nballs = 0;
        if constexpr (dim == 1)
        {
            levels = {12, 14, 16, 18, 19};
            nballs = 256;
        }
        else if constexpr (dim == 2)
        {
            levels = {8, 9, 10, 11, 12, 13};
            nballs = 16;
        }
        else
        {
            levels = {5, 6, 7, 8, 9};
            nballs = 8;
        }
        std::vector<MeshCase> cases;
        for (const std::size_t level : levels)
        {
            cases.push_back({level, nballs, 1});
        }
        return cases;
    }

    // Stencil radius 1 is the upwind scheme. The case of radius 3, the
    // stencil of a WENO5 scheme, triples the ghost width. The domain with a
    // hole has the scaling factor DomainBuilder computes for it, 1/4: at max
    // level L it has the resolution of the unit box at L + 2, hence its lower
    // max level. With the sweep, the meshes of sweep_cases follow.
    template <std::size_t dim>
    std::vector<MeshCase> mesh_cases()
    {
        constexpr std::size_t max_level = default_max_level<dim>();
        std::vector<MeshCase> cases;
        if constexpr (dim == 1)
        {
            // a 1D front is a point: many of them to get a non-trivial mesh
            cases = {
                {max_level, 16, 1},
                {max_level, 256, 1},
                {max_level, 256, 3},
                {max_level - 2, 256, 1, Geometry::hole},
                {max_level, 256, 1, Geometry::periodic}
            };
        }
        else if constexpr (dim == 2)
        {
            cases = {
                {max_level, 1, 1},
                {max_level, 16, 1},
                {max_level + 2, 16, 1},
                {max_level, 16, 3},
                {max_level - 2, 16, 1, Geometry::hole},
                {max_level, 16, 1, Geometry::periodic}
            };
        }
        else
        {
            cases = {
                {max_level, 1, 1},
                {max_level, 8, 1},
                {max_level, 8, 3},
                {max_level - 2, 8, 1, Geometry::hole},
                {max_level, 8, 1, Geometry::periodic}
            };
        }
        if constexpr (sweep_enabled)
        {
            for (const auto& c : sweep_cases<dim>())
            {
                if (std::find_if(cases.begin(),
                                 cases.end(),
                                 [&](const MeshCase& other)
                                 {
                                     return other.key() == c.key();
                                 })
                    == cases.end())
                {
                    cases.push_back(c);
                }
            }
        }
        return cases;
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

    // The partitions of the main mesh of each dimension, in the box and
    // periodic: the mesh cut along a Hilbert curve into `ranks` pieces of
    // equal number of cells (mr_fixtures.hpp, make_partition), as the SFC
    // load balancer cuts it with one atom per cell and a unit weight.
    template <std::size_t dim>
    std::vector<MeshCase> partition_cases()
    {
        constexpr std::size_t max_level = default_max_level<dim>();
        constexpr std::size_t nballs    = (dim == 1) ? 256 : ((dim == 2) ? 16 : 8);
        std::vector<MeshCase> cases;
        for (const std::size_t ranks : std::initializer_list<std::size_t>{2, 4, 16})
        {
            cases.push_back({max_level, nballs, 1, Geometry::box, ranks});
        }
        cases.push_back({max_level, nballs, 1, Geometry::periodic, 4});
        return cases;
    }

    // Registers b on each case of `cases`, with the number of subdomains as
    // fifth argument when it is a partition.
    inline void mesh_args(benchmark::internal::Benchmark* b, const std::vector<MeshCase>& cases)
    {
        for (const auto& c : cases)
        {
            std::vector<int64_t> args{static_cast<int64_t>(c.max_level),
                                      static_cast<int64_t>(c.nballs),
                                      c.stencil_radius,
                                      static_cast<int64_t>(c.geometry)};
            if (c.ranks > 0)
            {
                args.push_back(static_cast<int64_t>(c.ranks));
            }
            b->Args(args);
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
        // mesh.domain().is_box(): the library takes other paths for a domain
        // that is not a box (update_outer_ghost.hpp). It cannot be read from
        // the domain, which is a box only when built from one.
        bool domain_is_box = true;
        // mesh.periodicity()
        std::array<bool, dim> periodic{};
        // mesh.domain().min_indices() and max_indices(): the bounds of the
        // domain at max_level, which give the periodic shifts. Not in the
        // file: set from the domain when it is read (set_domain_bounds).
        std::array<int, dim> domain_min{};
        std::array<int, dim> domain_max{};

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

        // In a partition, the ranks find_neighbourhood checks
        // (Mesh_base::find_neighbourhood, mesh.hpp:1309-1413: those whose
        // bounding box may touch the subdomain), and those it keeps, the MPI
        // neighbourhood, in its order. Empty for a whole mesh.
        std::vector<std::size_t> candidates;
        std::vector<std::size_t> neighbours;

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

    // Sets SiteInputs::domain_min and domain_max from the domain.
    template <std::size_t dim>
    void set_domain_bounds(SiteInputs<dim>& in)
    {
        const auto min_indices = in.domain().min_indices();
        const auto max_indices = in.domain().max_indices();
        for (std::size_t d = 0; d < dim; ++d)
        {
            in.domain_min[d] = min_indices[d];
            in.domain_max[d] = max_indices[d];
        }
    }

    //////////////////////////////////////////////////////////////////////
    // Files
    //////////////////////////////////////////////////////////////////////

    inline constexpr int inputs_format_version = 5;

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

    inline std::string geometry_suffix(Geometry geometry)
    {
        switch (geometry)
        {
            case Geometry::box:
                return "";
            case Geometry::hole:
                return "_hole";
            case Geometry::periodic:
                return "_periodic";
        }
        return "";
    }

    // The name of the files of the mesh c, without the suffix of the file.
    template <std::size_t dim>
    std::string mesh_stem(const MeshCase& c)
    {
        return std::to_string(dim) + "d_level" + std::to_string(c.max_level) + "_balls" + std::to_string(c.nballs) + "_radius"
             + std::to_string(c.stencil_radius) + geometry_suffix(c.geometry);
    }

    template <std::size_t dim>
    std::filesystem::path inputs_file(const MeshCase& c, bool moved)
    {
        return inputs_dir() / (mesh_stem<dim>(c) + (moved ? "_moved" : "") + ".txt");
    }

    template <std::size_t dim>
    std::filesystem::path partition_file(const MeshCase& c)
    {
        return inputs_dir() / (mesh_stem<dim>(c) + "_ranks" + std::to_string(c.ranks) + ".txt");
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

    inline constexpr const char* inputs_magic    = "samurai-set-algebra-inputs";
    inline constexpr const char* partition_magic = "samurai-set-algebra-partition";

    inline void write_ranks(std::ostream& out, const char* name, const std::vector<std::size_t>& ranks)
    {
        out << "ranks " << name << ' ' << ranks.size();
        for (const std::size_t r : ranks)
        {
            out << ' ' << r;
        }
        out << '\n';
    }

    // The records of one mesh; with its candidates and neighbours in a
    // partition.
    template <std::size_t dim>
    void write_block(std::ostream& out, const SiteInputs<dim>& in, bool partition)
    {
        out << inputs_magic << ' ' << inputs_format_version << ' ' << dim << '\n';
        out << "levels " << in.min_level << ' ' << in.max_level << '\n';
        out << "config " << in.graduation_width << ' ' << in.max_stencil_radius << '\n';
        out << "geometry " << int(in.domain_is_box);
        for (const bool p : in.periodic)
        {
            out << ' ' << int(p);
        }
        out << '\n';
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
        if (partition)
        {
            write_ranks(out, "candidates", in.candidates);
            write_ranks(out, "neighbours", in.neighbours);
        }
    }

    template <std::size_t dim>
    void write_site_inputs(const SiteInputs<dim>& in, const std::filesystem::path& path)
    {
        std::ofstream out(path);
        write_block(out, in, false);
        if (!out)
        {
            throw std::runtime_error("cannot write " + path.string());
        }
    }

    template <std::size_t dim>
    void write_partition(const std::vector<SiteInputs<dim>>& ranks, const std::filesystem::path& path)
    {
        std::ofstream out(path);
        out << partition_magic << ' ' << inputs_format_version << ' ' << dim << ' ' << ranks.size() << '\n';
        for (const auto& in : ranks)
        {
            write_block(out, in, true);
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

    // Reads the header of a block, from the word after its magic word,
    // which the caller read into `magic`.
    template <std::size_t dim, class Fail>
    void read_header(std::istream& file, const std::string& magic, SiteInputs<dim>& in, const Fail& fail)
    {
        int version   = 0;
        std::size_t d = 0;
        std::string levels_key;
        std::string config_key;
        std::string geometry_key;
        file >> version >> d >> levels_key >> in.min_level >> in.max_level >> config_key >> in.graduation_width >> in.max_stencil_radius
            >> geometry_key >> in.domain_is_box;
        for (auto&& p : in.periodic)
        {
            bool periodic = false;
            file >> periodic;
            p = periodic;
        }
        if (magic != inputs_magic || version != inputs_format_version || d != dim || levels_key != "levels" || config_key != "config"
            || geometry_key != "geometry")
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

    // Reads the record "ranks <name> <n>" from <n> on: the n ranks.
    inline void read_ranks(std::istream& file, std::vector<std::size_t>& ranks)
    {
        std::size_t n = 0;
        file >> n;
        ranks.resize(n);
        for (auto& r : ranks)
        {
            file >> r;
        }
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
        else if (key == "ranks" && name == "candidates")
        {
            read_ranks(file, in.candidates);
        }
        else if (key == "ranks" && name == "neighbours")
        {
            read_ranks(file, in.neighbours);
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

    // Reads one block of records, its first line included, up to the end of
    // the file or the first line of the next block, whose first word it
    // leaves in `next`.
    template <std::size_t dim, class Fail>
    SiteInputs<dim> read_block(std::istream& file, std::string& next, const Fail& fail)
    {
        std::string magic = next;
        if (magic.empty())
        {
            file >> magic;
        }
        next.clear();
        SiteInputs<dim> in;
        read_header(file, magic, in, fail);

        std::string key;
        std::string name;
        while (file >> key)
        {
            if (key == inputs_magic)
            {
                next = key;
                break;
            }
            file >> name;
            read_record(file, key, name, in, fail);
        }
        set_domain_bounds(in);
        return in;
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
        auto file = open_inputs(path);
        std::string next;
        auto in = read_block<dim>(file, next, read_failure(path));
        if (!next.empty())
        {
            throw read_failure(path)("more than one mesh");
        }
        return in;
    }

    template <std::size_t dim>
    std::vector<SiteInputs<dim>> read_partition(const std::filesystem::path& path)
    {
        auto file       = open_inputs(path);
        const auto fail = read_failure(path);
        std::string magic;
        int version         = 0;
        std::size_t d       = 0;
        std::size_t n_ranks = 0;
        file >> magic >> version >> d >> n_ranks;
        if (magic != partition_magic || version != inputs_format_version || d != dim)
        {
            throw fail("not a set algebra partition file of version " + std::to_string(inputs_format_version) + " in " + std::to_string(dim)
                       + "D");
        }
        std::vector<SiteInputs<dim>> ranks;
        std::string next;
        for (std::size_t r = 0; r < n_ranks; ++r)
        {
            ranks.push_back(read_block<dim>(file, next, fail));
        }
        if (!next.empty())
        {
            throw fail("more ranks than announced");
        }
        return ranks;
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

    // The meshes of the ranks of a partition, read once per binary. Reports
    // the whole mesh, the number of ranks and of neighbour pairs (counted
    // from each side).
    template <std::size_t dim>
    const std::vector<SiteInputs<dim>>& partition_inputs(benchmark::State& state)
    {
        static std::map<mesh_key_t, std::vector<SiteInputs<dim>>> cache;
        const auto c = partition_case(state);
        auto it      = cache.find(c.key());
        if (it == cache.end())
        {
            try
            {
                it = cache.emplace(c.key(), read_partition<dim>(partition_file<dim>(c))).first;
            }
            catch (const std::exception& e)
            {
                std::cerr << e.what() << std::endl;
                std::exit(EXIT_FAILURE); // NOLINT(concurrency-mt-unsafe)
            }
        }
        std::size_t leaves    = 0;
        std::size_t intervals = 0;
        std::size_t pairs     = 0;
        for (const auto& in : it->second)
        {
            leaves += in.cells.nb_cells();
            bench::for_each_interval(in.cells,
                                     [&](std::size_t, const auto&, const auto&)
                                     {
                                         ++intervals;
                                     });
            pairs += in.neighbours.size();
        }
        state.counters["leaves"]     = static_cast<double>(leaves);
        state.counters["leaf_itv"]   = static_cast<double>(intervals);
        state.counters["ranks"]      = static_cast<double>(it->second.size());
        state.counters["neighbours"] = static_cast<double>(pairs);
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
        out.domain_is_box      = in.domain_is_box;
        out.periodic           = in.periodic;
        out.domain_min         = in.domain_min;
        out.domain_max         = in.domain_max;
        out.candidates         = in.candidates;
        out.neighbours         = in.neighbours;
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

    // partition_inputs() for backend B.
    template <class B, std::size_t dim>
    const std::vector<backend_inputs_t<B, dim>>& backend_partition(benchmark::State& state)
    {
        const auto& ranks = partition_inputs<dim>(state);
        if constexpr (std::is_same_v<backend_inputs_t<B, dim>, SiteInputs<dim>>)
        {
            return ranks;
        }
        else
        {
            static std::map<const std::vector<SiteInputs<dim>>*, std::vector<backend_inputs_t<B, dim>>> cache;
            auto it = cache.find(&ranks);
            if (it == cache.end())
            {
                std::vector<backend_inputs_t<B, dim>> copies;
                for (const auto& in : ranks)
                {
                    copies.push_back(to_backend<B>(in));
                }
                it = cache.emplace(&ranks, std::move(copies)).first;
            }
            return it->second;
        }
    }
} // namespace bench
