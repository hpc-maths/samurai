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
//   ca <name> <number of intervals>, then per interval: level start end [y [z]]
//   lca corner <level> <number of intervals>, then per interval: start end [y [z]]
// Each interval is the row [start, end) of cells at (y, z) of its level.

#pragma once

#include <array>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include <samurai/algorithm.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/level_cell_array.hpp>
#include <samurai/level_cell_list.hpp>
#include <samurai/stencil.hpp>

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

    // (max_level, number of balls) of each mesh: the mesh is refined to
    // max_level along the surfaces of the balls, so more balls means more,
    // smaller fronts.
    template <std::size_t dim>
    std::vector<std::array<long, 2>> mesh_cases()
    {
        const long max_level = static_cast<long>(default_max_level<dim>());
        if constexpr (dim == 1)
        {
            // a 1D front is a point: many of them to get a non-trivial mesh
            return {
                {max_level, 16 },
                {max_level, 256}
            };
        }
        else if constexpr (dim == 2)
        {
            return {
                {max_level,     1 },
                {max_level,     16},
                {max_level + 2, 16}
            };
        }
        else
        {
            return {
                {max_level, 1},
                {max_level, 8}
            };
        }
    }

    template <std::size_t dim>
    void mesh_args(benchmark::internal::Benchmark* b)
    {
        for (const auto& args : mesh_cases<dim>())
        {
            b->Args({args[0], args[1]});
        }
        b->Unit(benchmark::kMicrosecond);
    }

    // So that every report says what it was measured on.
    template <std::size_t dim>
    void report_mesh(benchmark::State& state, const samurai::CellArray<dim>& cells)
    {
        std::size_t intervals = 0;
        samurai::for_each_interval(cells,
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

    // Named after the MRMesh members they are copied from (make_site_inputs
    // in mr_fixtures.hpp), with accessors where the mesh has one.
    template <std::size_t dim>
    struct SiteInputs
    {
        using ca_t  = samurai::CellArray<dim>;
        using lca_t = typename ca_t::lca_type;

        // MRMesh::config_t::prediction_stencil_radius
        static constexpr int prediction_stencil_radius = 1;

        // mesh.min_level() and mesh.max_level(): the bounds of the mesh
        // configuration, not of the cells.
        std::size_t min_level = 0;
        std::size_t max_level = 0;

        ca_t cells;
        ca_t cells_and_ghosts;
        ca_t proj_cells;
        ca_t reference;
        ca_t union_; // mesh.get_union()
        ca_t domain_pyramid;
        ca_t subdomain_pyramid;
        // mesh.corner(d) for each d, in the order of for_each_diagonal_direction
        std::vector<lca_t> corners;

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

    template <std::size_t dim>
    bool operator==(const SiteInputs<dim>& a, const SiteInputs<dim>& b)
    {
        return a.min_level == b.min_level && a.max_level == b.max_level && a.cells == b.cells && a.cells_and_ghosts == b.cells_and_ghosts
            && a.proj_cells == b.proj_cells && a.reference == b.reference && a.union_ == b.union_ && a.domain_pyramid == b.domain_pyramid
            && a.subdomain_pyramid == b.subdomain_pyramid && a.corners == b.corners;
    }

    //////////////////////////////////////////////////////////////////////
    // Files
    //////////////////////////////////////////////////////////////////////

    inline constexpr int inputs_format_version = 1;

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
        };
    }

    // SAMURAI_SET_ALGEBRA_INPUTS in the environment overrides the directory
    // set at configure time, to share one set of files between builds.
    inline std::filesystem::path inputs_dir()
    {
        const char* dir = std::getenv("SAMURAI_SET_ALGEBRA_INPUTS"); // NOLINT(concurrency-mt-unsafe)
        return (dir != nullptr) ? dir : SAMURAI_SET_ALGEBRA_INPUTS_DIR;
    }

    template <std::size_t dim>
    std::filesystem::path inputs_file(std::size_t max_level, std::size_t nballs, bool moved)
    {
        return inputs_dir()
             / (std::to_string(dim) + "d_level" + std::to_string(max_level) + "_balls" + std::to_string(nballs) + (moved ? "_moved" : "")
                + ".txt");
    }

    // Set is a LevelCellArray or a CellArray.
    template <class Set>
    void write_intervals(std::ostream& out, const Set& set, bool with_level)
    {
        samurai::for_each_interval(set,
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

    template <std::size_t dim>
    void write_site_inputs(const SiteInputs<dim>& in, const std::filesystem::path& path)
    {
        std::ofstream out(path);
        out << "samurai-set-algebra-inputs " << inputs_format_version << ' ' << dim << '\n';
        out << "levels " << in.min_level << ' ' << in.max_level << '\n';
        for (const auto& [name, ca] : named_cell_arrays(in))
        {
            std::size_t n = 0;
            samurai::for_each_interval(*ca,
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

    template <std::size_t dim>
    SiteInputs<dim> read_site_inputs(const std::filesystem::path& path)
    {
        std::ifstream file(path);
        if (!file)
        {
            throw std::runtime_error("cannot open " + path.string() + ": run generate_set_algebra_inputs first");
        }
        const auto fail = [&path](const std::string& what)
        {
            return std::runtime_error(path.string() + ": " + what);
        };

        std::string magic;
        int version   = 0;
        std::size_t d = 0;
        std::string key;
        SiteInputs<dim> in;
        file >> magic >> version >> d >> key >> in.min_level >> in.max_level;
        if (magic != "samurai-set-algebra-inputs" || version != inputs_format_version || d != dim || key != "levels")
        {
            throw fail("not a set algebra inputs file of version " + std::to_string(inputs_format_version) + " in " + std::to_string(dim)
                       + "D");
        }

        const auto cell_arrays = named_cell_arrays(in);
        std::string name;
        std::size_t n = 0;
        while (file >> key >> name)
        {
            if (key == "ca")
            {
                file >> n;
                samurai::CellList<dim> cl;
                for (std::size_t k = 0; k < n; ++k)
                {
                    std::size_t level = 0;
                    file >> level;
                    read_interval(file, cl[level]);
                }
                bool found = false;
                for (const auto& [ca_name, ca] : cell_arrays)
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
            else if (key == "lca" && name == "corner")
            {
                std::size_t level = 0;
                file >> level >> n;
                samurai::LevelCellList<dim> lcl(level);
                for (std::size_t k = 0; k < n; ++k)
                {
                    read_interval(file, lcl);
                }
                in.corners.emplace_back(lcl);
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
        return in;
    }

    // Read once per binary. Reports the mesh it returns: a benchmark that
    // reads two meshes reports the last one.
    template <std::size_t dim>
    const SiteInputs<dim>& site_inputs(benchmark::State& state, bool moved = false)
    {
        static std::map<std::tuple<std::size_t, std::size_t, bool>, SiteInputs<dim>> cache;
        const auto max_level = static_cast<std::size_t>(state.range(0));
        const auto nballs    = static_cast<std::size_t>(state.range(1));
        auto it              = cache.find({max_level, nballs, moved});
        if (it == cache.end())
        {
            try
            {
                it = cache.emplace(std::tuple{max_level, nballs, moved}, read_site_inputs<dim>(inputs_file<dim>(max_level, nballs, moved))).first;
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
} // namespace bench
