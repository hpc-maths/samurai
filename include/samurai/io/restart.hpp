// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <filesystem>
namespace fs = std::filesystem;

#define H5_USE_XTENSOR

#include <highfive/H5Easy.hpp>
#include <highfive/H5PropertyList.hpp>

#include "../cell_array.hpp"
#include "../interval.hpp"
#include "../level_cell_array.hpp"
#include "../mesh.hpp"
#include "../uniform_mesh.hpp"

namespace HighFive
{
    template <class value_t, class index_t>
    class AtomicType<samurai::Interval<value_t, index_t>> : public DataType
    {
      public:

        AtomicType()
            : DataType(CompoundType(
                  {
                      {"start", create_datatype<value_t>()},
                      {"end", create_datatype<value_t>(), sizeof(value_t)},
                      {"step", create_datatype<value_t>(), 2 * sizeof(value_t)},
                      {"index", create_datatype<index_t>(), 4 * sizeof(value_t)}
        },
                  sizeof(samurai::Interval<value_t, index_t>)))
        {
        }
    };
}

namespace samurai
{
    template <std::size_t dim, class interval_t>
    void dump_mesh(HighFive::File& file, const LevelCellArray<dim, interval_t>& lca)
    {
        H5Easy::dump(file, "/mesh/dim", dim);
        H5Easy::dump(file, "/mesh/min_level", lca.level());
        H5Easy::dump(file, "/mesh/max_level", lca.level());
        H5Easy::dump(file, "/mesh/origin_point", lca.origin_point());
        H5Easy::dump(file, "/mesh/scaling_factor", lca.scaling_factor());

        for (std::size_t d = 0; d < dim; ++d)
        {
            H5Easy::dump(file, fmt::format("/mesh/level/{}/intervals/{}", lca.level(), d), lca[d]);
        }

        for (std::size_t d = 1; d < dim; ++d)
        {
            H5Easy::dump(file, fmt::format("/mesh/level/{}/offsets/{}", lca.level(), d), lca.offsets(d));
        }
    }

    template <std::size_t dim, class interval_t, std::size_t max_size>
    void dump_mesh(HighFive::File& file, const CellArray<dim, interval_t, max_size>& ca)
    {
        std::size_t min_level = ca.min_level();
        std::size_t max_level = ca.max_level();

        H5Easy::dump(file, "/mesh/dim", dim);
        H5Easy::dump(file, "/mesh/min_level", min_level);
        H5Easy::dump(file, "/mesh/max_level", max_level);
        H5Easy::dump(file, "/mesh/origin_point", ca.origin_point());
        H5Easy::dump(file, "/mesh/scaling_factor", ca.scaling_factor());

        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            for (std::size_t d = 0; d < dim; ++d)
            {
                H5Easy::dump(file, fmt::format("/mesh/level/{}/intervals/{}", level, d), ca[level][d]);
            }

            for (std::size_t d = 1; d < dim; ++d)
            {
                H5Easy::dump(file, fmt::format("/mesh/level/{}/offsets/{}", level, d), ca[level].offsets(d));
            }
        }
    }

    template <class Config>
    void dump_mesh(HighFive::File& file, const UniformMesh<Config>& mesh)
    {
        using Mesh      = UniformMesh<Config>;
        using mesh_id_t = typename Mesh::mesh_id_t;
        dump_mesh(file, mesh[mesh_id_t::cells]);
    }

    template <class D, class Config>
    void dump_mesh(HighFive::File& file, const Mesh_base<D, Config>& mesh)
    {
        using Mesh      = Mesh_base<D, Config>;
        using mesh_id_t = typename Mesh::mesh_id_t;
        dump_mesh(file, mesh[mesh_id_t::cells]);
        H5Easy::dump(file, "/mesh/min_level", mesh.min_level(), H5Easy::DumpMode::Overwrite);
        H5Easy::dump(file, "/mesh/max_level", mesh.max_level(), H5Easy::DumpMode::Overwrite);
    }

    template <class Mesh>
    void dump_mesh(const fs::path& path, const std::string& filename, const Mesh& mesh)
    {
        HighFive::File file(fmt::format("{}.h5", (path / filename).string()), HighFive::File::Overwrite);
        dump_mesh(file, mesh);
    }

    template <class Mesh>
    void dump_mesh(const std::string& filename, const Mesh& mesh)
    {
        dump_mesh(fs::current_path(), filename, mesh);
    }

    template <std::size_t dim_, class interval_t>
    void load_mesh(const HighFive::File& file, LevelCellArray<dim_, interval_t>& lca)
    {
        using lca_type = LevelCellArray<dim_, interval_t>;

        auto dim = H5Easy::load<std::size_t>(file, "/mesh/dim");
        if (dim != dim_)
        {
            throw std::runtime_error(
                fmt::format("The dimension of the mesh is not the same as the one of the mesh to be loaded. {} != {}", dim, dim_));
        }

        auto min_level = H5Easy::load<std::size_t>(file, "/mesh/min_level");
        auto max_level = H5Easy::load<std::size_t>(file, "/mesh/max_level");
        if (min_level != max_level)
        {
            throw std::runtime_error(
                fmt::format("The mesh to be loaded is not a LevelCellArray. min_level != max_level. {} != {}", min_level, max_level));
        }

        lca = {min_level};

        auto origin_point   = H5Easy::load<typename lca_type::coords_t>(file, "/mesh/origin_point");
        auto scaling_factor = H5Easy::load<double>(file, "/mesh/scaling_factor");

        lca.set_origin_point(origin_point);
        lca.set_scaling_factor(scaling_factor);

        if (!file.exist(fmt::format("/mesh/level/{}", min_level)))
        {
            throw std::runtime_error(fmt::format("The mesh to be loaded does not contain the level {}.", min_level));
        }

        for (std::size_t d = 0; d < dim; ++d)
        {
            lca[d] = H5Easy::load<std::vector<interval_t>>(file, fmt::format("/mesh/level/{}/intervals/{}", min_level, d));
        }
        for (std::size_t d = 1; d < dim; ++d)
        {
            lca.offsets(d) = H5Easy::load<std::vector<std::size_t>>(file, fmt::format("/mesh/level/{}/offsets/{}", min_level, d));
        }
    }

    template <std::size_t dim_, class interval_t, std::size_t max_size>
    void load_mesh(const HighFive::File& file, CellArray<dim_, interval_t, max_size>& ca)
    {
        using ca_type = CellArray<dim_, interval_t, max_size>;

        auto dim = H5Easy::load<std::size_t>(file, "/mesh/dim");
        if (dim != dim_)
        {
            throw std::runtime_error(
                fmt::format("The dimension of the mesh is not the same as the one of the mesh to be loaded. {} != {}", dim, dim_));
        }

        auto min_level      = H5Easy::load<std::size_t>(file, "/mesh/min_level");
        auto max_level      = H5Easy::load<std::size_t>(file, "/mesh/max_level");
        auto origin_point   = H5Easy::load<typename ca_type::coords_t>(file, "/mesh/origin_point");
        auto scaling_factor = H5Easy::load<double>(file, "/mesh/scaling_factor");

        ca.clear();
        ca.set_origin_point(origin_point);
        ca.set_scaling_factor(scaling_factor);

        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            if (file.exist(fmt::format("/mesh/level/{}", level)))
            {
                for (std::size_t d = 0; d < dim; ++d)
                {
                    ca[level][d] = H5Easy::load<std::vector<interval_t>>(file, fmt::format("/mesh/level/{}/intervals/{}", level, d));
                }
                for (std::size_t d = 1; d < dim; ++d)
                {
                    ca[level].offsets(d) = H5Easy::load<std::vector<std::size_t>>(file, fmt::format("/mesh/level/{}/offsets/{}", level, d));
                }
            }
        }
    }

    template <std::size_t dim, class interval_t, class lca_t = LevelCellArray<dim, interval_t>>
    void load_mesh(const HighFive::File& file, lca_t& lca)
    {
        lca = load_mesh<lca_t>(file);
    }

    template <std::size_t dim, class interval_t, std::size_t max_size, class ca_t = CellArray<dim, interval_t, max_size>>
    void load_mesh(const HighFive::File& file, ca_t& ca)
    {
        ca = load_mesh<ca_t>(file);
    }

    template <class Config>
    void load_mesh(const HighFive::File& file, UniformMesh<Config>& mesh)
    {
        using ca_type = typename UniformMesh<Config>::ca_type;

        ca_type ca;
        load_mesh(file, ca);
        mesh = {ca};
    }

    template <class Mesh>
    void load_mesh(const HighFive::File& file, Mesh& mesh)
    {
        using ca_type  = typename Mesh::ca_type;
        auto min_level = H5Easy::load<std::size_t>(file, "/mesh/min_level");
        auto max_level = H5Easy::load<std::size_t>(file, "/mesh/max_level");

        ca_type ca;
        load_mesh(file, ca);
        mesh = {ca, min_level, max_level};
    }

    template <class Mesh>
    void load_mesh(const fs::path& path, const std::string& filename, Mesh& mesh)
    {
        HighFive::File file(fmt::format("{}.h5", (path / filename).string()), HighFive::File::ReadOnly);
        load_mesh(file, mesh);
    }

    template <class Mesh>
    void load_mesh(const std::string& filename, Mesh& mesh)
    {
        load_mesh(fs::current_path(), filename, mesh);
    }
}