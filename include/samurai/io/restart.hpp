// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <filesystem>
namespace fs = std::filesystem;

#define H5_USE_XTENSOR

#include <highfive/H5Easy.hpp>
#include <highfive/H5PropertyList.hpp>

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
namespace mpi = boost::mpi;
#endif

#include "../cell_array.hpp"
#include "../interval.hpp"
#include "../level_cell_array.hpp"
#include "../mesh.hpp"
#include "../uniform_mesh.hpp"
#include "util.hpp"

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

    template <class T>
    void dump(HighFive::File& file, const std::string& name, const std::vector<T>& data)
    {
        auto xfer_props = HighFive::DataTransferProps{};
#ifdef SAMURAI_WITH_MPI
        xfer_props.add(HighFive::UseCollectiveIO{});
        mpi::communicator world;
        auto rank = static_cast<std::size_t>(world.rank());
        auto size = static_cast<std::size_t>(world.size());

        std::vector<std::size_t> local_sizes(size);
        // Gather all sizes from all processes
        mpi::all_gather(world, data.size(), local_sizes);
#else
        std::size_t rank = 0;
        std::size_t size = 1;

        std::vector<std::size_t> local_sizes(size, data.size());
#endif

        // Calculate cumulative sizes
        std::vector<std::size_t> cumulative_sizes(local_sizes.size() + 1, 0);
        for (std::size_t i = 0; i < local_sizes.size(); ++i)
        {
            cumulative_sizes[i + 1] = cumulative_sizes[i] + local_sizes[i];
        }

        if (cumulative_sizes.back() == 0)
        {
            return;
        }

        H5Easy::dump(file, fmt::format("{}/partition", name), cumulative_sizes);
        auto dataset = file.createDataSet<T>(fmt::format("{}/data", name),
                                             HighFive::DataSpace(std::vector<std::size_t>{cumulative_sizes.back()}));

        auto dataset_slice = dataset.select({cumulative_sizes[rank]}, {data.size()});
        dataset_slice.write_raw(data.data(), HighFive::AtomicType<T>{}, xfer_props);
    }

    template <std::size_t dim, class interval_t>
    void dump(HighFive::File& file, const LevelCellArray<dim, interval_t>& lca, bool with_metadata = true)
    {
        if (with_metadata)
        {
#ifdef SAMURAI_WITH_MPI
            mpi::communicator world;
            auto size = world.size();
#else
            std::size_t size = 1;
#endif
            H5Easy::dump(file, "/n_process", size);
            H5Easy::dump(file, "/mesh/dim", dim);
            H5Easy::dump(file, "/mesh/min_level", lca.level());
            H5Easy::dump(file, "/mesh/max_level", lca.level());
            H5Easy::dump(file, "/mesh/origin_point", lca.origin_point());
            H5Easy::dump(file, "/mesh/scaling_factor", lca.scaling_factor());
        }

        for (std::size_t d = 0; d < dim; ++d)
        {
            auto name = fmt::format("/mesh/level/{}/dim/{}/intervals", lca.level(), d);
            dump(file, name, lca[d]);
        }
        for (std::size_t d = 1; d < dim; ++d)
        {
            auto name = fmt::format("/mesh/level/{}/dim/{}/offsets", lca.level(), d);
            dump(file, name, lca.offsets(d));
        }
    }

    template <std::size_t dim, class interval_t, std::size_t max_size>
    void dump(HighFive::File& file, const CellArray<dim, interval_t, max_size>& ca)
    {
#ifdef SAMURAI_WITH_MPI
        mpi::communicator world;
        auto min_level = mpi::all_reduce(world, ca.min_level(), mpi::minimum<std::size_t>());
        auto max_level = mpi::all_reduce(world, ca.max_level(), mpi::maximum<std::size_t>());
        auto size      = world.size();
#else
        std::size_t min_level = ca.min_level();
        std::size_t max_level = ca.max_level();
        std::size_t size      = 1;
#endif

        H5Easy::dump(file, "/n_process", size);
        H5Easy::dump(file, "/mesh/dim", dim);
        H5Easy::dump(file, "/mesh/min_level", min_level);
        H5Easy::dump(file, "/mesh/max_level", max_level);
        H5Easy::dump(file, "/mesh/origin_point", ca.origin_point());
        H5Easy::dump(file, "/mesh/scaling_factor", ca.scaling_factor());

        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            dump(file, ca[level], false); // false to avoid dumping metadata for each level
        }
    }

    template <class Config>
    void dump(HighFive::File& file, const UniformMesh<Config>& mesh)
    {
        using Mesh      = UniformMesh<Config>;
        using mesh_id_t = typename Mesh::mesh_id_t;
        dump(file, mesh[mesh_id_t::cells]);
    }

    void dump_field(HighFive::File& file, const auto& mesh, const auto& field)
    {
        auto data = extract_data_as_vector(field, mesh);

        H5Easy::dump(file, fmt::format("/fields/{}/n_comp", field.name()), field.n_comp);

        dump(file, fmt::format("/fields/{}/data", field.name()), data);
    }

    template <class... Fields>
    void dump_fields(HighFive::File& file, const auto& mesh, const Fields&... fields)
    {
        if (sizeof...(Fields) > 0)
        {
            (dump_field(file, mesh, fields), ...);
        }
    }

    template <class D, class Config>
    void dump(HighFive::File& file, const Mesh_base<D, Config>& mesh, const auto&... fields)
    {
        using Mesh      = Mesh_base<D, Config>;
        using mesh_id_t = typename Mesh::mesh_id_t;
        dump(file, mesh[mesh_id_t::cells]);
        H5Easy::dump(file, "/mesh/min_level", mesh.min_level(), H5Easy::DumpMode::Overwrite);
        H5Easy::dump(file, "/mesh/max_level", mesh.max_level(), H5Easy::DumpMode::Overwrite);
        dump_fields(file, mesh[mesh_id_t::cells], fields...);
    }

    template <class Mesh, class... Fields>
    void dump(const fs::path& path, const std::string& filename, const Mesh& mesh, const Fields&... fields)
    {
        HighFive::FileAccessProps fapl;
#ifdef SAMURAI_WITH_MPI
        fapl.add(HighFive::MPIOFileAccess{MPI_COMM_WORLD, MPI_INFO_NULL});
        fapl.add(HighFive::MPIOCollectiveMetadata{});
#endif
        HighFive::File file(fmt::format("{}.h5", (path / filename).string()), HighFive::File::Overwrite, fapl);
        dump(file, mesh, fields...);
    }

    template <class Mesh, class... Fields>
    void dump(const std::string& filename, const Mesh& mesh, const Fields&... fields)
    {
        dump(fs::current_path(), filename, mesh, fields...);
    }

    template <class T>
    auto load(const HighFive::File& file, const std::string& name)
    {
        auto xfer_props = HighFive::DataTransferProps{};
#ifdef SAMURAI_WITH_MPI
        xfer_props.add(HighFive::UseCollectiveIO{});
        mpi::communicator world;
        auto rank = static_cast<std::size_t>(world.rank());
#else
        std::size_t rank = 0;
#endif
        auto partition = H5Easy::load<std::vector<std::size_t>>(file, fmt::format("{}/partition", name));

        auto dataset = file.getDataSet(fmt::format("{}/data", name));

        auto dataset_slice = dataset.select({partition[rank]}, {partition[rank + 1] - partition[rank]});
        T output(partition[rank + 1] - partition[rank]);
        dataset_slice.read_raw(output.data(), xfer_props);
        return output;
    }

    template <std::size_t dim_, class interval_t>
    void load(const HighFive::File& file, LevelCellArray<dim_, interval_t>& lca)
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
            lca[d] = load<std::vector<interval_t>>(file, fmt::format("/mesh/level/{}/dim/{}/intervals", min_level, d));
        }
        for (std::size_t d = 1; d < dim; ++d)
        {
            lca.offsets(d) = load<std::vector<std::size_t>>(file, fmt::format("/mesh/level/{}/dim/{}/offsets", min_level, d));
        }
    }

    template <std::size_t dim_, class interval_t, std::size_t max_size>
    void load(const HighFive::File& file, CellArray<dim_, interval_t, max_size>& ca)
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
                    ca[level][d] = load<std::vector<interval_t>>(file, fmt::format("/mesh/level/{}/dim/{}/intervals", level, d));
                }
                for (std::size_t d = 1; d < dim; ++d)
                {
                    ca[level].offsets(d) = load<std::vector<std::size_t>>(file, fmt::format("/mesh/level/{}/dim/{}/offsets", level, d));
                }
            }
        }
    }

    void load_field(const HighFive::File& file, const auto& mesh, auto& field)
    {
        using Field     = std::decay_t<decltype(field)>;
        using size_type = typename Field::size_type;

        if (field.name().empty())
        {
            throw std::runtime_error("The field has no name.");
        }

        if (!file.exist(fmt::format("/fields/{}", field.name())))
        {
            throw std::runtime_error(fmt::format("The field {} does not exist in the file.", field.name()));
        }

        auto n_comp = H5Easy::load<std::size_t>(file, fmt::format("/fields/{}/n_comp", field.name()));
        if (n_comp != Field::n_comp)
        {
            throw std::runtime_error(
                fmt::format("The number of components of the field ({}) does not match the expected number of components ({}).",
                            n_comp,
                            Field::n_comp));
        }

        using data_t = std::vector<typename Field::value_type>;

        auto data = load<data_t>(file, fmt::format("/fields/{}/data", field.name()));

        field.resize();

        std::size_t index = 0;
        for_each_cell(mesh,
                      [&](auto cell)
                      {
                          if constexpr (Field::n_comp == 1)
                          {
                              field[cell] = data[index++];
                          }
                          else
                          {
                              for (size_type i = 0; i < field.n_comp; ++i)
                              {
                                  field[cell][i] = data[index + i];
                              }
                              index += field.n_comp;
                          }
                      });
    }

    template <class... Fields>
    void load_fields(const HighFive::File& file, auto& mesh, Fields&... fields)
    {
        if (sizeof...(Fields) > 0)
        {
            (load_field(file, mesh, fields), ...);
        }
    }

    template <std::size_t dim, class interval_t, class lca_t = LevelCellArray<dim, interval_t>>
    void load(const HighFive::File& file, lca_t& lca)
    {
        lca = load<lca_t>(file);
    }

    template <std::size_t dim, class interval_t, std::size_t max_size, class ca_t = CellArray<dim, interval_t, max_size>>
    void load(const HighFive::File& file, ca_t& ca)
    {
        ca = load<ca_t>(file);
    }

    template <class Config, class... Fields>
    void load(const HighFive::File& file, UniformMesh<Config>& mesh, Fields&... fields)
    {
        using ca_type = typename UniformMesh<Config>::ca_type;

        ca_type ca;
        load(file, ca);
        UniformMesh<Config> new_mesh{ca};
        std::swap(mesh, new_mesh);
        load_fields(file, mesh, fields...);
    }

    template <class Mesh, class... Fields>
    void load(const HighFive::File& file, Mesh& mesh, Fields&... fields)
    {
        using ca_type = typename Mesh::ca_type;
#ifdef SAMURAI_WITH_MPI
        mpi::communicator world;
        auto size = static_cast<std::size_t>(world.size());
#else
        std::size_t size = 1;
#endif

        auto n_process = H5Easy::load<std::size_t>(file, "n_process");
        if (n_process != size)
        {
            throw std::runtime_error(
                fmt::format("The number of processes in the restart file ({}) does not match the current number of processes ({}).",
                            n_process,
                            size));
        }

        auto min_level = H5Easy::load<std::size_t>(file, "/mesh/min_level");
        auto max_level = H5Easy::load<std::size_t>(file, "/mesh/max_level");

        ca_type ca;
        load(file, ca);
        auto mesh_cfg = mesh_config<Mesh::dim>().min_level(min_level).max_level(max_level);
        Mesh new_mesh{mesh_cfg, ca};
        std::swap(mesh, new_mesh);
        load_fields(file, mesh, fields...);
    }

    template <class Mesh, class... Fields>
    void load(const fs::path& path, const std::string& filename, Mesh& mesh, Fields&... fields)
    {
        HighFive::FileAccessProps fapl;
#ifdef SAMURAI_WITH_MPI
        fapl.add(HighFive::MPIOFileAccess{MPI_COMM_WORLD, MPI_INFO_NULL});
        fapl.add(HighFive::MPIOCollectiveMetadata{});
#endif
        HighFive::File file(fmt::format("{}.h5", (path / filename).string()), HighFive::File::ReadOnly, fapl);
        load(file, mesh, fields...);
    }

    template <class Mesh, class... Fields>
    void load(const std::string& filename, Mesh& mesh, Fields&... fields)
    {
        load(fs::current_path(), filename, mesh, fields...);
    }
}
