// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <concepts>
#include <filesystem>
#include <utility>
namespace fs = std::filesystem;

#include <highfive/H5Easy.hpp>
#include <highfive/H5PropertyList.hpp>

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
namespace mpi = boost::mpi;
#endif

#include "../cell_array.hpp"
#include "../concepts.hpp"
#include "../interval.hpp"
#include "../level_cell_array.hpp"
#include "../load_balancing/load_balancer.hpp"
#include "../load_balancing/strategies/sfc.hpp"
#include "../load_balancing/weight.hpp"
#include "../mesh.hpp"
#include "../uniform_mesh.hpp"
#include "metadata.hpp"
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
            // HighFive v3 does not support xfixed_container so we have to dump from an xtensor_container first
            H5Easy::dump(file, "/mesh/origin_point", xt::xtensor<double, 1>{lca.origin_point()});
            H5Easy::dump(file, "/mesh/scaling_factor", lca.scaling_factor());
            H5Easy::dump(file, "/mesh/is_box", lca.is_box());
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
        // HighFive v3 does not support xfixed_container so we have to dump from an xtensor_container first
        H5Easy::dump(file, "/mesh/origin_point", xt::xtensor<double, 1>{ca.origin_point()});
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
        // The domain is rebuilt from the cells on load, which cannot tell whether it is a box.
        H5Easy::dump(file, "/mesh/is_box", mesh.domain().is_box());
        dump_fields(file, mesh[mesh_id_t::cells], fields...);
    }

    template <class Mesh, class... Fields>
        requires mesh_like<Mesh>
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

    // Same as above, with a user callback to write arbitrary metadata (time,
    // simulation parameters, ...) into the checkpoint file. The callback is a
    // template parameter invoked in place: no std::function, no allocation.
    template <class Callback, class Mesh, class... Fields>
        requires mesh_like<Mesh> && std::invocable<Callback&&, MetadataWriter&>
    void dump(const fs::path& path, const std::string& filename, Callback&& metadata, const Mesh& mesh, const Fields&... fields)
    {
        HighFive::FileAccessProps fapl;
#ifdef SAMURAI_WITH_MPI
        fapl.add(HighFive::MPIOFileAccess{MPI_COMM_WORLD, MPI_INFO_NULL});
        fapl.add(HighFive::MPIOCollectiveMetadata{});
#endif
        HighFive::File file(fmt::format("{}.h5", (path / filename).string()), HighFive::File::Overwrite, fapl);
        dump(file, mesh, fields...);
        MetadataWriter writer(file); // no XDMF for a checkpoint: HDF5 only
        // NOTE: Under MPI, this callback must be executed collectively on all ranks.
        std::forward<Callback>(metadata)(writer);
    }

    template <class Mesh, class... Fields>
        requires mesh_like<Mesh>
    void dump(const std::string& filename, const Mesh& mesh, const Fields&... fields)
    {
        dump(fs::current_path(), filename, mesh, fields...);
    }

    template <class Callback, class Mesh, class... Fields>
        requires mesh_like<Mesh> && std::invocable<Callback&, MetadataWriter&>
    void dump(const std::string& filename, Callback&& metadata, const Mesh& mesh, const Fields&... fields)
    {
        dump(fs::current_path(), filename, std::forward<Callback>(metadata), mesh, fields...);
    }

    namespace detail
    {
        // Every dataset of a restart file is one array cut into contiguous
        // partitions, one per process that wrote it; "<name>/partition" holds the
        // n_process + 1 bounds of these partitions. A process reading the file
        // takes the contiguous block [first, last) of partitions, so the number
        // of processes that read a file does not have to match the number that
        // wrote it. With as many readers as writers, each reader gets the
        // partition written by the process of the same rank.
        struct PartitionRange
        {
            std::size_t first;
            std::size_t last;
        };

        inline void check_mesh_dim(const HighFive::File& file, std::size_t expected)
        {
            auto dim = H5Easy::load<std::size_t>(file, "/mesh/dim");
            if (dim != expected)
            {
                throw std::runtime_error(
                    fmt::format("The dimension of the mesh is not the same as the one of the mesh to be loaded. {} != {}", dim, expected));
            }
        }

        template <class array_t>
        void load_geometry(const HighFive::File& file, array_t& array)
        {
            // HighFive v3 does not support xfixed_container so we have to load into an xtensor_container first
            array.set_origin_point(H5Easy::load<xt::xtensor<double, 1>>(file, "/mesh/origin_point"));
            array.set_scaling_factor(H5Easy::load<double>(file, "/mesh/scaling_factor"));
        }

        inline PartitionRange partitions_to_read(const HighFive::File& file)
        {
#ifdef SAMURAI_WITH_MPI
            mpi::communicator world;
            auto rank = static_cast<std::size_t>(world.rank());
            auto size = static_cast<std::size_t>(world.size());
#else
            std::size_t rank = 0;
            std::size_t size = 1;
#endif
            auto n_process = H5Easy::load<std::size_t>(file, "/n_process");
            return {rank * n_process / size, (rank + 1) * n_process / size};
        }

        // Reads the partitions of `range` of the dataset `name`, and returns one
        // vector per partition.
        // NOTE: Under MPI, collective: every process must call it for the same
        // datasets in the same order, even when its range is empty.
        template <class T>
        std::vector<std::vector<T>> load_partitions(const HighFive::File& file, const std::string& name, const PartitionRange& range)
        {
            auto xfer_props = HighFive::DataTransferProps{};
#ifdef SAMURAI_WITH_MPI
            xfer_props.add(HighFive::UseCollectiveIO{});
#endif
            auto bounds = H5Easy::load<std::vector<std::size_t>>(file, fmt::format("{}/partition", name));
            if (bounds.size() != H5Easy::load<std::size_t>(file, "/n_process") + 1)
            {
                throw std::runtime_error(
                    fmt::format("The partition of {} does not match the number of processes stored in the restart file.", name));
            }

            // The partitions of the range are contiguous in the file: read them in one call.
            auto dataset = file.getDataSet(fmt::format("{}/data", name));
            std::vector<T> data(bounds[range.last] - bounds[range.first]);
            dataset.select({bounds[range.first]}, {data.size()}).read_raw(data.data(), xfer_props);

            std::vector<std::vector<T>> partitions;
            partitions.reserve(range.last - range.first);
            for (std::size_t p = range.first; p < range.last; ++p)
            {
                auto begin = data.begin() + static_cast<std::ptrdiff_t>(bounds[p] - bounds[range.first]);
                auto end   = data.begin() + static_cast<std::ptrdiff_t>(bounds[p + 1] - bounds[range.first]);
                partitions.emplace_back(begin, end);
            }
            return partitions;
        }

        // Reads the cells of `level` written by each partition of `range`, in
        // the LevelCellArray layout of the process that wrote them. A level with
        // no cell in the whole file has no group, and gives empty arrays.
        template <class lca_t>
        std::vector<lca_t> load_level_partitions(const HighFive::File& file, std::size_t level, const PartitionRange& range)
        {
            constexpr std::size_t dim = lca_t::dim;
            using interval_t          = typename lca_t::interval_t;

            std::vector<lca_t> partitions(range.last - range.first, lca_t{level});
            if (!file.exist(fmt::format("/mesh/level/{}", level)))
            {
                return partitions;
            }

            for (std::size_t d = 0; d < dim; ++d)
            {
                auto intervals = load_partitions<interval_t>(file, fmt::format("/mesh/level/{}/dim/{}/intervals", level, d), range);
                for (std::size_t p = 0; p < partitions.size(); ++p)
                {
                    partitions[p][d] = std::move(intervals[p]);
                }
            }
            for (std::size_t d = 1; d < dim; ++d)
            {
                auto offsets = load_partitions<std::size_t>(file, fmt::format("/mesh/level/{}/dim/{}/offsets", level, d), range);
                for (std::size_t p = 0; p < partitions.size(); ++p)
                {
                    partitions[p].offsets(d) = std::move(offsets[p]);
                }
            }
            return partitions;
        }

        // Reads the cells written by each partition of `range`, in the
        // CellArray layout of the process that wrote them. The field values of
        // a partition are stored in the cell order of its CellArray.
        template <class ca_t>
        std::vector<ca_t> load_cell_array_partitions(const HighFive::File& file, const PartitionRange& range)
        {
            using lca_t = typename ca_t::lca_type;

            check_mesh_dim(file, ca_t::dim);

            auto min_level = H5Easy::load<std::size_t>(file, "/mesh/min_level");
            auto max_level = H5Easy::load<std::size_t>(file, "/mesh/max_level");

            std::vector<ca_t> partitions(range.last - range.first);
            for (std::size_t level = min_level; level <= max_level; ++level)
            {
                auto lcas = load_level_partitions<lca_t>(file, level, range);
                for (std::size_t p = 0; p < partitions.size(); ++p)
                {
                    partitions[p][level] = std::move(lcas[p]);
                }
            }
            for (auto& ca : partitions)
            {
                ca.update_index();
            }
            return partitions;
        }

        // Union of the cells of several partitions. A single partition is
        // returned as is, so that a file read with the number of processes that
        // wrote it gives back the very same arrays.
        template <std::size_t dim, class interval_t>
        LevelCellArray<dim, interval_t> merge_partitions(const std::vector<LevelCellArray<dim, interval_t>>& partitions, std::size_t level)
        {
            if (partitions.size() == 1)
            {
                return partitions.front();
            }
            LevelCellList<dim, interval_t> lcl{level};
            for (const auto& partition : partitions)
            {
                for_each_interval(partition,
                                  [&](std::size_t, const auto& interval, const auto& index)
                                  {
                                      lcl[index].add_interval(interval);
                                  });
            }
            return {lcl};
        }

        template <std::size_t dim, class interval_t, std::size_t max_size>
        CellArray<dim, interval_t, max_size> merge_partitions(const std::vector<CellArray<dim, interval_t, max_size>>& partitions)
        {
            if (partitions.size() == 1)
            {
                return partitions.front();
            }
            CellList<dim, interval_t, max_size> cl;
            for (const auto& partition : partitions)
            {
                for_each_interval(partition,
                                  [&](std::size_t level, const auto& interval, const auto& index)
                                  {
                                      cl[level][index].add_interval(interval);
                                  });
            }
            return {cl};
        }
    }

    // Files written before "/mesh/is_box" existed are read as non-box domains.
    inline bool load_is_box(const HighFive::File& file)
    {
        return file.exist("/mesh/is_box") && H5Easy::load<bool>(file, "/mesh/is_box");
    }

    namespace detail
    {
        template <class Mesh, class interval_t, class index_t, std::size_t... Is>
        const auto&
        mesh_interval(const Mesh& mesh, std::size_t level, const interval_t& interval, const index_t& index, std::index_sequence<Is...>)
        {
            return mesh.get_interval(level, interval, index[Is]...);
        }

        // Fills `field` with the values of each partition of `range`. The values
        // of a partition follow the cell order of its CellArray, and land at
        // the place of the same cells in the mesh of the field.
        template <class Field, class ca_t>
        void load_field(const HighFive::File& file, const PartitionRange& range, const std::vector<ca_t>& partitions, Field& field)
        {
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

            auto data = load_partitions<typename Field::value_type>(file, fmt::format("/fields/{}/data", field.name()), range);

            field.resize();

            for (std::size_t p = 0; p < partitions.size(); ++p)
            {
                if (data[p].size() != partitions[p].nb_cells() * Field::n_comp)
                {
                    throw std::runtime_error(fmt::format("The field {} does not have one value per cell in the restart file.", field.name()));
                }

                std::size_t k = 0;
                for_each_interval(
                    partitions[p],
                    [&](std::size_t level, const auto& interval, const auto& index)
                    {
                        const auto& in_mesh = mesh_interval(field.mesh(), level, interval, index, std::make_index_sequence<ca_t::dim - 1>{});
                        for (auto i = interval.start; i < interval.end; ++i)
                        {
                            auto cell_index = static_cast<size_type>(in_mesh.index + i);
                            if constexpr (Field::is_scalar)
                            {
                                field[cell_index] = data[p][k++];
                            }
                            else
                            {
                                for (size_type c = 0; c < Field::n_comp; ++c)
                                {
                                    field[cell_index][c] = data[p][k++];
                                }
                            }
                        }
                    });
            }
        }

        template <class ca_t, class... Fields>
        void load_fields(const HighFive::File& file, const PartitionRange& range, const std::vector<ca_t>& partitions, Fields&... fields)
        {
            (load_field(file, range, partitions, fields), ...);
        }

#ifdef SAMURAI_WITH_MPI
        // Spreads the cells evenly over the processes with the SFC Hilbert
        // strategy, moving the values of the fields with them.
        template <class Mesh, class... Fields>
        void balance_after_load(Mesh& mesh, Fields&... fields)
        {
            namespace lb = load_balancing;

            auto balancer = lb::make_load_balancer<lb::SFC<lb::Hilbert>>();
            if constexpr (sizeof...(Fields) > 0)
            {
                balancer.load_balance(lb::weight::uniform(), fields...);
            }
            else
            {
                // The load balancer moves cells together with at least one field.
                auto cells = make_scalar_field<int>("cells", mesh);
                balancer.load_balance(lb::weight::uniform(), cells);
            }
        }
#endif
    }

    template <std::size_t dim, class interval_t>
    void load(const HighFive::File& file, LevelCellArray<dim, interval_t>& lca)
    {
        detail::check_mesh_dim(file, dim);

        auto min_level = H5Easy::load<std::size_t>(file, "/mesh/min_level");
        auto max_level = H5Easy::load<std::size_t>(file, "/mesh/max_level");
        if (min_level != max_level)
        {
            throw std::runtime_error(
                fmt::format("The mesh to be loaded is not a LevelCellArray. min_level != max_level. {} != {}", min_level, max_level));
        }

        if (!file.exist(fmt::format("/mesh/level/{}", min_level)))
        {
            throw std::runtime_error(fmt::format("The mesh to be loaded does not contain the level {}.", min_level));
        }

        auto partitions = detail::load_level_partitions<LevelCellArray<dim, interval_t>>(file, min_level, detail::partitions_to_read(file));
        lca             = detail::merge_partitions(partitions, min_level);
        detail::load_geometry(file, lca);
        if (load_is_box(file))
        {
            lca.box_like();
        }
    }

    template <std::size_t dim, class interval_t, std::size_t max_size>
    void load(const HighFive::File& file, CellArray<dim, interval_t, max_size>& ca)
    {
        auto partitions = detail::load_cell_array_partitions<CellArray<dim, interval_t, max_size>>(file, detail::partitions_to_read(file));
        ca              = detail::merge_partitions(partitions);
        detail::load_geometry(file, ca);
    }

    template <class Config, class... Fields>
    void load(const HighFive::File& file, UniformMesh<Config>& mesh, Fields&... fields)
    {
        // The cells of a uniform mesh are a single LevelCellArray.
        using ca_type = typename UniformMesh<Config>::ca_type;

        detail::check_mesh_dim(file, ca_type::dim);
        auto level      = H5Easy::load<std::size_t>(file, "/mesh/min_level");
        auto range      = detail::partitions_to_read(file);
        auto partitions = detail::load_level_partitions<ca_type>(file, level, range);
        auto ca         = detail::merge_partitions(partitions, level);
        detail::load_geometry(file, ca);

        UniformMesh<Config> new_mesh{ca};
        std::swap(mesh, new_mesh);
        detail::load_fields(file, range, partitions, fields...);
    }

    // The file can be read by any number of processes. Under MPI, when that
    // number differs from the number of processes that wrote the file, the
    // cells and the fields are then balanced over the processes.
    template <class Mesh, class... Fields>
    void load(const HighFive::File& file, Mesh& mesh, Fields&... fields)
    {
        using ca_type = typename Mesh::ca_type;

        auto min_level = H5Easy::load<std::size_t>(file, "/mesh/min_level");
        auto max_level = H5Easy::load<std::size_t>(file, "/mesh/max_level");

        auto range      = detail::partitions_to_read(file);
        auto partitions = detail::load_cell_array_partitions<ca_type>(file, range);
        auto ca         = detail::merge_partitions(partitions);
        detail::load_geometry(file, ca);

        auto mesh_cfg = mesh.cfg();
        mesh_cfg.min_level(min_level).max_level(max_level).disable_args_parse();
        Mesh new_mesh{ca, mesh_cfg};
        if (load_is_box(file))
        {
            new_mesh.box_like();
        }
        std::swap(mesh, new_mesh);
        detail::load_fields(file, range, partitions, fields...);

#ifdef SAMURAI_WITH_MPI
        mpi::communicator world;
        if (H5Easy::load<std::size_t>(file, "/n_process") != static_cast<std::size_t>(world.size()))
        {
            detail::balance_after_load(mesh, fields...);
        }
#endif
    }

    template <class Mesh, class... Fields>
        requires mesh_like<Mesh>
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

    // Same as above, with a user callback to read back arbitrary metadata
    // (time, simulation parameters, ...) previously written by dump.
    template <class Callback, class Mesh, class... Fields>
        requires mesh_like<Mesh> && std::invocable<Callback&&, MetadataReader&>
    void load(const fs::path& path, const std::string& filename, Callback&& metadata, Mesh& mesh, Fields&... fields)
    {
        HighFive::FileAccessProps fapl;
#ifdef SAMURAI_WITH_MPI
        fapl.add(HighFive::MPIOFileAccess{MPI_COMM_WORLD, MPI_INFO_NULL});
        fapl.add(HighFive::MPIOCollectiveMetadata{});
#endif
        HighFive::File file(fmt::format("{}.h5", (path / filename).string()), HighFive::File::ReadOnly, fapl);
        load(file, mesh, fields...);
        MetadataReader reader(file);
        // NOTE: Under MPI, this callback must be executed collectively on all ranks.
        std::forward<Callback>(metadata)(reader);
    }

    template <class Mesh, class... Fields>
        requires mesh_like<Mesh>
    void load(const std::string& filename, Mesh& mesh, Fields&... fields)
    {
        load(fs::current_path(), filename, mesh, fields...);
    }

    template <class Callback, class Mesh, class... Fields>
        requires mesh_like<Mesh> && std::invocable<Callback&, MetadataReader&>
    void load(const std::string& filename, Callback&& metadata, Mesh& mesh, Fields&... fields)
    {
        load(fs::current_path(), filename, std::forward<Callback>(metadata), mesh, fields...);
    }
}
