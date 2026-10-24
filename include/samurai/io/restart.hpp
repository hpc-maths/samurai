// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <concepts>
#include <filesystem>
#include <utility>
namespace fs = std::filesystem;

#include <highfive/H5Easy.hpp>
#include <highfive/H5PropertyList.hpp>
// H5Easy only enables its xtensor support when xtensor was included before it.
// Including the plug-in explicitly lets this header compile whatever comes first.
#include <highfive/xtensor.hpp>

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

// HDF5 compound type of an Interval, so that the intervals of a mesh are
// written and read as they are stored in memory.
// Not part of the API reference.
/// @cond
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

/// @endcond

namespace samurai
{

    namespace detail
    {
        /**
         * Writes the dataset `name`, one partition per process.
         *
         * Each process writes its `data` in rank order into `<name>/data`, and
         * `<name>/partition` holds the n_process + 1 bounds of the partitions.
         * Nothing is written when `data` is empty on every process.
         *
         * @param file the HDF5 file
         * @param name the name of the dataset
         * @param data the values of this process
         *
         * @note MPI: collective: every process must call it for the same
         *       datasets in the same order, even with an empty `data`.
         */
        template <class T>
        void dump_partitions(HighFive::File& file, const std::string& name, const std::vector<T>& data)
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

        /**
         * The block of partitions a process reads.
         *
         * Every dataset of a restart file is one array cut into contiguous
         * partitions, one per process that wrote it; `<name>/partition` holds
         * the n_process + 1 bounds of these partitions. A process reading the
         * file takes the contiguous block [first, last) of partitions, so the
         * number of processes that read a file does not have to match the
         * number that wrote it. With as many readers as writers, each reader
         * gets the partition written by the process of the same rank.
         */
        struct PartitionRange
        {
            std::size_t first;     ///< First partition of the block.
            std::size_t last;      ///< One past the last partition of the block.
            std::size_t n_process; ///< Number of processes that wrote the file.
        };

        /**
         * Checks the dimension stored in `/mesh/dim`.
         *
         * @param file the HDF5 file
         * @param expected the dimension of the mesh to load
         *
         * @throws std::runtime_error if `/mesh/dim` differs from `expected`.
         */
        inline void check_mesh_dim(const HighFive::File& file, std::size_t expected)
        {
            auto dim = H5Easy::load<std::size_t>(file, "/mesh/dim");
            if (dim != expected)
            {
                throw std::runtime_error(
                    fmt::format("The dimension of the mesh is not the same as the one of the mesh to be loaded. {} != {}", dim, expected));
            }
        }

        /**
         * Sets the origin point and the scaling factor of `array` from
         * `/mesh/origin_point` and `/mesh/scaling_factor`.
         */
        template <class array_t>
        void load_geometry(const HighFive::File& file, array_t& array)
        {
            // HighFive v3 does not support xfixed_container so we have to load into an xtensor_container first
            array.set_origin_point(H5Easy::load<xt::xtensor<double, 1>>(file, "/mesh/origin_point"));
            array.set_scaling_factor(H5Easy::load<double>(file, "/mesh/scaling_factor"));
        }

        /**
         * Returns the block of partitions this process reads: process `rank` of
         * `size` reads the partitions [rank * n / size, (rank + 1) * n / size),
         * where n is `/n_process`. Without MPI, the single process reads all of
         * them.
         */
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
            return {rank * n_process / size, (rank + 1) * n_process / size, n_process};
        }

        /**
         * Returns the single partition this process reads for a uniform mesh.
         *
         * A uniform mesh is replicated on every process, so each partition of
         * its file is a full copy: process `rank` reads partition
         * rank % n_process, where n_process is `/n_process`.
         */
        inline PartitionRange replicated_partition_to_read(const HighFive::File& file)
        {
#ifdef SAMURAI_WITH_MPI
            mpi::communicator world;
            auto rank = static_cast<std::size_t>(world.rank());
#else
            std::size_t rank = 0;
#endif
            auto n_process = H5Easy::load<std::size_t>(file, "/n_process");
            return {rank % n_process, rank % n_process + 1, n_process};
        }

        /**
         * The values of the partitions of a block, one after the other in a
         * single buffer.
         */
        template <class T>
        struct PartitionData
        {
            std::vector<T> values;           ///< The values of all the partitions of the block.
            std::vector<std::size_t> bounds; ///< Partition p is [bounds[p], bounds[p + 1]) in `values`.

            /// Number of values of partition `p`.
            std::size_t size(std::size_t p) const
            {
                return bounds[p + 1] - bounds[p];
            }

            /**
             * Returns the values of partition `p`. A buffer that holds a single
             * partition is moved out: do not use `values` afterwards.
             */
            std::vector<T> extract(std::size_t p)
            {
                if (bounds.size() == 2)
                {
                    return std::move(values);
                }
                return {values.begin() + static_cast<std::ptrdiff_t>(bounds[p]), values.begin() + static_cast<std::ptrdiff_t>(bounds[p + 1])};
            }
        };

        /**
         * Reads the partitions of `block` of the dataset `name` in one call.
         *
         * @param file the HDF5 file
         * @param name the name of the dataset, as given to dump_partitions
         * @param block the partitions to read
         * @return the values of the partitions and their bounds
         *
         * @throws std::runtime_error if `<name>/partition` does not hold
         *         `block.n_process` + 1 bounds.
         *
         * @note MPI: collective: every process must call it for the same
         *       datasets in the same order, even when its block is empty.
         */
        template <class T>
        PartitionData<T> load_partitions(const HighFive::File& file, const std::string& name, const PartitionRange& block)
        {
            auto xfer_props = HighFive::DataTransferProps{};
#ifdef SAMURAI_WITH_MPI
            xfer_props.add(HighFive::UseCollectiveIO{});
#endif
            auto bounds = H5Easy::load<std::vector<std::size_t>>(file, fmt::format("{}/partition", name));
            if (bounds.size() != block.n_process + 1)
            {
                throw std::runtime_error(
                    fmt::format("The partition of {} does not match the number of processes stored in the restart file.", name));
            }

            // The partitions of the block are contiguous in the file: read them in one call.
            auto dataset = file.getDataSet(fmt::format("{}/data", name));
            PartitionData<T> partitions;
            partitions.values.resize(bounds[block.last] - bounds[block.first]);
            dataset.select({bounds[block.first]}, {partitions.values.size()}).read_raw(partitions.values.data(), xfer_props);

            for (std::size_t p = block.first; p <= block.last; ++p)
            {
                partitions.bounds.push_back(bounds[p] - bounds[block.first]);
            }
            return partitions;
        }

        /**
         * Reads the cells of `level` written by each partition of `block`, in
         * the LevelCellArray layout of the process that wrote them. A level with
         * no cell in the whole file has no group, and gives empty arrays.
         *
         * @return one LevelCellArray per partition of `block`
         *
         * @note MPI: collective, as load_partitions.
         */
        template <class lca_t>
        std::vector<lca_t> load_level_partitions(const HighFive::File& file, std::size_t level, const PartitionRange& block)
        {
            constexpr std::size_t dim = lca_t::dim;
            using interval_t          = typename lca_t::interval_t;

            std::vector<lca_t> partitions(block.last - block.first, lca_t{level});
            if (!file.exist(fmt::format("/mesh/level/{}", level)))
            {
                return partitions;
            }

            for (std::size_t d = 0; d < dim; ++d)
            {
                auto intervals = load_partitions<interval_t>(file, fmt::format("/mesh/level/{}/dim/{}/intervals", level, d), block);
                for (std::size_t p = 0; p < partitions.size(); ++p)
                {
                    partitions[p][d] = intervals.extract(p);
                }
            }
            for (std::size_t d = 1; d < dim; ++d)
            {
                auto offsets = load_partitions<std::size_t>(file, fmt::format("/mesh/level/{}/dim/{}/offsets", level, d), block);
                for (std::size_t p = 0; p < partitions.size(); ++p)
                {
                    partitions[p].offsets(d) = offsets.extract(p);
                }
            }
            return partitions;
        }

        /**
         * Reads the cells written by each partition of `block`, in the
         * CellArray layout of the process that wrote them. The field values of
         * a partition are stored in the cell order of its CellArray.
         *
         * @return one CellArray per partition of `block`
         *
         * @throws std::runtime_error as check_mesh_dim and load_partitions.
         *
         * @note MPI: collective, as load_partitions.
         */
        template <class ca_t>
        std::vector<ca_t> load_cell_array_partitions(const HighFive::File& file, const PartitionRange& block)
        {
            using lca_t = typename ca_t::lca_type;

            check_mesh_dim(file, ca_t::dim);

            auto min_level = H5Easy::load<std::size_t>(file, "/mesh/min_level");
            auto max_level = H5Easy::load<std::size_t>(file, "/mesh/max_level");

            std::vector<ca_t> partitions(block.last - block.first);
            for (std::size_t level = min_level; level <= max_level; ++level)
            {
                auto lcas = load_level_partitions<lca_t>(file, level, block);
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

        /**
         * Returns the value of `/mesh/is_box`, or `false` when the file has no
         * such dataset: such files are read as non-box domains.
         */
        inline bool load_is_box(const HighFive::File& file)
        {
            return file.exist("/mesh/is_box") && H5Easy::load<bool>(file, "/mesh/is_box");
        }

        /**
         * Returns the union of the cells of several partitions of `level`.
         *
         * A single partition is moved out as is, so that a file read with the
         * number of processes that wrote it gives back the very same arrays.
         */
        template <std::size_t dim, class interval_t>
        LevelCellArray<dim, interval_t> merge_partitions(std::vector<LevelCellArray<dim, interval_t>>& partitions, std::size_t level)
        {
            if (partitions.size() == 1)
            {
                auto single = std::move(partitions.front());
                return single;
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

        /// Same as above, for the cells of every level.
        template <std::size_t dim, class interval_t, std::size_t max_size>
        CellArray<dim, interval_t, max_size> merge_partitions(std::vector<CellArray<dim, interval_t, max_size>>& partitions)
        {
            if (partitions.size() == 1)
            {
                auto single = std::move(partitions.front());
                return single;
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

        /**
         * Writes the number of components of `field` to `/fields/<name>/n_comp`
         * and its values on the cells of `mesh` to the dataset
         * `/fields/<name>/data`, where `<name>` is the name of the field.
         *
         * @note MPI: collective, as dump_partitions.
         */
        void dump_field(HighFive::File& file, const auto& mesh, const auto& field)
        {
            auto data = extract_data_as_vector(field, mesh);

            H5Easy::dump(file, fmt::format("/fields/{}/n_comp", field.name()), field.n_comp);

            dump_partitions(file, fmt::format("/fields/{}/data", field.name()), data);
        }

        /// Calls dump_field for each field, in order.
        template <class... Fields>
        void dump_fields(HighFive::File& file, const auto& mesh, const Fields&... fields)
        {
            if (sizeof...(Fields) > 0)
            {
                (dump_field(file, mesh, fields), ...);
            }
        }

        /**
         * Resizes `field` on its mesh and fills it with the values of each
         * partition of `block`. The values of a partition follow the cell order
         * of its CellArray, and land at the place of the same cells in the mesh
         * of the field.
         *
         * @param file the HDF5 file
         * @param block the partitions to read
         * @param partitions the cells of each partition, as read by
         *        load_cell_array_partitions or load_level_partitions. With a
         *        single partition, its cells are not read: the mesh of the field
         *        was built from them and has the same cell order.
         * @param field the field to fill, found in the file by its name
         *
         * @throws std::runtime_error if the field has no name, if the file has
         *         no group `/fields/<name>`, if `/fields/<name>/n_comp` differs
         *         from the number of components of the field, or if a partition
         *         does not hold one value per component for each of its cells.
         *
         * @note MPI: collective, as load_partitions.
         */
        template <class Field, class ca_t>
        void load_field(const HighFive::File& file, const PartitionRange& block, const std::vector<ca_t>& partitions, Field& field)
        {
            using size_type = typename Field::size_type;
            using mesh_id_t = typename std::decay_t<decltype(field.mesh())>::mesh_id_t;

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

            auto data = load_partitions<typename Field::value_type>(file, fmt::format("/fields/{}/data", field.name()), block);

            field.resize();

            // The partitions are one after the other in the buffer: k runs over all of them.
            std::size_t k    = 0;
            auto copy_values = [&](size_type cell_index)
            {
                if constexpr (Field::is_scalar)
                {
                    field[cell_index] = data.values[k++];
                }
                else
                {
                    for (size_type c = 0; c < Field::n_comp; ++c)
                    {
                        field[cell_index][c] = data.values[k++];
                    }
                }
            };
            auto check_size = [&](std::size_t n_values, std::size_t n_cells)
            {
                if (n_values != n_cells * Field::n_comp)
                {
                    throw std::runtime_error(fmt::format("The field {} does not have one value per cell in the restart file.", field.name()));
                }
            };

            // The mesh was built from the single partition (moved out by
            // merge_partitions), whose cell order is the order of the values.
            if (partitions.size() == 1)
            {
                const auto& cells = field.mesh()[mesh_id_t::cells];
                check_size(data.values.size(), cells.nb_cells());
                for_each_interval(cells,
                                  [&](std::size_t, const auto& interval, const auto&)
                                  {
                                      for (auto i = interval.start; i < interval.end; ++i)
                                      {
                                          copy_values(static_cast<size_type>(interval.index + i));
                                      }
                                  });
                return;
            }

            for (std::size_t p = 0; p < partitions.size(); ++p)
            {
                check_size(data.size(p), partitions[p].nb_cells());
                for_each_interval(partitions[p],
                                  [&](std::size_t level, const auto& interval, const auto& index)
                                  {
                                      // An interval of a partition lies inside one interval of the mesh, where its cells are contiguous.
                                      auto first = static_cast<size_type>(field.mesh().get_index(level, interval.start, index));
                                      for (auto i = interval.start; i < interval.end; ++i)
                                      {
                                          copy_values(first + static_cast<size_type>(i - interval.start));
                                      }
                                  });
            }
        }

        /// Calls load_field for each field, in order.
        template <class ca_t, class... Fields>
        void load_fields(const HighFive::File& file, const PartitionRange& block, const std::vector<ca_t>& partitions, Fields&... fields)
        {
            (load_field(file, block, partitions, fields), ...);
        }

#ifdef SAMURAI_WITH_MPI
        /**
         * Spreads the cells evenly over the processes with the SFC Hilbert
         * strategy and a uniform weight, moving the values of the fields with
         * them.
         *
         * @note MPI: collective on the world communicator.
         */
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

    /**
     * Writes a level cell array to an open checkpoint file.
     *
     * The intervals of each direction go to `/mesh/level/<level>/dim/<d>/intervals`
     * and, for d >= 1, the offsets to `/mesh/level/<level>/dim/<d>/offsets`.
     * Each of these datasets holds one partition per process.
     *
     * @param file the HDF5 file, open for writing
     * @param lca the level cell array of this process
     * @param with_metadata if `true`, also writes `/n_process`, `/mesh/dim`,
     *        `/mesh/min_level` and `/mesh/max_level` (both the level of `lca`),
     *        `/mesh/origin_point`, `/mesh/scaling_factor` and `/mesh/is_box`
     *
     * @note MPI: collective: every process must call it.
     */
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
            detail::dump_partitions(file, name, lca[d]);
        }
        for (std::size_t d = 1; d < dim; ++d)
        {
            auto name = fmt::format("/mesh/level/{}/dim/{}/offsets", lca.level(), d);
            detail::dump_partitions(file, name, lca.offsets(d));
        }
    }

    /**
     * Writes a cell array to an open checkpoint file.
     *
     * Writes `/n_process`, `/mesh/dim`, `/mesh/min_level` and
     * `/mesh/max_level` (the lowest and highest levels over all processes),
     * `/mesh/origin_point` and `/mesh/scaling_factor`, then the cells of each
     * level as dump(file, lca, false) does. A level with no cell on any process
     * has no group in the file.
     *
     * @param file the HDF5 file, open for writing
     * @param ca the cell array of this process
     *
     * @note MPI: collective: every process must call it.
     */
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

    /**
     * Writes a uniform mesh and fields to an open checkpoint file.
     *
     * Writes the real cells as dump(file, lca) does. Each field goes to the
     * group `/fields/<name>`: its number of components in `n_comp` and its
     * values on the real cells in the dataset `data`. Every process holds the
     * whole uniform mesh, so each partition of the file is a full copy.
     *
     * @param file the HDF5 file, open for writing
     * @param mesh the mesh
     * @param fields the fields defined on `mesh`; each must have a name
     *
     * @note MPI: collective: every process must call it with the same fields
     *       in the same order.
     */
    template <class Config>
    void dump(HighFive::File& file, const UniformMesh<Config>& mesh, const auto&... fields)
    {
        using Mesh      = UniformMesh<Config>;
        using mesh_id_t = typename Mesh::mesh_id_t;
        dump(file, mesh[mesh_id_t::cells]);
        detail::dump_fields(file, mesh[mesh_id_t::cells], fields...);
    }

    /**
     * Writes a multiresolution or AMR mesh and fields to an open checkpoint
     * file.
     *
     * Writes the real cells as dump(file, ca) does, then overwrites
     * `/mesh/min_level` and `/mesh/max_level` with the levels of the mesh
     * configuration, and writes `/mesh/is_box`, whether the domain is a box.
     * Each field goes to the group `/fields/<name>`: its number of components
     * in `n_comp` and its values on the real cells in the dataset `data`.
     *
     * @param file the HDF5 file, open for writing
     * @param mesh the mesh
     * @param fields the fields defined on `mesh`; each must have a name
     *
     * @note MPI: collective: every process must call it with the same fields
     *       in the same order.
     */
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
        detail::dump_fields(file, mesh[mesh_id_t::cells], fields...);
    }

    /**
     * Writes the mesh and the fields to a checkpoint file, which load reads
     * back to restart a simulation.
     *
     * The file is `<path>/<filename>.h5`. An existing file is overwritten, and
     * the directory `path` must exist. The file holds the number of processes
     * that wrote it (`/n_process`), the real cells of the mesh level by level
     * with its origin point, scaling factor and level range, and for each field
     * its number of components and its values on the real cells. Each process
     * writes its own partition of every dataset, in rank order. Ghosts and
     * boundary conditions are not written.
     *
     * @tparam Mesh a LevelCellArray, a CellArray, a UniformMesh, or a
     *         multiresolution or AMR mesh
     * @param path the directory of the file
     * @param filename the name of the file, without the `.h5` extension
     * @param mesh the mesh to write
     * @param fields the fields to write, defined on `mesh`; each must have a
     *        name. A LevelCellArray or a CellArray takes no field.
     *
     * @note MPI: collective on `MPI_COMM_WORLD`: every process must call it
     *       with the same file name and the same fields in the same order.
     */
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

    /**
     * Writes a checkpoint as dump(path, filename, mesh, fields) does, then
     * calls `metadata` to write user metadata (simulation time, parameters,
     * ...) to the same file.
     *
     * The callback receives a MetadataWriter on the HDF5 file only (a
     * checkpoint has no XDMF file). The values it writes are attributes of the
     * root group, which load gives back to its own callback through a
     * MetadataReader.
     *
     * @param path the directory of the file
     * @param filename the name of the file, without the `.h5` extension
     * @param metadata a callable invoked as `metadata(writer)` with a
     *        `MetadataWriter&`
     * @param mesh the mesh to write
     * @param fields the fields to write, defined on `mesh`
     *
     * @note MPI: collective on `MPI_COMM_WORLD`. The callback runs on every
     *       process and must write the same attributes with the same values
     *       everywhere.
     */
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
        std::forward<Callback>(metadata)(writer);
    }

    /**
     * Writes a checkpoint in the current working directory, as
     * dump(path, filename, mesh, fields) does.
     */
    template <class Mesh, class... Fields>
        requires mesh_like<Mesh>
    void dump(const std::string& filename, const Mesh& mesh, const Fields&... fields)
    {
        dump(fs::current_path(), filename, mesh, fields...);
    }

    /**
     * Writes a checkpoint and user metadata in the current working directory,
     * as dump(path, filename, metadata, mesh, fields) does.
     */
    template <class Callback, class Mesh, class... Fields>
        requires mesh_like<Mesh> && std::invocable<Callback&, MetadataWriter&>
    void dump(const std::string& filename, Callback&& metadata, const Mesh& mesh, const Fields&... fields)
    {
        dump(fs::current_path(), filename, std::forward<Callback>(metadata), mesh, fields...);
    }

    /**
     * Reads a level cell array from an open checkpoint file.
     *
     * Each process reads its block of partitions (see the load overload for a
     * mesh) and gets their union, without load balancing. The level cell array
     * is box-like when `/mesh/is_box` is `true`.
     *
     * @param file the HDF5 file
     * @param lca the level cell array to replace
     *
     * @throws std::runtime_error if `/mesh/dim` differs from `dim`, if
     *         `/mesh/min_level` differs from `/mesh/max_level`, if the file has
     *         no group for this level, or if the partition bounds of a dataset
     *         do not match `/n_process`.
     *
     * @note MPI: collective: every process must call it.
     */
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
        if (detail::load_is_box(file))
        {
            lca.box_like();
        }
    }

    /**
     * Reads a cell array from an open checkpoint file.
     *
     * Each process reads its block of partitions (see the load overload for a
     * mesh) and gets their union, without load balancing.
     *
     * @param file the HDF5 file
     * @param ca the cell array to replace
     *
     * @throws std::runtime_error if `/mesh/dim` differs from `dim`, or if the
     *         partition bounds of a dataset do not match `/n_process`.
     *
     * @note MPI: collective: every process must call it.
     */
    template <std::size_t dim, class interval_t, std::size_t max_size>
    void load(const HighFive::File& file, CellArray<dim, interval_t, max_size>& ca)
    {
        auto partitions = detail::load_cell_array_partitions<CellArray<dim, interval_t, max_size>>(file, detail::partitions_to_read(file));
        ca              = detail::merge_partitions(partitions);
        detail::load_geometry(file, ca);
    }

    /**
     * Reads a uniform mesh and fields from an open checkpoint file.
     *
     * The cells are those of level `/mesh/min_level`. A uniform mesh is
     * replicated on every process, so each partition of the file is a full
     * copy: process `rank` reads partition rank % n_process, where n_process
     * is `/n_process`, and gets the whole mesh, whatever the number of
     * processes. There is no merging and no load balancing. Then each field
     * is resized on the new mesh and filled with the values stored under its
     * name.
     *
     * @param file the HDF5 file
     * @param mesh the mesh to replace
     * @param fields the fields to fill, defined on `mesh`
     *
     * @throws std::runtime_error if `/mesh/dim` differs from the dimension of
     *         the mesh, if the partition bounds of a dataset do not match
     *         `/n_process`, if a field has no name, if the file has no group
     *         `/fields/<name>` for a field, if the number of components of a
     *         field differs from `/fields/<name>/n_comp`, or if the values of a
     *         field do not hold one value per component for each cell.
     *
     * @note MPI: collective: every process must call it with the same fields
     *       in the same order.
     */
    template <class Config, class... Fields>
    void load(const HighFive::File& file, UniformMesh<Config>& mesh, Fields&... fields)
    {
        // The cells of a uniform mesh are a single LevelCellArray.
        using ca_type = typename UniformMesh<Config>::ca_type;

        detail::check_mesh_dim(file, ca_type::dim);
        auto level      = H5Easy::load<std::size_t>(file, "/mesh/min_level");
        auto block      = detail::replicated_partition_to_read(file);
        auto partitions = detail::load_level_partitions<ca_type>(file, level, block);
        auto ca         = detail::merge_partitions(partitions, level);
        detail::load_geometry(file, ca);

        UniformMesh<Config> new_mesh{ca};
        std::swap(mesh, new_mesh);
        detail::load_fields(file, block, partitions, fields...);
    }

    /**
     * Reads a multiresolution or AMR mesh and fields from an open checkpoint
     * file.
     *
     * The file can be read by any number of processes. Process `rank` of
     * `size` reads the partitions [rank * n / size, (rank + 1) * n / size),
     * where n is `/n_process`, and merges them:
     * - with n processes, each one gets back the cells it wrote;
     * - with fewer processes, a process merges several partitions;
     * - with more processes, some processes read no partition.
     *
     * Without MPI, the single process reads every partition.
     *
     * The new mesh takes `/mesh/min_level` and `/mesh/max_level` from the file
     * and the rest of its configuration from `mesh`, and is box-like when
     * `/mesh/is_box` is `true`. It replaces `mesh`. Then each field is resized
     * on the new mesh and filled with the values stored under its name. Under
     * MPI, when the number of processes differs from `/n_process`, the cells
     * and the field values are then balanced over the processes with the SFC
     * Hilbert strategy.
     *
     * @param file the HDF5 file
     * @param mesh the mesh to replace, which gives the configuration
     * @param fields the fields to fill, defined on `mesh`
     *
     * @throws std::runtime_error if `/mesh/dim` differs from the dimension of
     *         the mesh, if the partition bounds of a dataset do not match
     *         `/n_process`, if a field has no name, if the file has no group
     *         `/fields/<name>` for a field, if the number of components of a
     *         field differs from `/fields/<name>/n_comp`, or if the values of a
     *         field do not hold one value per component for each cell.
     *
     * @note MPI: collective: every process must call it with the same fields
     *       in the same order.
     */
    template <class Mesh, class... Fields>
    void load(const HighFive::File& file, Mesh& mesh, Fields&... fields)
    {
        using ca_type = typename Mesh::ca_type;

        auto min_level = H5Easy::load<std::size_t>(file, "/mesh/min_level");
        auto max_level = H5Easy::load<std::size_t>(file, "/mesh/max_level");

        auto block      = detail::partitions_to_read(file);
        auto partitions = detail::load_cell_array_partitions<ca_type>(file, block);
        auto ca         = detail::merge_partitions(partitions);
        detail::load_geometry(file, ca);

        auto mesh_cfg = mesh.cfg();
        mesh_cfg.min_level(min_level).max_level(max_level).disable_args_parse();
        Mesh new_mesh{ca, mesh_cfg};
        if (detail::load_is_box(file))
        {
            new_mesh.box_like();
        }
        std::swap(mesh, new_mesh);
        detail::load_fields(file, block, partitions, fields...);

#ifdef SAMURAI_WITH_MPI
        mpi::communicator world;
        if (block.n_process != static_cast<std::size_t>(world.size()))
        {
            detail::balance_after_load(mesh, fields...);
        }
#endif
    }

    /**
     * Reads the mesh and the fields from a checkpoint file written by dump.
     *
     * Opens `<path>/<filename>.h5` read-only and calls the load overload that
     * matches the mesh type. The file can be read by any number of processes.
     * Under MPI, for a multiresolution or AMR mesh, when that number differs
     * from the number of processes that wrote the file, the cells and the
     * fields are balanced over the processes with the SFC Hilbert strategy.
     * A LevelCellArray or a CellArray gets the union of the partitions its
     * process reads, without balancing. A UniformMesh is replicated on every
     * process: each process reads one full copy of it.
     *
     * For a multiresolution or AMR mesh, the new mesh takes its minimum and
     * maximum levels from the file and the rest of its configuration from
     * `mesh`. Each field is resized and filled with the values stored under its
     * name. Ghosts and boundary conditions are not restored.
     *
     * @tparam Mesh a LevelCellArray, a CellArray, a UniformMesh, or a
     *         multiresolution or AMR mesh
     * @param path the directory of the file
     * @param filename the name of the file, without the `.h5` extension
     * @param mesh the mesh to replace
     * @param fields the fields to fill, defined on `mesh`
     *
     * @throws std::runtime_error if `/mesh/dim` differs from the dimension of
     *         the mesh, if the partition bounds of a dataset do not match
     *         `/n_process`, if a field has no name, if the file has no group
     *         `/fields/<name>` for a field, if the number of components of a
     *         field differs from `/fields/<name>/n_comp`, or if the values of a
     *         field do not hold one value per component for each cell. For a
     *         LevelCellArray, also if `/mesh/min_level` differs from
     *         `/mesh/max_level` or if the file has no group for that level.
     * @throws HighFive::Exception if the file cannot be opened or a dataset is
     *         missing.
     *
     * @note MPI: collective on `MPI_COMM_WORLD`: every process must call it
     *       with the same file name and the same fields in the same order.
     */
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

    /**
     * Reads a checkpoint as load(path, filename, mesh, fields) does, then calls
     * `metadata` to read the user metadata written by dump.
     *
     * The callback receives a MetadataReader on the file, after the mesh and
     * the fields are loaded.
     *
     * @param path the directory of the file
     * @param filename the name of the file, without the `.h5` extension
     * @param metadata a callable invoked as `metadata(reader)` with a
     *        `MetadataReader&`
     * @param mesh the mesh to replace
     * @param fields the fields to fill, defined on `mesh`
     *
     * @note MPI: collective on `MPI_COMM_WORLD`. The callback runs on every
     *       process.
     */
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
        std::forward<Callback>(metadata)(reader);
    }

    /**
     * Reads a checkpoint from the current working directory, as
     * load(path, filename, mesh, fields) does.
     */
    template <class Mesh, class... Fields>
        requires mesh_like<Mesh>
    void load(const std::string& filename, Mesh& mesh, Fields&... fields)
    {
        load(fs::current_path(), filename, mesh, fields...);
    }

    /**
     * Reads a checkpoint and user metadata from the current working directory,
     * as load(path, filename, metadata, mesh, fields) does.
     */
    template <class Callback, class Mesh, class... Fields>
        requires mesh_like<Mesh> && std::invocable<Callback&, MetadataReader&>
    void load(const std::string& filename, Callback&& metadata, Mesh& mesh, Fields&... fields)
    {
        load(fs::current_path(), filename, std::forward<Callback>(metadata), mesh, fields...);
    }
}
