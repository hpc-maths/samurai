// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <concepts>
#include <filesystem>
#include <numeric>
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
        /// Returns the rank of this process and the number of processes: 0 and 1 without MPI.
        inline std::pair<std::size_t, std::size_t> rank_and_size()
        {
#ifdef SAMURAI_WITH_MPI
            mpi::communicator world;
            return {static_cast<std::size_t>(world.rank()), static_cast<std::size_t>(world.size())};
#else
            return {0, 1};
#endif
        }

        /// Transfer properties of the partition reads and writes: collective under MPI.
        inline HighFive::DataTransferProps transfer_props()
        {
            HighFive::DataTransferProps xfer_props;
#ifdef SAMURAI_WITH_MPI
            xfer_props.add(HighFive::UseCollectiveIO{});
#endif
            return xfer_props;
        }

        /**
         * Opens `<path>/<filename>.h5` with `mode`, through MPI-IO under MPI.
         *
         * @note MPI: collective on `MPI_COMM_WORLD`.
         */
        inline HighFive::File open_file(const fs::path& path, const std::string& filename, HighFive::File::AccessMode mode)
        {
            HighFive::FileAccessProps fapl;
#ifdef SAMURAI_WITH_MPI
            fapl.add(HighFive::MPIOFileAccess{MPI_COMM_WORLD, MPI_INFO_NULL});
            fapl.add(HighFive::MPIOCollectiveMetadata{});
#endif
            return HighFive::File(fmt::format("{}.h5", (path / filename).string()), mode, fapl);
        }

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
            auto [rank, size] = rank_and_size();
            std::vector<std::size_t> local_sizes(size, data.size());
#ifdef SAMURAI_WITH_MPI
            mpi::all_gather(mpi::communicator{}, data.size(), local_sizes);
#endif
            std::vector<std::size_t> bounds(size + 1, 0);
            std::partial_sum(local_sizes.begin(), local_sizes.end(), bounds.begin() + 1);
            if (bounds.back() == 0)
            {
                return;
            }

            H5Easy::dump(file, fmt::format("{}/partition", name), bounds);
            auto dataset = file.createDataSet<T>(fmt::format("{}/data", name), HighFive::DataSpace(std::vector<std::size_t>{bounds.back()}));
            dataset.select({bounds[rank]}, {data.size()}).write_raw(data.data(), HighFive::AtomicType<T>{}, transfer_props());
        }

        /**
         * Writes `/n_process`, `/mesh/dim`, `/mesh/min_level` and
         * `/mesh/max_level`, and the origin point and scaling factor of `array`
         * to `/mesh/origin_point` and `/mesh/scaling_factor`.
         */
        template <class array_t>
        void dump_mesh_metadata(HighFive::File& file, const array_t& array, std::size_t min_level, std::size_t max_level)
        {
#ifdef SAMURAI_WITH_MPI
            H5Easy::dump(file, "/n_process", mpi::communicator{}.size());
#else
            H5Easy::dump(file, "/n_process", std::size_t{1});
#endif
            H5Easy::dump(file, "/mesh/dim", array_t::dim);
            H5Easy::dump(file, "/mesh/min_level", min_level);
            H5Easy::dump(file, "/mesh/max_level", max_level);
            // HighFive v3 does not support xfixed_container so we have to dump from an xtensor_container first
            H5Easy::dump(file, "/mesh/origin_point", xt::xtensor<double, 1>{array.origin_point()});
            H5Easy::dump(file, "/mesh/scaling_factor", array.scaling_factor());
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

        /// How the cells of a mesh are spread over the processes.
        enum class Layout
        {
            distributed, ///< Each process holds its own part of the mesh.
            replicated   ///< Each process holds the whole mesh, as a UniformMesh.
        };

        /**
         * Returns the partitions this process reads, where n is `/n_process`.
         *
         * For a distributed mesh, process `rank` of `size` reads the block
         * [rank * n / size, (rank + 1) * n / size); without MPI, the single
         * process reads all of them. For a replicated mesh, each partition is a
         * full copy, and process `rank` reads partition rank % n.
         */
        inline PartitionRange partitions_to_read(const HighFive::File& file, Layout layout)
        {
            auto [rank, size] = rank_and_size();
            auto n_process    = H5Easy::load<std::size_t>(file, "/n_process");
            if (layout == Layout::replicated)
            {
                return {rank % n_process, rank % n_process + 1, n_process};
            }
            return {rank * n_process / size, (rank + 1) * n_process / size, n_process};
        }

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
         * Returns the value of `/mesh/is_box`, or `false` when the file has no
         * such dataset: such files are read as non-box domains.
         */
        inline bool load_is_box(const HighFive::File& file)
        {
            return file.exist("/mesh/is_box") && H5Easy::load<bool>(file, "/mesh/is_box");
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
            auto bounds = H5Easy::load<std::vector<std::size_t>>(file, fmt::format("{}/partition", name));
            if (bounds.size() != block.n_process + 1)
            {
                throw std::runtime_error(
                    fmt::format("The partition of {} does not match the number of processes stored in the restart file.", name));
            }

            // The partitions of the block are contiguous in the file: read them in one call.
            PartitionData<T> partitions;
            partitions.values.resize(bounds[block.last] - bounds[block.first]);
            file.getDataSet(fmt::format("{}/data", name))
                .select({bounds[block.first]}, {partitions.values.size()})
                .read_raw(partitions.values.data(), transfer_props());
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
            std::vector<lca_t> partitions(block.last - block.first, lca_t{level});
            if (!file.exist(fmt::format("/mesh/level/{}", level)))
            {
                return partitions;
            }

            for (std::size_t d = 0; d < lca_t::dim; ++d)
            {
                auto intervals = load_partitions<typename lca_t::interval_t>(file,
                                                                             fmt::format("/mesh/level/{}/dim/{}/intervals", level, d),
                                                                             block);
                for (std::size_t p = 0; p < partitions.size(); ++p)
                {
                    partitions[p][d] = intervals.extract(p);
                }
            }
            for (std::size_t d = 1; d < lca_t::dim; ++d)
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
            check_mesh_dim(file, ca_t::dim);

            auto min_level = H5Easy::load<std::size_t>(file, "/mesh/min_level");
            auto max_level = H5Easy::load<std::size_t>(file, "/mesh/max_level");

            std::vector<ca_t> partitions(block.last - block.first);
            for (std::size_t level = min_level; level <= max_level; ++level)
            {
                auto lcas = load_level_partitions<typename ca_t::lca_type>(file, level, block);
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
         * Returns the union of the cells of several partitions, built in `list`:
         * a LevelCellList of the level for LevelCellArray partitions, an empty
         * CellList for CellArray partitions.
         *
         * A single partition is moved out as is, so that a file read with the
         * number of processes that wrote it gives back the very same arrays.
         */
        template <class array_t, class list_t>
        array_t merge_partitions(std::vector<array_t>& partitions, list_t list)
        {
            if (partitions.size() == 1)
            {
                auto single = std::move(partitions.front());
                return single;
            }
            for (const auto& partition : partitions)
            {
                for_each_interval(partition,
                                  [&](std::size_t level, const auto& interval, const auto& index)
                                  {
                                      if constexpr (requires { typename array_t::lca_type; }) // a CellArray
                                      {
                                          list[level][index].add_interval(interval);
                                      }
                                      else
                                      {
                                          list[index].add_interval(interval);
                                      }
                                  });
            }
            return array_t{list};
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
            H5Easy::dump(file, fmt::format("/fields/{}/n_comp", field.name()), field.n_comp);
            dump_partitions(file, fmt::format("/fields/{}/data", field.name()), extract_data_as_vector(field, mesh));
        }

        /// Calls dump_field for each field, in order.
        template <class... Fields>
        void dump_fields(HighFive::File& file, const auto& mesh, const Fields&... fields)
        {
            (dump_field(file, mesh, fields), ...);
        }

        /**
         * Checks that the file holds `field`: the field has a name, the file
         * has the group `/fields/<name>`, and `/fields/<name>/n_comp` is the
         * number of components of the field.
         *
         * @throws std::runtime_error if one of these conditions fails.
         */
        template <class Field>
        void check_field(const HighFive::File& file, const Field& field)
        {
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
         * @throws std::runtime_error as check_field, or if a partition does not
         *         hold one value per component for each of its cells.
         *
         * @note MPI: collective, as load_partitions.
         */
        template <class Field, class array_t>
        void load_field(const HighFive::File& file, const PartitionRange& block, const std::vector<array_t>& partitions, Field& field)
        {
            using size_type = typename Field::size_type;
            using mesh_id_t = typename std::decay_t<decltype(field.mesh())>::mesh_id_t;

            check_field(file, field);
            auto data = load_partitions<typename Field::value_type>(file, fmt::format("/fields/{}/data", field.name()), block);
            field.resize();

            // Copies the values of the cells of `source`, in its cell order. The
            // partitions are one after the other in the buffer: k runs over all
            // of them. An interval of `source` lies inside one interval of the
            // mesh, where its cells are contiguous: `first_index` gives the
            // storage index of its first cell.
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
            auto copy_cells = [&](const auto& source, std::size_t n_values, auto first_index)
            {
                if (n_values != source.nb_cells() * Field::n_comp)
                {
                    throw std::runtime_error(fmt::format("The field {} does not have one value per cell in the restart file.", field.name()));
                }
                for_each_interval(source,
                                  [&](std::size_t level, const auto& interval, const auto& index)
                                  {
                                      auto first = static_cast<size_type>(first_index(level, interval, index));
                                      for (auto i = interval.start; i < interval.end; ++i)
                                      {
                                          copy_values(first + static_cast<size_type>(i - interval.start));
                                      }
                                  });
            };

            // A single partition is the source of the mesh: its intervals hold
            // the storage index of their cells. Otherwise, each interval of a
            // partition is searched in the mesh.
            if (partitions.size() == 1)
            {
                copy_cells(field.mesh()[mesh_id_t::cells],
                           data.values.size(),
                           [](std::size_t, const auto& interval, const auto&)
                           {
                               return interval.index + interval.start;
                           });
                return;
            }
            for (std::size_t p = 0; p < partitions.size(); ++p)
            {
                copy_cells(partitions[p],
                           data.size(p),
                           [&](std::size_t level, const auto& interval, const auto& index)
                           {
                               return field.mesh().get_index(level, interval.start, index);
                           });
            }
        }

        /// Calls load_field for each field, in order.
        template <class array_t, class... Fields>
        void load_fields(const HighFive::File& file, const PartitionRange& block, const std::vector<array_t>& partitions, Fields&... fields)
        {
            (load_field(file, block, partitions, fields), ...);
        }

        /**
         * Replaces `mesh` with the mesh that `make_mesh` builds from the union
         * of `partitions`, then fills the fields.
         *
         * @param file the HDF5 file
         * @param block the partitions read
         * @param partitions the cells of each partition of `block`
         * @param list the cell list given to merge_partitions
         * @param mesh the mesh to replace
         * @param make_mesh a callable that returns the new mesh from the merged
         *        cells, origin point and scaling factor set
         * @param fields the fields to fill, defined on `mesh`
         *
         * @note MPI: collective, as load_partitions.
         */
        template <class array_t, class list_t, class Mesh, class MakeMesh, class... Fields>
        void load_mesh(const HighFive::File& file,
                       const PartitionRange& block,
                       std::vector<array_t>& partitions,
                       list_t list,
                       Mesh& mesh,
                       MakeMesh&& make_mesh,
                       Fields&... fields)
        {
            auto cells = merge_partitions(partitions, std::move(list));
            load_geometry(file, cells);
            Mesh new_mesh = make_mesh(cells);
            std::swap(mesh, new_mesh);
            load_fields(file, block, partitions, fields...);
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
            detail::dump_mesh_metadata(file, lca, lca.level(), lca.level());
            H5Easy::dump(file, "/mesh/is_box", lca.is_box());
        }
        for (std::size_t d = 0; d < dim; ++d)
        {
            detail::dump_partitions(file, fmt::format("/mesh/level/{}/dim/{}/intervals", lca.level(), d), lca[d]);
        }
        for (std::size_t d = 1; d < dim; ++d)
        {
            detail::dump_partitions(file, fmt::format("/mesh/level/{}/dim/{}/offsets", lca.level(), d), lca.offsets(d));
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
        std::size_t min_level = ca.min_level();
        std::size_t max_level = ca.max_level();
#ifdef SAMURAI_WITH_MPI
        mpi::communicator world;
        min_level = mpi::all_reduce(world, min_level, mpi::minimum<std::size_t>());
        max_level = mpi::all_reduce(world, max_level, mpi::maximum<std::size_t>());
#endif
        detail::dump_mesh_metadata(file, ca, min_level, max_level);
        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            dump(file, ca[level], false);
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
        const auto& cells = mesh[UniformMesh<Config>::mesh_id_t::cells];
        dump(file, cells);
        detail::dump_fields(file, cells, fields...);
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
        const auto& cells = mesh[Mesh_base<D, Config>::mesh_id_t::cells];
        dump(file, cells);
        H5Easy::dump(file, "/mesh/min_level", mesh.min_level(), H5Easy::DumpMode::Overwrite);
        H5Easy::dump(file, "/mesh/max_level", mesh.max_level(), H5Easy::DumpMode::Overwrite);
        // The domain is rebuilt from the cells on load, which cannot tell whether it is a box.
        H5Easy::dump(file, "/mesh/is_box", mesh.domain().is_box());
        detail::dump_fields(file, cells, fields...);
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
        auto file = detail::open_file(path, filename, HighFive::File::Overwrite);
        dump(file, mesh, fields...);
        MetadataWriter writer(file); // no XDMF for a checkpoint: HDF5 only
        std::forward<Callback>(metadata)(writer);
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
        dump(path, filename, [](MetadataWriter&) {}, mesh, fields...);
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

        auto block      = detail::partitions_to_read(file, detail::Layout::distributed);
        auto partitions = detail::load_level_partitions<LevelCellArray<dim, interval_t>>(file, min_level, block);
        lca             = detail::merge_partitions(partitions, LevelCellList<dim, interval_t>{min_level});
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
        using ca_type   = CellArray<dim, interval_t, max_size>;
        auto block      = detail::partitions_to_read(file, detail::Layout::distributed);
        auto partitions = detail::load_cell_array_partitions<ca_type>(file, block);
        ca              = detail::merge_partitions(partitions, typename ca_type::cl_type{});
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
        using ca_type = typename UniformMesh<Config>::ca_type; // a LevelCellArray

        detail::check_mesh_dim(file, ca_type::dim);
        auto level      = H5Easy::load<std::size_t>(file, "/mesh/min_level");
        auto block      = detail::partitions_to_read(file, detail::Layout::replicated);
        auto partitions = detail::load_level_partitions<ca_type>(file, level, block);
        detail::load_mesh(
            file,
            block,
            partitions,
            LevelCellList<ca_type::dim, typename ca_type::interval_t>{level},
            mesh,
            [](const auto& cells)
            {
                return UniformMesh<Config>{cells};
            },
            fields...);
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

        auto mesh_cfg = mesh.cfg();
        mesh_cfg.min_level(H5Easy::load<std::size_t>(file, "/mesh/min_level"))
            .max_level(H5Easy::load<std::size_t>(file, "/mesh/max_level"))
            .disable_args_parse();

        auto block      = detail::partitions_to_read(file, detail::Layout::distributed);
        auto partitions = detail::load_cell_array_partitions<ca_type>(file, block);
        detail::load_mesh(
            file,
            block,
            partitions,
            typename ca_type::cl_type{},
            mesh,
            [&](const auto& cells)
            {
                Mesh new_mesh{cells, mesh_cfg};
                if (detail::load_is_box(file))
                {
                    new_mesh.box_like();
                }
                return new_mesh;
            },
            fields...);

#ifdef SAMURAI_WITH_MPI
        if (block.n_process != detail::rank_and_size().second)
        {
            detail::balance_after_load(mesh, fields...);
        }
#endif
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
        auto file = detail::open_file(path, filename, HighFive::File::ReadOnly);
        load(file, mesh, fields...);
        MetadataReader reader(file);
        std::forward<Callback>(metadata)(reader);
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
        load(path, filename, [](MetadataReader&) {}, mesh, fields...);
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
