// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <array>
#include <string>
#include <type_traits>

#include <filesystem>
namespace fs = std::filesystem;

#include <pugixml.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>

#include <fmt/core.h>

#include <highfive/H5Easy.hpp>
#include <highfive/H5PropertyList.hpp>

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <mpi.h>
namespace mpi = boost::mpi;
#endif

#include "../algorithm.hpp"
#include "../arguments.hpp"
#include "../concepts.hpp"
#include "../field.hpp"
#include "../interval.hpp"
#include "../timers.hpp"
#include "../utils.hpp"
#include "util.hpp"

namespace samurai
{
    SAMURAI_INLINE std::string element_type(std::size_t dim)
    {
        switch (dim)
        {
            case 1:
                return "Polyline";
            case 2:
                return "Quadrilateral";
            case 3:
                return "Hexahedron";
            default:
                break;
        }
        return "Unknown Element Type";
    }

    template <std::size_t dim>
    auto get_element(std::integral_constant<std::size_t, dim>);

    SAMURAI_INLINE auto get_element(std::integral_constant<std::size_t, 1>)
    {
        return std::array<double, 2>{
            {0, 1}
        };
    }

    SAMURAI_INLINE auto get_element(std::integral_constant<std::size_t, 2>)
    {
        return std::array<xt::xtensor_fixed<std::size_t, xt::xshape<2>>, 4>{
            {{0, 0}, {1, 0}, {1, 1}, {0, 1}}
        };
    }

    SAMURAI_INLINE auto get_element(std::integral_constant<std::size_t, 3>)
    {
        return std::array<xt::xtensor_fixed<std::size_t, xt::xshape<3>>, 8>{
            {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}}
        };
    }

    template <class Mesh>
    auto extract_coords_and_connectivity(const Mesh& mesh)
    {
        static constexpr std::size_t dim = Mesh::dim;
        std::size_t nb_cells             = mesh.nb_cells();

        if (nb_cells == 0)
        {
            xt::xtensor<std::size_t, 2> connectivity = xt::zeros<std::size_t>({0, 0});
            xt::xtensor<double, 2> coords            = xt::zeros<double>({0, 0});
            return std::make_pair(coords, connectivity);
        }

        std::size_t nb_points_per_cell = 1 << dim;

        std::map<std::array<double, dim>, std::size_t> points_id;
        auto element = get_element(std::integral_constant<std::size_t, dim>{});

        xt::xtensor<std::size_t, 2> connectivity;
        connectivity.resize({nb_cells, nb_points_per_cell});

        std::size_t id    = 0;
        std::size_t index = 0;
        for_each_cell(mesh,
                      [&](auto cell)
                      {
                          std::array<double, dim> a;
                          auto start_corner = cell.corner();
                          auto c            = xt::xtensor<std::size_t, 1>::from_shape({element.size()});
                          ;

                          for (std::size_t i = 0; i < element.size(); ++i)
                          {
                              auto corner = start_corner + cell.length * element[i];

                              std::copy(corner.cbegin(), corner.cend(), a.begin());
                              auto search = points_id.find(a);
                              if (search == points_id.end())
                              {
                                  points_id.emplace(std::make_pair(a, id));
                                  c[i] = id++;
                              }
                              else
                              {
                                  c[i] = search->second;
                              }
                          }
                          auto connectivity_view = xt::view(connectivity, index, xt::all());
                          connectivity_view      = c;
                          index++;
                      });

        auto coords = xt::xtensor<double, 2>::from_shape({points_id.size(), 3});
        coords.fill(0.);
        for (auto& e : points_id)
        {
            std::size_t idx  = e.second;
            auto coords_view = xt::view(coords, idx, xt::range(0, dim));
            coords_view      = xt::adapt(e.first);
        }
        return std::make_pair(coords, connectivity);
    }

    template <class D>
    struct Hdf5Options
    {
        Hdf5Options(bool level = false, bool mesh_id = false)
            : by_level(level)
            , by_mesh_id(mesh_id)
        {
        }

        bool by_level   = false;
        bool by_mesh_id = false;
    };

    template <class Config>
    class UniformMesh;

    template <class Config>
    struct Hdf5Options<UniformMesh<Config>>
    {
        Hdf5Options(bool mesh_id = false)
            : by_mesh_id(mesh_id)
        {
        }

        bool by_mesh_id;
    };

    template <class D>
    class Hdf5
    {
      public:

        using derived_type_save = D;

#ifdef SAMURAI_WITH_MPI
        Hdf5(const fs::path& path, const std::string& filename, MPI_Comm comm);
#else
        Hdf5(const fs::path& path, const std::string& filename);
#endif

        ~Hdf5();

        Hdf5(const Hdf5&)            = delete;
        Hdf5& operator=(const Hdf5&) = delete;

        Hdf5(Hdf5&&) noexcept            = default;
        Hdf5& operator=(Hdf5&&) noexcept = default;

        derived_type_save& derived_cast() & noexcept;
        const derived_type_save& derived_cast() const& noexcept;
        derived_type_save derived_cast() && noexcept;

      protected:

        pugi::xml_node& domain();

        template <class Submesh>
        void save_on_mesh(pugi::xml_node& grid_parent, const std::string& prefix, const Submesh& submesh, const std::string& mesh_name);

        template <class Submesh, class Field>
        SAMURAI_INLINE void save_field(pugi::xml_node& grid, const std::string& prefix, const Submesh& submesh, const Field& field);

#ifdef SAMURAI_WITH_MPI
        MPI_Comm m_mpi_comm;
#endif

      private:

#ifdef SAMURAI_WITH_MPI
        static HighFive::File create_h5file(const fs::path& path, const std::string& filename, MPI_Comm comm);
#else
        static HighFive::File create_h5file(const fs::path& path, const std::string& filename);
#endif

        HighFive::File h5_file;
        fs::path m_path;
        std::string m_filename;
        pugi::xml_document m_doc;
        pugi::xml_node m_domain;
    };

    template <class D, class Mesh, class... T>
    class SaveBase : public Hdf5<SaveBase<D, Mesh, T...>>
    {
      public:

        using derived_type               = D;
        using hdf5_t                     = Hdf5<SaveBase<D, Mesh, T...>>;
        using options_t                  = Hdf5Options<Mesh>;
        using mesh_t                     = Mesh;
        static constexpr std::size_t dim = mesh_t::dim;

#ifdef SAMURAI_WITH_MPI
        SaveBase(const fs::path& path, const std::string& filename, MPI_Comm comm, const options_t& options, const Mesh& mesh, const T&... fields);
#else
        SaveBase(const fs::path& path, const std::string& filename, const options_t& options, const Mesh& mesh, const T&... fields);
#endif
        SaveBase(const SaveBase&)            = delete;
        SaveBase& operator=(const SaveBase&) = delete;

        SaveBase(SaveBase&&) noexcept            = default;
        SaveBase& operator=(SaveBase&&) noexcept = default;

        ~SaveBase() = default;

        void save();

        template <class Submesh>
        void save_fields(pugi::xml_node& grid, const std::string& prefix, const Submesh& submesh);

        template <class Submesh, std::size_t... I>
        void save_fields_impl(pugi::xml_node& grid, const std::string& prefix, const Submesh& submesh, std::index_sequence<I...>);

        derived_type& derived_cast() & noexcept;
        const derived_type& derived_cast() const& noexcept;
        derived_type derived_cast() && noexcept;

      protected:

        const mesh_t& mesh() const;
        const options_t& options() const;

      private:

        using fields_type = std::tuple<const T&...>;

        const mesh_t& m_mesh; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
        options_t m_options;
        fields_type m_fields;
    };

#ifdef SAMURAI_WITH_MPI
    template <class D, class Mesh, class... T>
    SAMURAI_INLINE SaveBase<D, Mesh, T...>::SaveBase(const fs::path& path,
                                                     const std::string& filename,
                                                     MPI_Comm comm,
                                                     const options_t& options,
                                                     const Mesh& mesh,
                                                     const T&... fields)
        : hdf5_t(path, filename, comm)
        , m_mesh(mesh)
        , m_options(options)
        , m_fields(fields...)
    {
    }
#else
    template <class D, class Mesh, class... T>
    SAMURAI_INLINE SaveBase<D, Mesh, T...>::SaveBase(const fs::path& path,
                                                     const std::string& filename,
                                                     const options_t& options,
                                                     const Mesh& mesh,
                                                     const T&... fields)
        : hdf5_t(path, filename)
        , m_mesh(mesh)
        , m_options(options)
        , m_fields(fields...)
    {
    }
#endif
    template <class D, class Mesh, class... T>
    template <class Submesh>
    SAMURAI_INLINE void SaveBase<D, Mesh, T...>::save_fields(pugi::xml_node& grid, const std::string& prefix, const Submesh& submesh)
    {
        save_fields_impl(grid, prefix, submesh, std::make_index_sequence<sizeof...(T)>());
    }

    template <class D, class Mesh, class... T>
    template <class Submesh, std::size_t... I>
    SAMURAI_INLINE void SaveBase<D, Mesh, T...>::save_fields_impl(pugi::xml_node& grid,
                                                                  const std::string& prefix,
                                                                  const Submesh& submesh,
                                                                  std::index_sequence<I...>)
    {
        (this->save_field(grid, prefix, submesh, std::get<I>(m_fields)), ...);
    }

    template <class D, class Mesh, class... T>
    SAMURAI_INLINE void SaveBase<D, Mesh, T...>::save()
    {
        this->derived_cast().save();
    }

    template <class D, class Mesh, class... T>
    SAMURAI_INLINE auto SaveBase<D, Mesh, T...>::derived_cast() & noexcept -> derived_type&
    {
        return *static_cast<derived_type*>(this);
    }

    template <class D, class Mesh, class... T>
    SAMURAI_INLINE auto SaveBase<D, Mesh, T...>::derived_cast() const& noexcept -> const derived_type&
    {
        return *static_cast<const derived_type*>(this);
    }

    template <class D, class Mesh, class... T>
    SAMURAI_INLINE auto SaveBase<D, Mesh, T...>::derived_cast() && noexcept -> derived_type
    {
        return *static_cast<derived_type*>(this);
    }

    template <class D, class Mesh, class... T>
    SAMURAI_INLINE auto SaveBase<D, Mesh, T...>::mesh() const -> const mesh_t&
    {
        return m_mesh;
    }

    template <class D, class Mesh, class... T>
    SAMURAI_INLINE auto SaveBase<D, Mesh, T...>::options() const -> const options_t&
    {
        return m_options;
    }

    template <class D, class Mesh, class... T>
    class SaveCellArray : public SaveBase<D, Mesh, T...>
    {
      public:

        using base_class                 = SaveBase<D, Mesh, T...>;
        using derived_type               = D;
        using options_t                  = typename base_class::options_t;
        using mesh_t                     = typename base_class::mesh_t;
        static constexpr std::size_t dim = base_class::dim;

#ifdef SAMURAI_WITH_MPI
        SaveCellArray(const fs::path& path,
                      const std::string& filename,
                      MPI_Comm comm,
                      const options_t& options,
                      const mesh_t& mesh,
                      const T&... fields);
#else
        SaveCellArray(const fs::path& path, const std::string& filename, const options_t& options, const mesh_t& mesh, const T&... fields);
#endif
        void save();
    };

#ifdef SAMURAI_WITH_MPI
    template <class D, class Mesh, class... T>
    SAMURAI_INLINE SaveCellArray<D, Mesh, T...>::SaveCellArray(const fs::path& path,
                                                               const std::string& filename,
                                                               MPI_Comm comm,
                                                               const options_t& options,
                                                               const mesh_t& mesh,
                                                               const T&... fields)
        : base_class(path, filename, comm, options, mesh, fields...)
    {
    }
#else
    template <class D, class Mesh, class... T>
    SAMURAI_INLINE SaveCellArray<D, Mesh, T...>::SaveCellArray(const fs::path& path,
                                                               const std::string& filename,
                                                               const options_t& options,
                                                               const mesh_t& mesh,
                                                               const T&... fields)
        : base_class(path, filename, options, mesh, fields...)
    {
    }
#endif

    template <class D, class Mesh, class... T>
    SAMURAI_INLINE void SaveCellArray<D, Mesh, T...>::save()
    {
        if (this->options().by_level)
        {
#ifdef SAMURAI_WITH_MPI
            int result;
            MPI_Comm_compare(this->m_mpi_comm, MPI_COMM_SELF, &result);

            std::size_t min_level;
            std::size_t max_level;

            if (result == MPI_IDENT || result == MPI_CONGRUENT)
            {
                min_level = this->mesh().min_level();
                max_level = this->mesh().max_level();
            }
            else
            {
                mpi::communicator world;
                min_level = mpi::all_reduce(world, this->mesh().min_level(), mpi::minimum<std::size_t>());
                max_level = mpi::all_reduce(world, this->mesh().max_level(), mpi::maximum<std::size_t>());
            }
#else
            auto min_level = this->mesh().min_level();
            auto max_level = this->mesh().max_level();
#endif
            if (min_level > 0)
            {
                min_level--;
            }
            if (min_level > 0)
            {
                min_level--;
            }

            for (std::size_t level = min_level; level <= max_level; ++level)
            {
                auto grid_level                               = this->domain().append_child("Grid");
                grid_level.append_attribute("Name")           = fmt::format("Level {}", level).data();
                grid_level.append_attribute("GridType")       = "Collection";
                grid_level.append_attribute("CollectionType") = "Spatial";

                if (this->options().by_mesh_id)
                {
                    for (std::size_t im = 0; im < this->derived_cast().nb_submesh(); ++im)
                    {
                        const auto& submesh = this->derived_cast().get_submesh(im);

                        std::string mesh_name = fmt::format("level_{}_{}", level, this->derived_cast().get_submesh_name(im));
                        std::string prefix    = fmt::format("/level/{}/mesh/{}", level, mesh_name);
                        this->save_on_mesh(grid_level, prefix, submesh[level], mesh_name);
                    }
                }
                else
                {
                    const auto& mesh = this->derived_cast().get_mesh();

                    std::string prefix = fmt::format("/level/{}/mesh", level);
                    this->save_on_mesh(grid_level, prefix, mesh[level], "mesh");
                }
            }
        }
        else
        {
            if (this->options().by_mesh_id)
            {
                for (std::size_t im = 0; im < this->derived_cast().nb_submesh(); ++im)
                {
                    const auto& submesh   = this->derived_cast().get_submesh(im);
                    std::string mesh_name = this->derived_cast().get_submesh_name(im);
                    std::string prefix    = fmt::format("/mesh/{}", mesh_name);

                    auto grid_mesh_id                         = this->domain().append_child("Grid");
                    grid_mesh_id.append_attribute("Name")     = mesh_name.data();
                    grid_mesh_id.append_attribute("GridType") = "Collection";

                    this->save_on_mesh(grid_mesh_id, prefix, submesh, mesh_name);
                }
            }
            else
            {
                const auto& mesh = this->derived_cast().get_mesh();

                std::string prefix = fmt::format("/mesh");
                this->save_on_mesh(this->domain(), prefix, mesh, "mesh");
            }
        }
    }

    template <class D, class Mesh, class... T>
    class SaveLevelCellArray : public SaveBase<D, Mesh, T...>
    {
      public:

        using base_class                 = SaveBase<D, Mesh, T...>;
        using derived_type               = D;
        using options_t                  = typename base_class::options_t;
        using mesh_t                     = typename base_class::mesh_t;
        static constexpr std::size_t dim = base_class::dim;

#ifdef SAMURAI_WITH_MPI
        SaveLevelCellArray(const fs::path& path,
                           const std::string& filename,
                           MPI_Comm comm,
                           const options_t& options,
                           const mesh_t& mesh,
                           const T&... fields);
#else
        SaveLevelCellArray(const fs::path& path, const std::string& filename, const options_t& options, const mesh_t& mesh, const T&... fields);
#endif

        void save();
    };

#ifdef SAMURAI_WITH_MPI
    template <class D, class Mesh, class... T>
    SAMURAI_INLINE SaveLevelCellArray<D, Mesh, T...>::SaveLevelCellArray(const fs::path& path,
                                                                         const std::string& filename,
                                                                         MPI_Comm comm,
                                                                         const options_t& options,
                                                                         const mesh_t& mesh,
                                                                         const T&... fields)
        : base_class(path, filename, comm, options, mesh, fields...)
    {
    }
#else
    template <class D, class Mesh, class... T>
    SAMURAI_INLINE SaveLevelCellArray<D, Mesh, T...>::SaveLevelCellArray(const fs::path& path,
                                                                         const std::string& filename,
                                                                         const options_t& options,
                                                                         const mesh_t& mesh,
                                                                         const T&... fields)
        : base_class(path, filename, options, mesh, fields...)
    {
    }
#endif

    template <class D, class Mesh, class... T>
    SAMURAI_INLINE void SaveLevelCellArray<D, Mesh, T...>::save()
    {
        if (this->options().by_mesh_id)
        {
            for (std::size_t im = 0; im < this->derived_cast().nb_submesh(); ++im)
            {
                const auto& submesh   = this->derived_cast().get_submesh(im);
                std::string mesh_name = this->derived_cast().get_submesh_name(im);
                std::string prefix    = fmt::format("/mesh/{}", mesh_name);

                auto grid_mesh_id                         = this->domain().append_child("Grid");
                grid_mesh_id.append_attribute("Name")     = mesh_name.data();
                grid_mesh_id.append_attribute("GridType") = "Collection";

                this->save_on_mesh(grid_mesh_id, prefix, submesh, mesh_name);
            }
        }
        else
        {
            const auto& mesh = this->derived_cast().get_mesh();

            std::string prefix = fmt::format("/mesh");
            this->save_on_mesh(this->domain(), prefix, mesh, "mesh");
        }
    }

#ifdef SAMURAI_WITH_MPI
    namespace detail
    {
        inline auto adjust_filename_for_comm(const std::string& filename, MPI_Comm comm)
        {
            mpi::communicator world;
            mpi::communicator comm_wrapped(comm, mpi::comm_create_kind::comm_duplicate);

            if (comm_wrapped.size() == 1 && world.size() > 1)
            {
                return fmt::format("{}_rank{}", filename, world.rank());
            }
            return filename;
        }
    }

    template <class D>
    Hdf5<D>::Hdf5(const fs::path& path, const std::string& filename, MPI_Comm comm)
        : m_mpi_comm(comm)
        , h5_file(create_h5file(path, detail::adjust_filename_for_comm(filename, comm), comm))
        , m_path(path)
        , m_filename(detail::adjust_filename_for_comm(filename, comm))
    {
        auto xdmf = m_doc.append_child("Xdmf");
        m_domain  = xdmf.append_child("Domain");
    }

    template <class D>
    HighFive::File Hdf5<D>::create_h5file(const fs::path& path, const std::string& filename, MPI_Comm comm)
    {
        HighFive::FileAccessProps fapl;
        // Only use collective metadata for communicators larger than SELF
        mpi::communicator comm_wrapped(comm, mpi::comm_create_kind::comm_duplicate);

        if (comm_wrapped.size() > 1)
        {
            fapl.add(HighFive::MPIOFileAccess{comm, MPI_INFO_NULL});
            fapl.add(HighFive::MPIOCollectiveMetadata{});
        }

        return HighFive::File(fmt::format("{}.h5", (path / filename).string()), HighFive::File::Overwrite, fapl);
    }

#else
    template <class D>
    Hdf5<D>::Hdf5(const fs::path& path, const std::string& filename)
        : h5_file(create_h5file(path, filename))
        , m_path(path)
        , m_filename(filename)
    {
        auto xdmf = m_doc.append_child("Xdmf");
        m_domain  = xdmf.append_child("Domain");
    }

    template <class D>
    HighFive::File Hdf5<D>::create_h5file(const fs::path& path, const std::string& filename)
    {
        return HighFive::File(fmt::format("{}.h5", (path / filename).string()), HighFive::File::Overwrite);
    }
#endif

    template <class D>
    SAMURAI_INLINE Hdf5<D>::~Hdf5()
    {
#ifdef SAMURAI_WITH_MPI
        mpi::communicator comm_wrapped(m_mpi_comm, mpi::comm_create_kind::comm_duplicate);
        if (comm_wrapped.rank() == 0)
#endif
        {
            m_doc.save_file(fmt::format("{}.xdmf", (m_path / m_filename).string()).data());
        }
    }

    template <class D>
    SAMURAI_INLINE pugi::xml_node& Hdf5<D>::domain()
    {
        return m_domain;
    }

    template <class D>
    template <class Submesh>
    SAMURAI_INLINE void
    Hdf5<D>::save_on_mesh(pugi::xml_node& grid_parent, const std::string& prefix, const Submesh& submesh, const std::string& mesh_name)
    {
        static constexpr std::size_t dim = derived_type_save::dim;
        std::size_t rank                 = 0;
        std::size_t size                 = 1;

        xt::xtensor<std::size_t, 2> local_connectivity;
        xt::xtensor<double, 2> local_coords;
        std::tie(local_coords, local_connectivity) = extract_coords_and_connectivity(submesh);

#ifdef SAMURAI_WITH_MPI
        mpi::communicator comm(m_mpi_comm, mpi::comm_create_kind::comm_duplicate);

        size = static_cast<std::size_t>(comm.size());
        rank = static_cast<std::size_t>(comm.rank());

        xt::xtensor<std::size_t, 1> connectivity_sizes = xt::empty<std::size_t>({size});
        xt::xtensor<std::size_t, 1> coords_sizes       = xt::empty<std::size_t>({size});

        mpi::all_gather(comm, local_connectivity.shape(0), connectivity_sizes.begin());
        mpi::all_gather(comm, local_coords.shape(0), coords_sizes.begin());
#else
        xt::xtensor_fixed<std::size_t, xt::xshape<1>> connectivity_sizes = {local_connectivity.shape(0)};
        xt::xtensor_fixed<std::size_t, xt::xshape<1>> coords_sizes       = {local_coords.shape(0)};
#endif

        std::vector<std::size_t> connectivity_cumsum(size + 1, 0);
        std::vector<std::size_t> coords_cumsum(size + 1, 0);
        for (std::size_t i = 0; i < size; ++i)
        {
            connectivity_cumsum[i + 1] += connectivity_cumsum[i] + connectivity_sizes[i];
        }

        for (std::size_t i = 0; i < size; ++i)
        {
            coords_cumsum[i + 1] += coords_cumsum[i] + coords_sizes[i];
        }

        if (coords_cumsum.back() != 0)
        {
            auto xfer_props = HighFive::DataTransferProps{};
            if (size == 1)
            {
                auto connectivity = h5_file.createDataSet<std::size_t>(
                    prefix + "/connectivity",
                    HighFive::DataSpace(std::vector<std::size_t>{connectivity_sizes[rank], 1 << dim}));
                auto coords = h5_file.createDataSet<double>(prefix + "/points",
                                                            HighFive::DataSpace(std::vector<std::size_t>{coords_sizes[rank], 3}));

                auto connectivity_slice = connectivity.select({connectivity_cumsum[rank], 0}, {connectivity_sizes[rank], 1 << dim});
                local_connectivity += coords_cumsum[rank];
                connectivity_slice.write_raw(local_connectivity.data(), HighFive::AtomicType<std::size_t>{}, xfer_props);

                auto coords_slice = coords.select({0, 0}, {coords_sizes[rank], 3});
                coords_slice.write_raw(local_coords.data(), HighFive::AtomicType<double>{}, xfer_props);
            }
            else
            {
#ifdef SAMURAI_WITH_MPI
                xfer_props.add(HighFive::UseCollectiveIO{});
#endif
                for (std::size_t r = 0; r < size; ++r)
                {
                    if (coords_sizes[r] != 0)
                    {
                        auto connectivity = h5_file.createDataSet<std::size_t>(
                            prefix + fmt::format("/rank_{}/connectivity", r),
                            HighFive::DataSpace(std::vector<std::size_t>{connectivity_sizes[r], 1 << dim}));
                        std::vector<std::size_t> conn_size(2, 0);
                        if (rank == r && connectivity_sizes[r] != 0)
                        {
                            conn_size = {connectivity_sizes[r], 1 << dim};
                        }
                        std::size_t* conn_ptr = (rank == r && connectivity_sizes[r] != 0) ? local_connectivity.data() : nullptr;

                        auto connectivity_slice = connectivity.select({0, 0}, conn_size);
                        connectivity_slice.write_raw(conn_ptr, HighFive::AtomicType<std::size_t>{}, xfer_props);

                        auto coords = h5_file.createDataSet<double>(prefix + fmt::format("/rank_{}/points", r),
                                                                    HighFive::DataSpace(std::vector<std::size_t>{coords_sizes[r], 3}));

                        std::vector<std::size_t> coord_size(2, 0);
                        double* coord_ptr = nullptr;
                        if (rank == r && coords_sizes[r] != 0)
                        {
                            coord_size = {coords_sizes[r], 3};
                            coord_ptr  = local_coords.data();
                        }
                        auto coords_slice = coords.select({0, 0}, coord_size);
                        coords_slice.write_raw(coord_ptr, HighFive::AtomicType<double>{}, xfer_props);
                    }
                }
            }

            auto grid = grid_parent.append_child("Grid");
            if (rank == 0)
            {
                if (size == 1)
                {
                    grid.append_attribute("Name") = mesh_name.data();

                    auto topo                                 = grid.append_child("Topology");
                    topo.append_attribute("TopologyType")     = element_type(derived_type_save::dim).c_str();
                    topo.append_attribute("NumberOfElements") = connectivity_sizes[rank];

                    auto topo_data                           = topo.append_child("DataItem");
                    topo_data.append_attribute("Dimensions") = connectivity_sizes[rank] * (1 << dim);
                    topo_data.append_attribute("Format")     = "HDF";
                    topo_data.text()                         = fmt::format("{}.h5:{}/connectivity", m_filename, prefix).data();

                    auto geom                             = grid.append_child("Geometry");
                    geom.append_attribute("GeometryType") = "XYZ";

                    auto geom_data                           = geom.append_child("DataItem");
                    geom_data.append_attribute("Dimensions") = coords_sizes[rank] * 3;
                    geom_data.append_attribute("Format")     = "HDF";
                    geom_data.text()                         = fmt::format("{}.h5:{}/points", m_filename, prefix).data();
                }
                else
                {
                    grid.append_attribute("GridType")       = "Collection";
                    grid.append_attribute("CollectionType") = "Spatial";
                    for (std::size_t irank = 0; irank < size; ++irank)
                    {
                        if (coords_sizes[irank] != 0)
                        {
                            auto subgrid                     = grid.append_child("Grid");
                            subgrid.append_attribute("Name") = fmt::format("{}_rank_{}", mesh_name, irank).data();
                            subgrid.append_attribute("Rank") = irank;

                            auto topo                                 = subgrid.append_child("Topology");
                            topo.append_attribute("TopologyType")     = element_type(derived_type_save::dim).c_str();
                            topo.append_attribute("NumberOfElements") = connectivity_sizes[irank];

                            auto topo_data                           = topo.append_child("DataItem");
                            topo_data.append_attribute("Dimensions") = connectivity_sizes[irank] * (1 << dim);
                            topo_data.append_attribute("Format")     = "HDF";
                            topo_data.text() = fmt::format("{}.h5:{}/rank_{}/connectivity", m_filename, prefix, irank).data();

                            auto geom                             = subgrid.append_child("Geometry");
                            geom.append_attribute("GeometryType") = "XYZ";

                            auto geom_data                           = geom.append_child("DataItem");
                            geom_data.append_attribute("Dimensions") = coords_sizes[irank] * 3;
                            geom_data.append_attribute("Format")     = "HDF";
                            geom_data.text() = fmt::format("{}.h5:{}/rank_{}/points", m_filename, prefix, irank).data();
                        }
                    }
                }
            }

            this->derived_cast().save_fields(grid, prefix, submesh);
        }
    }

    template <class D>
    template <class Submesh, class Field>
    SAMURAI_INLINE void Hdf5<D>::save_field(pugi::xml_node& grid, const std::string& prefix, const Submesh& submesh, const Field& field)
    {
        auto xfer_props = HighFive::DataTransferProps{};
#ifdef SAMURAI_WITH_MPI

        mpi::communicator comm(m_mpi_comm, mpi::comm_create_kind::comm_duplicate);

        std::size_t size = static_cast<std::size_t>(comm.size());
        std::size_t rank = static_cast<std::size_t>(comm.rank());

        xt::xtensor<std::size_t, 1> field_sizes = xt::empty<std::size_t>({size});

        if (size == 1)
        {
            field_sizes[0] = submesh.nb_cells();
        }
        else
        {
            xfer_props.add(HighFive::UseCollectiveIO{});
            mpi::all_gather(comm, submesh.nb_cells(), field_sizes.begin());
        }
#else
        std::size_t rank                                          = 0;
        std::size_t size                                          = 1;
        xt::xtensor_fixed<std::size_t, xt::xshape<1>> field_sizes = {submesh.nb_cells()};
#endif
        std::vector<std::size_t> field_cumsum(size + 1, 0);
        for (std::size_t i = 0; i < size; ++i)
        {
            field_cumsum[i + 1] += field_cumsum[i] + field_sizes[i];
        }

        for (std::size_t i = 0; i < field.n_comp; ++i)
        {
            std::string field_name;
            if constexpr (Field::n_comp == 1)
            {
                field_name = field.name();
            }
            else
            {
                field_name = fmt::format("{}_{}", field.name(), i);
            }

            if (size == 1)
            {
                auto local_data  = extract_data(field, submesh);
                std::string path = fmt::format("{}/fields/{}", prefix, field_name);

                auto data = h5_file.createDataSet<typename Field::value_type>(
                    path,
                    HighFive::DataSpace(std::vector<std::size_t>{field_cumsum.back()}));

                auto data_slice = data.select({field_cumsum[rank]}, {field_sizes[rank]});
                data_slice.write_raw(xt::eval(xt::view(local_data, xt::all(), i)).data(),
                                     HighFive::AtomicType<typename Field::value_type>{},
                                     xfer_props);

                auto attribute                       = grid.append_child("Attribute");
                attribute.append_attribute("Name")   = field_name.data();
                attribute.append_attribute("Center") = "Cell";

                auto dataitem                           = attribute.append_child("DataItem");
                dataitem.append_attribute("Dimensions") = field_cumsum.back();
                dataitem.append_attribute("Format")     = "HDF";
                dataitem.append_attribute("Precision")  = "8";
                dataitem.text()                         = fmt::format("{}.h5:{}", m_filename, path).data();
            }
            else
            {
                auto local_data = extract_data(field, submesh);
                xt::xtensor<typename Field::value_type, 1> data_tmp;
                for (std::size_t irank = 0; irank < size; ++irank)
                {
                    if (field_sizes[irank] != 0)
                    {
                        std::string path = fmt::format("{}/rank_{}/fields/{}", prefix, irank, field_name);
                        auto data        = h5_file.createDataSet<typename Field::value_type>(
                            path,
                            HighFive::DataSpace(std::vector<std::size_t>{field_sizes[irank]}));

                        std::vector<std::size_t> data_size(1, 0);
                        typename Field::value_type* data_ptr = nullptr;

                        if (rank == irank)
                        {
                            data_tmp     = xt::eval(xt::view(local_data, xt::all(), i));
                            data_ptr     = data_tmp.data();
                            data_size[0] = field_sizes[irank];
                        }
                        auto data_slice = data.select({0}, data_size);
                        data_slice.write_raw(data_ptr, HighFive::AtomicType<typename Field::value_type>{}, xfer_props);
                    }
                }
                if (rank == 0)
                {
                    for (pugi::xml_node subgrid : grid.children("Grid"))
                    {
                        std::size_t irank = subgrid.attribute("Rank").as_uint();
                        std::string path  = fmt::format("{}/rank_{}/fields/{}", prefix, irank, field_name);

                        auto attribute                       = subgrid.append_child("Attribute");
                        attribute.append_attribute("Name")   = field_name.data();
                        attribute.append_attribute("Center") = "Cell";

                        auto dataitem                           = attribute.append_child("DataItem");
                        dataitem.append_attribute("Dimensions") = field_sizes[irank];
                        dataitem.append_attribute("Format")     = "HDF";
                        dataitem.text()                         = fmt::format("{}.h5:{}", m_filename, path).data();
                    }
                }
            }
        }
    }

    template <class D>
    SAMURAI_INLINE auto Hdf5<D>::derived_cast() & noexcept -> derived_type_save&
    {
        return *static_cast<derived_type_save*>(this);
    }

    template <class D>
    SAMURAI_INLINE auto Hdf5<D>::derived_cast() const& noexcept -> const derived_type_save&
    {
        return *static_cast<const derived_type_save*>(this);
    }

    template <class D>
    SAMURAI_INLINE auto Hdf5<D>::derived_cast() && noexcept -> derived_type_save
    {
        return *static_cast<derived_type_save*>(this);
    }

    template <class Mesh, class... T>
    class Hdf5_CellArray : public SaveCellArray<Hdf5_CellArray<Mesh, T...>, Mesh, T...>
    {
      public:

        using base_type                  = SaveCellArray<Hdf5_CellArray<Mesh, T...>, Mesh, T...>;
        using options_t                  = typename base_type::options_t;
        using mesh_t                     = Mesh;
        static constexpr std::size_t dim = mesh_t::dim;

#ifdef SAMURAI_WITH_MPI
        Hdf5_CellArray(const fs::path& path,
                       const std::string& filename,
                       MPI_Comm comm,
                       const options_t& options,
                       const Mesh& mesh,
                       const T&... fields)
            : base_type(path, filename, comm, options, mesh, fields...)
        {
        }
#else
        Hdf5_CellArray(const fs::path& path, const std::string& filename, const options_t& options, const Mesh& mesh, const T&... fields)
            : base_type(path, filename, options, mesh, fields...)
        {
        }
#endif

        const mesh_t& get_mesh() const
        {
            return this->mesh();
        }

        const mesh_t& get_submesh(std::size_t) const
        {
            return this->mesh();
        }

        std::string get_submesh_name(std::size_t) const
        {
            return "cell_array";
        }

        std::size_t nb_submesh() const
        {
            return 1;
        }
    };

    template <class Mesh, class... T>
    class Hdf5_mesh_base_level : public SaveLevelCellArray<Hdf5_mesh_base_level<Mesh, T...>, Mesh, T...>
    {
      public:

        using base_type                  = SaveLevelCellArray<Hdf5_mesh_base_level<Mesh, T...>, Mesh, T...>;
        using options_t                  = typename base_type::options_t;
        using mesh_t                     = Mesh;
        using mesh_id_t                  = typename mesh_t::mesh_id_t;
        using ca_type                    = typename mesh_t::ca_type;
        static constexpr std::size_t dim = mesh_t::dim;

#ifdef SAMURAI_WITH_MPI
        Hdf5_mesh_base_level(const fs::path& path,
                             const std::string& filename,
                             MPI_Comm comm,
                             const options_t& options,
                             const Mesh& mesh,
                             const T&... fields)
            : base_type(path, filename, comm, options, mesh, fields...)
        {
        }
#else
        Hdf5_mesh_base_level(const fs::path& path, const std::string& filename, const options_t& options, const Mesh& mesh, const T&... fields)
            : base_type(path, filename, options, mesh, fields...)
        {
        }
#endif

        const ca_type& get_mesh() const
        {
            return this->mesh()[mesh_id_t::cells];
        }

        const ca_type& get_submesh(std::size_t i) const
        {
            return this->mesh()[static_cast<mesh_id_t>(i)];
        }

        std::string get_submesh_name(std::size_t i) const
        {
            return fmt::format("{}", static_cast<mesh_id_t>(i));
        }

        std::size_t nb_submesh() const
        {
            return static_cast<std::size_t>(mesh_id_t::count);
        }
    };

    template <class Mesh, class... T>
    class Hdf5_LevelCellArray : public SaveLevelCellArray<Hdf5_LevelCellArray<Mesh, T...>, Mesh, T...>
    {
      public:

        using base_type                  = SaveLevelCellArray<Hdf5_LevelCellArray<Mesh, T...>, Mesh, T...>;
        using options_t                  = typename base_type::options_t;
        using ca_type                    = Mesh;
        static constexpr std::size_t dim = ca_type::dim;

#ifdef SAMURAI_WITH_MPI
        Hdf5_LevelCellArray(const fs::path& path,
                            const std::string& filename,
                            MPI_Comm comm,
                            const options_t& options,
                            const Mesh& mesh,
                            const T&... fields)
            : base_type(path, filename, comm, options, mesh, fields...)
        {
        }
#else
        Hdf5_LevelCellArray(const fs::path& path, const std::string& filename, const options_t& options, const Mesh& mesh, const T&... fields)
            : base_type(path, filename, options, mesh, fields...)
        {
        }
#endif

        const ca_type& get_mesh() const
        {
            return this->mesh();
        }

        const ca_type& get_submesh(std::size_t) const
        {
            return this->mesh();
        }

        std::string get_submesh_name(std::size_t) const
        {
            return fmt::format("LevelCellArray");
        }

        std::size_t nb_submesh() const
        {
            return 1;
        }
    };

    template <class Mesh, class... T>
    class Hdf5_mesh_base : public SaveCellArray<Hdf5_mesh_base<Mesh, T...>, Mesh, T...>
    {
      public:

        using base_type                  = SaveCellArray<Hdf5_mesh_base<Mesh, T...>, Mesh, T...>;
        using options_t                  = typename base_type::options_t;
        using mesh_t                     = Mesh;
        using mesh_id_t                  = typename Mesh::mesh_id_t;
        using ca_type                    = typename mesh_t::ca_type;
        static constexpr std::size_t dim = mesh_t::dim;

#ifdef SAMURAI_WITH_MPI
        Hdf5_mesh_base(const fs::path& path,
                       const std::string& filename,
                       MPI_Comm comm,
                       const options_t& options,
                       const Mesh& mesh,
                       const T&... fields)
            : base_type(path, filename, comm, options, mesh, fields...)
        {
        }
#else
        Hdf5_mesh_base(const fs::path& path, const std::string& filename, const options_t& options, const Mesh& mesh, const T&... fields)
            : base_type(path, filename, options, mesh, fields...)
        {
        }
#endif
        const ca_type& get_mesh() const
        {
            return this->mesh()[mesh_id_t::cells];
        }

        const ca_type& get_submesh(std::size_t i) const
        {
            return this->mesh()[static_cast<mesh_id_t>(i)];
        }

        std::string get_submesh_name(std::size_t i) const
        {
            return fmt::format("{}", static_cast<mesh_id_t>(i));
        }

        std::size_t nb_submesh() const
        {
            return static_cast<std::size_t>(mesh_id_t::count);
        }
    };

    namespace detail
    {
        template <class mesh_t>
            requires std::is_base_of_v<Mesh_base<mesh_t, typename mesh_t::config>, mesh_t>
        const auto& get_all_cells(const mesh_t& mesh)
        {
            using mesh_id_t = typename mesh_t::mesh_id_t;
            return mesh[mesh_id_t::reference];
        }

        template <class mesh_t>
        const auto& get_all_cells(const mesh_t& mesh)
        {
            return mesh;
        }

        template <class mesh_t, class... T>
        struct hdf5_mesh
        {
            using type = Hdf5_mesh_base<mesh_t, T...>;
        };

        template <class config_t, class... T>
        struct hdf5_mesh<UniformMesh<config_t>, T...>
        {
            using type = Hdf5_mesh_base_level<UniformMesh<config_t>, T...>;
        };

        template <std::size_t dim, class TInterval, class... T>
        struct hdf5_mesh<LevelCellArray<dim, TInterval>, T...>
        {
            using type = Hdf5_LevelCellArray<LevelCellArray<dim, TInterval>, T...>;
        };

        template <std::size_t dim, class TInterval, std::size_t max_size, class... T>
        struct hdf5_mesh<CellArray<dim, TInterval, max_size>, T...>
        {
            using type = Hdf5_CellArray<CellArray<dim, TInterval, max_size>, T...>;
        };

        template <class D, class... T>
        using hdf5_mesh_t = typename hdf5_mesh<D, T...>::type;
    }

#ifdef SAMURAI_WITH_MPI
    template <class mesh_t, class... T>
        requires(mesh_like<mesh_t>)
    void save(const fs::path& path,
              const std::string& filename,
              MPI_Comm comm,
              const Hdf5Options<mesh_t>& options,
              const mesh_t& mesh,
              const T&... fields)
    {
        static constexpr std::size_t dim = mesh_t::dim;
        times::timers.start("data saving");

        if (!fs::exists(path))
        {
            fs::create_directory(path);
        }

        if (args::save_debug_fields)
        {
            const auto& mesh_ref = detail::get_all_cells(mesh);

            auto index_field = make_vector_field<int, dim>("indices", mesh);
            auto coord_field = make_vector_field<double, dim>("coordinates", mesh);
            auto level_field = make_scalar_field<std::size_t>("levels", mesh);

            using hdf5_t = detail::hdf5_mesh_t<mesh_t, decltype(index_field), decltype(coord_field), decltype(level_field), T...>;

            for_each_cell(mesh_ref,
                          [&](auto& cell)
                          {
                              index_field[cell] = cell.indices;
                              coord_field[cell] = cell.center();
                              level_field[cell] = cell.level;
                          });

            auto h5 = hdf5_t(path, filename, comm, options, mesh, index_field, coord_field, level_field, fields...);
            h5.save();
        }
        else
        {
            using hdf5_t = detail::hdf5_mesh_t<mesh_t, T...>;
            auto h5      = hdf5_t(path, filename, comm, options, mesh, fields...);
            h5.save();
        }
        times::timers.stop("data saving");
    }

    template <class mesh_t, class... T>
        requires(mesh_like<mesh_t>)
    void save(const fs::path& path, const std::string& filename, MPI_Comm comm, const mesh_t& mesh, const T&... fields)
    {
        save(path, filename, comm, Hdf5Options<mesh_t>{}, mesh, fields...);
    }

    template <class mesh_t, class... T>
        requires(mesh_like<mesh_t>)
    void save(const std::string& filename, MPI_Comm comm, const Hdf5Options<mesh_t>& options, const mesh_t& mesh, const T&... fields)
    {
        save(fs::current_path(), filename, comm, options, mesh, fields...);
    }

    template <class mesh_t, class... T>
        requires(mesh_like<mesh_t>)
    void save(const std::string& filename, MPI_Comm comm, const mesh_t& mesh, const T&... fields)
    {
        save(fs::current_path(), filename, comm, Hdf5Options<mesh_t>{}, mesh, fields...);
    }
#endif

    template <class mesh_t, class... T>
        requires(mesh_like<mesh_t>)
    void save(const fs::path& path, const std::string& filename, const Hdf5Options<mesh_t>& options, const mesh_t& mesh, const T&... fields)
    {
        static constexpr std::size_t dim = mesh_t::dim;
        times::timers.start("data saving");

        if (!fs::exists(path))
        {
            fs::create_directory(path);
        }

        if (args::save_debug_fields)
        {
            const auto& mesh_ref = detail::get_all_cells(mesh);

            auto index_field = make_vector_field<int, dim>("indices", mesh);
            auto coord_field = make_vector_field<double, dim>("coordinates", mesh);
            auto level_field = make_scalar_field<std::size_t>("levels", mesh);

            using hdf5_t = detail::hdf5_mesh_t<mesh_t, decltype(index_field), decltype(coord_field), decltype(level_field), T...>;

            for_each_cell(mesh_ref,
                          [&](auto& cell)
                          {
                              index_field[cell] = cell.indices;
                              coord_field[cell] = cell.center();
                              level_field[cell] = cell.level;
                          });

#ifdef SAMURAI_WITH_MPI
            auto h5 = hdf5_t(path, filename, MPI_COMM_WORLD, options, mesh, index_field, coord_field, level_field, fields...);
#else
            auto h5 = hdf5_t(path, filename, options, mesh, index_field, coord_field, level_field, fields...);
#endif
            h5.save();
        }
        else
        {
            using hdf5_t = detail::hdf5_mesh_t<mesh_t, T...>;
#ifdef SAMURAI_WITH_MPI
            auto h5 = hdf5_t(path, filename, MPI_COMM_WORLD, options, mesh, fields...);
#else
            auto h5 = hdf5_t(path, filename, options, mesh, fields...);
#endif
            h5.save();
        }
        times::timers.stop("data saving");
    }

    template <class mesh_t, class... T>
        requires(mesh_like<mesh_t>)
    void save(const fs::path& path, const std::string& filename, const mesh_t& mesh, const T&... fields)
    {
        save(path, filename, Hdf5Options<mesh_t>{}, mesh, fields...);
    }

    template <class mesh_t, class... T>
        requires(mesh_like<mesh_t>)
    void save(const std::string& filename, const Hdf5Options<mesh_t>& options, const mesh_t& mesh, const T&... fields)
    {
        save(fs::current_path(), filename, options, mesh, fields...);
    }

    template <class mesh_t, class... T>
        requires(mesh_like<mesh_t>)
    void save(const std::string& filename, const mesh_t& mesh, const T&... fields)
    {
        save(fs::current_path(), filename, Hdf5Options<mesh_t>{}, mesh, fields...);
    }

    template <class mesh_t, class... T>
        requires(mesh_like<mesh_t>)
    void
    local_save(const fs::path& path, const std::string& filename, const Hdf5Options<mesh_t>& options, const mesh_t& mesh, const T&... fields)
    {
#ifdef SAMURAI_WITH_MPI
        save(path, filename, MPI_COMM_SELF, options, mesh, fields...);
#else
        save(path, filename, options, mesh, fields...);
#endif
    }

    template <class mesh_t, class... T>
        requires(mesh_like<mesh_t>)
    void local_save(const fs::path& path, const std::string& filename, const mesh_t& mesh, const T&... fields)
    {
#ifdef SAMURAI_WITH_MPI
        save(path, filename, MPI_COMM_SELF, Hdf5Options<mesh_t>{}, mesh, fields...);
#else
        save(path, filename, Hdf5Options<mesh_t>{}, mesh, fields...);
#endif
    }

    template <class mesh_t, class... T>
        requires(mesh_like<mesh_t>)
    void local_save(const std::string& filename, const Hdf5Options<mesh_t>& options, const mesh_t& mesh, const T&... fields)
    {
#ifdef SAMURAI_WITH_MPI
        save(fs::current_path(), filename, MPI_COMM_SELF, options, mesh, fields...);
#else
        save(fs::current_path(), filename, options, mesh, fields...);
#endif
    }

    template <class mesh_t, class... T>
        requires(mesh_like<mesh_t>)
    void local_save(const std::string& filename, const mesh_t& mesh, const T&... fields)
    {
#ifdef SAMURAI_WITH_MPI
        save(fs::current_path(), filename, Hdf5Options<mesh_t>{}, mesh, fields...);
#else
        save(fs::current_path(), filename, Hdf5Options<mesh_t>{}, mesh, fields...);
#endif
    }

} // namespace samurai
