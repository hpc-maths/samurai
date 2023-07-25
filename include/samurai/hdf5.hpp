// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <array>
#include <fstream>
#include <functional>
#include <string>
#include <type_traits>

#include <filesystem>
namespace fs = std::filesystem;

#ifndef H5_USE_XTENSOR
#define H5_USE_XTENSOR
#endif

#include <highfive/H5Easy.hpp>
#include <pugixml.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

#include <fmt/core.h>

#include "algorithm.hpp"
#include "cell.hpp"
#include "utils.hpp"

namespace samurai
{
    inline std::string element_type(std::size_t dim)
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

    inline auto get_element(std::integral_constant<std::size_t, 1>)
    {
        return std::array<double, 2>{
            {0, 1}
        };
    }

    inline auto get_element(std::integral_constant<std::size_t, 2>)
    {
        return std::array<xt::xtensor_fixed<std::size_t, xt::xshape<2>>, 4>{
            {{0, 0}, {1, 0}, {1, 1}, {0, 1}}
        };
    }

    inline auto get_element(std::integral_constant<std::size_t, 3>)
    {
        return std::array<xt::xtensor_fixed<std::size_t, xt::xshape<3>>, 8>{
            {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}}
        };
    }

    template <class Field, class SubMesh>
    auto extract_data(const Field& field, const SubMesh& submesh)
    {
        std::array<std::size_t, 2> shape = {submesh.nb_cells(), field.size};
        xt::xtensor<typename Field::value_type, 2> data(shape);
        std::size_t index = 0;
        for_each_cell(submesh,
                      [&](auto cell)
                      {
                          xt::view(data, index) = field[cell.index];
                          index++;
                      });

        return data;
    }

    template <class Mesh>
    auto extract_coords_and_connectivity(const Mesh& mesh)
    {
        static constexpr std::size_t dim = Mesh::dim;
        std::size_t nb_cells             = mesh.nb_cells();

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

        Hdf5(const fs::path& path, const std::string& filename);

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
        inline void save_field(pugi::xml_node& grid, const std::string& prefix, const Submesh& submesh, const Field& field);

      private:

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

        SaveBase(const fs::path& path, const std::string& filename, const options_t& options, const Mesh& mesh, const T&... fields);

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

    template <class D, class Mesh, class... T>
    inline SaveBase<D, Mesh, T...>::SaveBase(const fs::path& path,
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

    template <class D, class Mesh, class... T>
    template <class Submesh>
    inline void SaveBase<D, Mesh, T...>::save_fields(pugi::xml_node& grid, const std::string& prefix, const Submesh& submesh)
    {
        save_fields_impl(grid, prefix, submesh, std::make_index_sequence<sizeof...(T)>());
    }

    template <class D, class Mesh, class... T>
    template <class Submesh, std::size_t... I>
    inline void SaveBase<D, Mesh, T...>::save_fields_impl(pugi::xml_node& grid,
                                                          const std::string& prefix,
                                                          const Submesh& submesh,
                                                          std::index_sequence<I...>)
    {
        (void)std::initializer_list<int>{(this->save_field(grid, prefix, submesh, std::get<I>(m_fields)), 0)...};
    }

    template <class D, class Mesh, class... T>
    inline void SaveBase<D, Mesh, T...>::save()
    {
        this->derived_cast().save();
    }

    template <class D, class Mesh, class... T>
    inline auto SaveBase<D, Mesh, T...>::derived_cast() & noexcept -> derived_type&
    {
        return *static_cast<derived_type*>(this);
    }

    template <class D, class Mesh, class... T>
    inline auto SaveBase<D, Mesh, T...>::derived_cast() const& noexcept -> const derived_type&
    {
        return *static_cast<const derived_type*>(this);
    }

    template <class D, class Mesh, class... T>
    inline auto SaveBase<D, Mesh, T...>::derived_cast() && noexcept -> derived_type
    {
        return *static_cast<derived_type*>(this);
    }

    template <class D, class Mesh, class... T>
    inline auto SaveBase<D, Mesh, T...>::mesh() const -> const mesh_t&
    {
        return m_mesh;
    }

    template <class D, class Mesh, class... T>
    inline auto SaveBase<D, Mesh, T...>::options() const -> const options_t&
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

        SaveCellArray(const fs::path& path, const std::string& filename, const options_t& options, const mesh_t& mesh, const T&... fields);
        void save();
    };

    template <class D, class Mesh, class... T>
    inline SaveCellArray<D, Mesh, T...>::SaveCellArray(const fs::path& path,
                                                       const std::string& filename,
                                                       const options_t& options,
                                                       const mesh_t& mesh,
                                                       const T&... fields)
        : base_class(path, filename, options, mesh, fields...)
    {
    }

    template <class D, class Mesh, class... T>
    inline void SaveCellArray<D, Mesh, T...>::save()
    {
        if (this->options().by_level)
        {
            auto min_level = this->mesh().min_level();
            if (min_level > 0)
            {
                min_level--;
            }
            auto max_level = this->mesh().max_level();
            for (std::size_t level = min_level; level <= max_level; ++level)
            {
                auto grid_level                         = this->domain().append_child("Grid");
                grid_level.append_attribute("Name")     = fmt::format("Level {}", level).data();
                grid_level.append_attribute("GridType") = "Collection";

                if (this->options().by_mesh_id)
                {
                    for (std::size_t im = 0; im < this->derived_cast().nb_submesh(); ++im)
                    {
                        auto& submesh = this->derived_cast().get_submesh(im);

                        if (!submesh[level].empty())
                        {
                            std::string mesh_name = this->derived_cast().get_submesh_name(im);
                            std::string prefix    = fmt::format("/level/{}/mesh/{}", level, mesh_name);
                            this->save_on_mesh(grid_level, prefix, submesh[level], mesh_name);
                        }
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

        SaveLevelCellArray(const fs::path& path, const std::string& filename, const options_t& options, const mesh_t& mesh, const T&... fields);

        void save();
    };

    template <class D, class Mesh, class... T>
    inline SaveLevelCellArray<D, Mesh, T...>::SaveLevelCellArray(const fs::path& path,
                                                                 const std::string& filename,
                                                                 const options_t& options,
                                                                 const mesh_t& mesh,
                                                                 const T&... fields)
        : base_class(path, filename, options, mesh, fields...)
    {
    }

    template <class D, class Mesh, class... T>
    inline void SaveLevelCellArray<D, Mesh, T...>::save()
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

    template <class D>
    inline Hdf5<D>::Hdf5(const fs::path& path, const std::string& filename)
        : h5_file(path.string() + '/' + filename + ".h5", HighFive::File::Overwrite)
        , m_path(path)
        , m_filename(filename)
    {
        m_doc.append_child(pugi::node_doctype).set_value("Xdmf SYSTEM \"Xdmf.dtd\"");
        auto xdmf = m_doc.append_child("Xdmf");
        m_domain  = xdmf.append_child("Domain");
    }

    template <class D>
    inline Hdf5<D>::~Hdf5()
    {
        m_doc.save_file(fmt::format("{}.xdmf", (m_path / m_filename).string()).data());
    }

    template <class D>
    inline pugi::xml_node& Hdf5<D>::domain()
    {
        return m_domain;
    }

    template <class D>
    template <class Submesh>
    inline void
    Hdf5<D>::save_on_mesh(pugi::xml_node& grid_parent, const std::string& prefix, const Submesh& submesh, const std::string& mesh_name)
    {
        xt::xtensor<std::size_t, 2> connectivity;
        xt::xtensor<double, 2> coords;
        std::tie(coords, connectivity) = extract_coords_and_connectivity(submesh);

        H5Easy::dump(h5_file, prefix + "/connectivity", connectivity);
        H5Easy::dump(h5_file, prefix + "/points", coords);

        auto grid                     = grid_parent.append_child("Grid");
        grid.append_attribute("Name") = mesh_name.data();

        auto topo                                 = grid.append_child("Topology");
        topo.append_attribute("TopologyType")     = element_type(derived_type_save::dim).c_str();
        topo.append_attribute("NumberOfElements") = connectivity.shape()[0];

        auto topo_data                           = topo.append_child("DataItem");
        topo_data.append_attribute("Dimensions") = connectivity.size();
        topo_data.append_attribute("Format")     = "HDF";
        topo_data.text()                         = fmt::format("{}.h5:{}/connectivity", m_filename, prefix).data();

        auto geom                             = grid.append_child("Geometry");
        geom.append_attribute("GeometryType") = "XYZ";

        auto geom_data                           = geom.append_child("DataItem");
        geom_data.append_attribute("Dimensions") = coords.size();
        geom_data.append_attribute("Format")     = "HDF";
        geom_data.text()                         = fmt::format("{}.h5:{}/points", m_filename, prefix).data();

        this->derived_cast().save_fields(grid, prefix, submesh);
    }

    template <class D>
    template <class Submesh, class Field>
    inline void Hdf5<D>::save_field(pugi::xml_node& grid, const std::string& prefix, const Submesh& submesh, const Field& field)
    {
        auto data = extract_data(field, submesh);

        for (std::size_t i = 0; i < field.size; ++i)
        {
            std::string field_name;
            if constexpr (Field::size == 1)
            {
                field_name = field.name();
            }
            else
            {
                field_name = fmt::format("{}_{}", field.name(), i);
            }
            std::string path = fmt::format("{}/fields/{}", prefix, field_name);
            H5Easy::dump(h5_file, path, xt::eval(xt::view(data, xt::all(), i)));

            auto attribute                       = grid.append_child("Attribute");
            attribute.append_attribute("Name")   = field_name.data();
            attribute.append_attribute("Center") = "Cell";

            auto dataitem                           = attribute.append_child("DataItem");
            dataitem.append_attribute("Dimensions") = submesh.nb_cells();
            dataitem.append_attribute("Format")     = "HDF";
            dataitem.append_attribute("Precision")  = "8";
            dataitem.text()                         = fmt::format("{}.h5:{}", m_filename, path).data();
        }
    }

    template <class D>
    inline auto Hdf5<D>::derived_cast() & noexcept -> derived_type_save&
    {
        return *static_cast<derived_type_save*>(this);
    }

    template <class D>
    inline auto Hdf5<D>::derived_cast() const& noexcept -> const derived_type_save&
    {
        return *static_cast<const derived_type_save*>(this);
    }

    template <class D>
    inline auto Hdf5<D>::derived_cast() && noexcept -> derived_type_save
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

        Hdf5_CellArray(const fs::path& path, const std::string& filename, const options_t& options, const Mesh& mesh, const T&... fields)
            : base_type(path, filename, options, mesh, fields...)
        {
        }

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

        Hdf5_mesh_base_level(const fs::path& path, const std::string& filename, const options_t& options, const Mesh& mesh, const T&... fields)
            : base_type(path, filename, options, mesh, fields...)
        {
        }

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

        Hdf5_LevelCellArray(const fs::path& path, const std::string& filename, const options_t& options, const Mesh& mesh, const T&... fields)
            : base_type(path, filename, options, mesh, fields...)
        {
        }

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

        Hdf5_mesh_base(const fs::path& path, const std::string& filename, const options_t& options, const Mesh& mesh, const T&... fields)
            : base_type(path, filename, options, mesh, fields...)
        {
        }

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

    template <std::size_t dim, class TInterval, class... T>
    void save(const fs::path& path, const std::string& filename, const LevelCellArray<dim, TInterval>& mesh, const T&... fields)
    {
        using hdf5_t = Hdf5_LevelCellArray<LevelCellArray<dim, TInterval>, T...>;
        auto h5      = hdf5_t(path, filename, {}, mesh, fields...);
        h5.save();
    }

    template <std::size_t dim, class TInterval, class... T>
    void save(const std::string& filename, const LevelCellArray<dim, TInterval>& mesh, const T&... fields)
    {
        using hdf5_t = Hdf5_LevelCellArray<LevelCellArray<dim, TInterval>, T...>;
        auto h5      = hdf5_t(fs::current_path(), filename, {}, mesh, fields...);
        h5.save();
    }

    template <std::size_t dim, class TInterval, class... T>
    void save(const fs::path& path,
              const std::string& filename,
              const Hdf5Options<LevelCellArray<dim, TInterval>>& options,
              const LevelCellArray<dim, TInterval>& mesh,
              const T&... fields)
    {
        using hdf5_t = Hdf5_LevelCellArray<LevelCellArray<dim, TInterval>, T...>;
        auto h5      = hdf5_t(path, filename, options, mesh, fields...);
        h5.save();
    }

    template <std::size_t dim, class TInterval, class... T>
    void save(const std::string& filename,
              const Hdf5Options<LevelCellArray<dim, TInterval>>& options,
              const LevelCellArray<dim, TInterval>& mesh,
              const T&... fields)
    {
        using hdf5_t = Hdf5_LevelCellArray<LevelCellArray<dim, TInterval>, T...>;
        auto h5      = hdf5_t(fs::current_path(), filename, options, mesh, fields...);
        h5.save();
    }

    template <std::size_t dim, class TInterval, std::size_t max_size, class... T>
    void save(const fs::path& path, const std::string& filename, const CellArray<dim, TInterval, max_size>& mesh, const T&... fields)
    {
        using hdf5_t = Hdf5_CellArray<CellArray<dim, TInterval, max_size>, T...>;
        auto h5      = hdf5_t(path, filename, {}, mesh, fields...);
        h5.save();
    }

    template <std::size_t dim, class TInterval, std::size_t max_size, class... T>
    void save(const std::string& filename, const CellArray<dim, TInterval, max_size>& mesh, const T&... fields)
    {
        using hdf5_t = Hdf5_CellArray<CellArray<dim, TInterval, max_size>, T...>;
        auto h5      = hdf5_t(fs::current_path(), filename, {}, mesh, fields...);
        h5.save();
    }

    template <std::size_t dim, class TInterval, std::size_t max_size, class... T>
    void save(const fs::path& path,
              const std::string& filename,
              const Hdf5Options<CellArray<dim, TInterval, max_size>>& options,
              const CellArray<dim, TInterval, max_size>& mesh,
              const T&... fields)
    {
        using hdf5_t = Hdf5_CellArray<CellArray<dim, TInterval, max_size>, T...>;
        auto h5      = hdf5_t(path, filename, options, mesh, fields...);
        h5.save();
    }

    template <std::size_t dim, class TInterval, std::size_t max_size, class... T>
    void save(const std::string& filename,
              const Hdf5Options<CellArray<dim, TInterval, max_size>>& options,
              const CellArray<dim, TInterval, max_size>& mesh,
              const T&... fields)
    {
        using hdf5_t = Hdf5_CellArray<CellArray<dim, TInterval, max_size>, T...>;
        auto h5      = hdf5_t(fs::current_path(), filename, options, mesh, fields...);
        h5.save();
    }

    template <class D, class Config, class... T>
    void save(const fs::path& path, const std::string& filename, const Mesh_base<D, Config>& mesh, const T&... fields)
    {
        using hdf5_t = Hdf5_mesh_base<Mesh_base<D, Config>, T...>;
        auto h5      = hdf5_t(path, filename, {}, mesh, fields...);
        h5.save();
    }

    template <class D, class Config, class... T>
    void save(const std::string& filename, const Mesh_base<D, Config>& mesh, const T&... fields)
    {
        using hdf5_t = Hdf5_mesh_base<Mesh_base<D, Config>, T...>;
        auto h5      = hdf5_t(fs::current_path(), filename, {}, mesh, fields...);
        h5.save();
    }

    template <class D, class Config, class... T>
    void save(const fs::path& path,
              const std::string& filename,
              const Hdf5Options<Mesh_base<D, Config>>& options,
              const Mesh_base<D, Config>& mesh,
              const T&... fields)
    {
        using hdf5_t = Hdf5_mesh_base<Mesh_base<D, Config>, T...>;
        auto h5      = hdf5_t(path, filename, options, mesh, fields...);
        h5.save();
    }

    template <class D, class Config, class... T>
    void
    save(const std::string& filename, const Hdf5Options<Mesh_base<D, Config>>& options, const Mesh_base<D, Config>& mesh, const T&... fields)
    {
        using hdf5_t = Hdf5_mesh_base<Mesh_base<D, Config>, T...>;
        auto h5      = hdf5_t(fs::current_path(), filename, options, mesh, fields...);
        h5.save();
    }

    template <class Config, class... T>
    void save(const fs::path& path, const std::string& filename, const UniformMesh<Config>& mesh, const T&... fields)
    {
        using hdf5_t = Hdf5_mesh_base_level<UniformMesh<Config>, T...>;
        auto h5      = hdf5_t(path, filename, {}, mesh, fields...);
        h5.save();
    }

    template <class Config, class... T>
    void save(const std::string& filename, const UniformMesh<Config>& mesh, const T&... fields)
    {
        using hdf5_t = Hdf5_mesh_base_level<UniformMesh<Config>, T...>;
        auto h5      = hdf5_t(fs::current_path(), filename, {}, mesh, fields...);
        h5.save();
    }

    template <class Config, class... T>
    void save(const fs::path& path,
              const std::string& filename,
              const Hdf5Options<UniformMesh<Config>>& options,
              const UniformMesh<Config>& mesh,
              const T&... fields)
    {
        using hdf5_t = Hdf5_mesh_base_level<UniformMesh<Config>, T...>;
        auto h5      = hdf5_t(path, filename, options, mesh, fields...);
        h5.save();
    }

    template <class Config, class... T>
    void
    save(const std::string& filename, const Hdf5Options<UniformMesh<Config>>& options, const UniformMesh<Config>& mesh, const T&... fields)
    {
        using hdf5_t = Hdf5_mesh_base_level<UniformMesh<Config>, T...>;
        auto h5      = hdf5_t(fs::current_path(), filename, options, mesh, fields...);
        h5.save();
    }
} // namespace samurai
