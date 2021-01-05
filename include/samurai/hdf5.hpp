#pragma once

#include <fstream>
#include <functional>
#include <string>
#include <type_traits>

#include <pugixml.hpp>
#include <xtensor-io/xhighfive.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

#include <fmt/core.h>

#include "cell.hpp"
#include "algorithm.hpp"
#include "mr/mesh_type.hpp"
#include "mesh.hpp"
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

    template<std::size_t dim>
    auto get_element(std::integral_constant<std::size_t, dim>);

    inline auto get_element(std::integral_constant<std::size_t, 1>)
    {
        return std::array<double, 2> {{ 0, 1 }};
    }

    inline auto get_element(std::integral_constant<std::size_t, 2>)
    {
        return std::array<xt::xtensor_fixed<std::size_t, xt::xshape<2>>, 4>
                {{ {0, 0}, {1, 0}, {1, 1}, {0, 1} }};
    }

    inline auto get_element(std::integral_constant<std::size_t, 3>)
    {
        return std::array<xt::xtensor_fixed<std::size_t, xt::xshape<3>>, 8>
                {{ {0, 0, 0}, {1, 0, 0}, {1, 1, 0},
                   {0, 1, 0}, {0, 0, 1}, {1, 0, 1},
                   {1, 1, 1}, {0, 1, 1}}};
    }

    template<class Field, class SubMesh>
    xt::xtensor<double, 2> extract_data(const Field& field, const SubMesh& submesh)
    {
        auto data_field = field.array();
        std::array<std::size_t, 2> shape = {submesh.nb_cells(), field.size};
        xt::xtensor<double, 2> data(shape);
        std::size_t index = 0;
        for_each_cell(submesh, [&](auto cell)
        {
            xt::view(data, index) = xt::view(data_field, cell.index);
            index++;
        });

        return data;
    }

    template<class Mesh>
    auto extract_coords_and_connectivity(const Mesh& mesh)
    {
        constexpr std::size_t dim = Mesh::dim;
        std::size_t nb_cells = mesh.nb_cells();

        std::size_t nb_points_per_cell = std::pow(2, dim);

        std::map<std::array<double, dim>, std::size_t> points_id;
        auto element = get_element(std::integral_constant<std::size_t, dim>{});

        xt::xtensor<std::size_t, 2> connectivity;
        connectivity.resize({nb_cells, nb_points_per_cell});

        std::size_t id = 0;
        std::size_t index = 0;
        for_each_cell(mesh, [&](auto cell)
        {
            auto start_corner = cell.corner();
            auto c = xt::xtensor<std::size_t, 1>::from_shape({element.size()});;

            for(std::size_t i = 0; i<element.size(); ++i)
            {
                auto corner = start_corner + cell.length * element[i];

                std::array<double, dim> a;
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
            connectivity_view = c;
            index++;
        });

        auto coords = xt::xtensor<double, 2>::from_shape({points_id.size(), 3});
        coords.fill(0.);
        for(auto& e: points_id)
        {
            std::size_t index = e.second;
            auto coords_view = xt::view(coords,
                                        index,
                                        xt::range(0, dim));
            coords_view = xt::adapt(e.first);
        }
        return std::make_pair(coords, connectivity);
    }

    struct Hdf5Options
    {
        Hdf5Options(bool level = false, bool mesh_id = false)
        : by_level(level), by_mesh_id(mesh_id)
        {}

        bool by_level;
        bool by_mesh_id;
    };

    template <class D, class Options, class Mesh, class... T>
    class Hdf5
    {
    public:

        using derived_type = D;
        using options_t = Options;
        using mesh_t = Mesh;
        static constexpr std::size_t dim = mesh_t::dim;

        Hdf5(const std::string& filename, const Options& options, const Mesh& mesh, const T&... fields);

        ~Hdf5();

        Hdf5(const Hdf5 &) = delete;
        Hdf5 &operator=(const Hdf5 &) = delete;

        Hdf5(Hdf5 &&) = default;
        Hdf5 &operator=(Hdf5 &&) = default;

        void save();

        derived_type &derived_cast() & noexcept;
        const derived_type &derived_cast() const &noexcept;
        derived_type derived_cast() && noexcept;

    protected:
        const mesh_t& m_mesh;

    private:
        using fields_type = std::tuple<const T&...>;

        template <class Submesh>
        void save_on_mesh(pugi::xml_node& grid_parent, const std::string& prefix, const Submesh& submesh, const std::string& mesh_name);

        template <class Submesh, class Field>
        inline void save_field(pugi::xml_node& grid, const std::string& prefix, const Submesh& submesh, const Field& field);

        template<class Submesh>
        void save_fields(pugi::xml_node& grid, const std::string& prefix, const Submesh& submesh);

        template <class Submesh, std::size_t... I>
        void save_fields_impl(pugi::xml_node& grid, const std::string& prefix, const Submesh& submesh, std::index_sequence<I...>);

        HighFive::File h5_file;
        std::string filename;
        pugi::xml_document doc;
        pugi::xml_node domain;
        std::ofstream xdmf_file;
        options_t m_options;
        fields_type m_fields;
    };

    template <class D, class Options, class Mesh, class... T>
    inline Hdf5<D, Options, Mesh, T...>::Hdf5(const std::string& filename, const Options& options, const Mesh& mesh, const T&... fields)
    : m_mesh(mesh)
    , h5_file(filename + ".h5", HighFive::File::Overwrite)
    , filename(filename)
    , m_options(options)
    , m_fields(fields...)
    {
        doc.append_child(pugi::node_doctype).set_value("Xdmf SYSTEM \"Xdmf.dtd\"");
        auto xdmf = doc.append_child("Xdmf");
        domain = xdmf.append_child("Domain");
    }

    template <class D, class Options, class Mesh, class... T>
    inline Hdf5<D, Options, Mesh, T...>::~Hdf5()
    {
        doc.save_file(fmt::format("{}.xdmf", filename).data());
    }

    template <class D, class Options, class Mesh, class... T>
    inline void Hdf5<D, Options, Mesh, T...>::save()
    {
        if (m_options.by_level)
        {
            auto min_level = m_mesh.min_level();
            auto max_level = m_mesh.max_level();
            for(std::size_t level = min_level; level <= max_level; ++level)
            {
                auto grid_level = domain.append_child("Grid");
                grid_level.append_attribute("Name") = fmt::format("Level {}", level).data();
                grid_level.append_attribute("GridType") = "Collection";

                if (m_options.by_mesh_id)
                {
                    for(std::size_t im = 0; im < this->derived_cast().nb_submesh(); ++im)
                    {
                        auto& submesh = this->derived_cast().get_submesh(im);

                        if (!submesh[level].empty())
                        {
                            std::string mesh_name = this->derived_cast().get_submesh_name(im);
                            std::string prefix = fmt::format("/level/{}/mesh/{}", level, mesh_name);
                            save_on_mesh(grid_level, prefix, submesh[level], mesh_name);
                        }
                    }
                }
                else
                {
                    auto& mesh = this->derived_cast().get_mesh();

                    std::string prefix = fmt::format("/level/{}/mesh", level);
                    save_on_mesh(grid_level, prefix, mesh[level], "mesh");
                }
            }
        }
        else
        {
            if (m_options.by_mesh_id)
            {
                for(std::size_t im = 0; im < this->derived_cast().nb_submesh(); ++im)
                {
                    auto& submesh = this->derived_cast().get_submesh(im);
                    std::string mesh_name = this->derived_cast().get_submesh_name(im);
                    std::string prefix = fmt::format("/mesh/{}", mesh_name);

                    auto grid_mesh_id = domain.append_child("Grid");
                    grid_mesh_id.append_attribute("Name") = mesh_name.data();
                    grid_mesh_id.append_attribute("GridType") = "Collection";

                    save_on_mesh(grid_mesh_id, prefix, submesh, mesh_name);
                }
            }
            else
            {
                auto& mesh = this->derived_cast().get_mesh();

                std::string prefix = fmt::format("/mesh");
                save_on_mesh(domain, prefix, mesh, "mesh");
            }
        }

    }

    template <class D, class Options, class Mesh, class... T>
    template <class Submesh>
    inline void Hdf5<D, Options, Mesh, T...>::save_on_mesh(pugi::xml_node& grid_parent, const std::string& prefix, const Submesh& submesh, const std::string& mesh_name)
    {
        xt::xtensor<std::size_t, 2> connectivity;
        xt::xtensor<double, 2> coords;
        std::tie(coords, connectivity) = extract_coords_and_connectivity(submesh);

        xt::dump(h5_file, prefix + "/connectivity", connectivity);
        xt::dump(h5_file, prefix + "/points", coords);

        auto grid = grid_parent.append_child("Grid");
        grid.append_attribute("Name") = mesh_name.data();

        auto topo = grid.append_child("Topology");
        topo.append_attribute("TopologyType") = element_type(dim).c_str();
        topo.append_attribute("NumberOfElements") = connectivity.shape()[0];

        auto topo_data = topo.append_child("DataItem");
        topo_data.append_attribute("Dimensions") = connectivity.size();
        topo_data.append_attribute("Format") = "HDF";
        topo_data.text() = fmt::format("{}.h5:{}/connectivity", filename, prefix).data();

        auto geom = grid.append_child("Geometry");
        geom.append_attribute("GeometryType") = "XYZ";

        auto geom_data = geom.append_child("DataItem");
        geom_data.append_attribute("Dimensions") = coords.size();
        geom_data.append_attribute("Format") = "HDF";
        geom_data.text() = fmt::format("{}.h5:{}/points", filename, prefix).data();

        save_fields(grid, prefix, submesh);
    }

    template <class D, class Options, class Mesh, class... T>
    template<class Submesh, class Field>
    inline void Hdf5<D, Options, Mesh, T...>::save_field(pugi::xml_node& grid, const std::string& prefix, const Submesh& submesh, const Field& field)
    {
        auto data = extract_data(field, submesh);

        for(std::size_t i = 0; i < field.size; ++i)
        {
            std::string field_name;
            if (field.size == 1)
            {
                field_name = field.name();
            }
            else
            {
                field_name = fmt::format("{}_{}", field.name(), i);
            }
            std::string path = fmt::format("{}/fields/{}", prefix, field_name);
            xt::dump(h5_file, path, xt::eval(xt::view(data, xt::all(), i)));

            auto attribute = grid.append_child("Attribute");
            attribute.append_attribute("Name") = field_name.data();
            attribute.append_attribute("Center") = "Cell";

            auto dataitem = attribute.append_child("DataItem");
            dataitem.append_attribute("Dimensions") = submesh.nb_cells();
            dataitem.append_attribute("Format") = "HDF";
            dataitem.text() = fmt::format("{}.h5:{}", filename, path).data();
        }
    }

    template <class D, class Options, class Mesh, class... T>
    template<class Submesh>
    inline void Hdf5<D, Options, Mesh, T...>::save_fields(pugi::xml_node& grid, const std::string& prefix, const Submesh& submesh)
    {
        save_fields_impl(grid, prefix, submesh, std::make_index_sequence<sizeof...(T)>());
    }

    template <class D, class Options, class Mesh, class... T>
    template <class Submesh, std::size_t... I>
    inline void Hdf5<D, Options, Mesh, T...>::save_fields_impl(pugi::xml_node& grid, const std::string& prefix, const Submesh& submesh, std::index_sequence<I...>)
    {
        (void)std::initializer_list<int>{(save_field(grid, prefix, submesh, std::get<I>(m_fields)), 0)...};
    }

    template <class D, class Options, class Mesh, class... T>
    inline auto Hdf5<D, Options, Mesh, T...>::derived_cast() & noexcept -> derived_type &
    {
        return *static_cast<derived_type *>(this);
    }

    template <class D, class Options, class Mesh, class... T>
    inline auto Hdf5<D, Options, Mesh, T...>::derived_cast() const & noexcept -> const derived_type &
    {
        return *static_cast<const derived_type *>(this);
    }

    template <class D, class Options, class Mesh, class... T>
    inline auto Hdf5<D, Options, Mesh, T...>::derived_cast() && noexcept -> derived_type
    {
        return *static_cast<derived_type *>(this);
    }

    template <class Options, class Mesh, class... T>
    class Hdf5_CellArray: public Hdf5<Hdf5_CellArray<Options, Mesh, T...>, Options, Mesh, T...>
    {
    public:
        using base_type = Hdf5<Hdf5_CellArray<Options, Mesh, T...>, Options, Mesh, T...>;
        using mesh_t = Mesh;

        Hdf5_CellArray(const std::string& filename, const Options& options, const Mesh& mesh, const T&... fields)
        : base_type(filename, options, mesh, fields...)
        {}

        const mesh_t& get_mesh() const
        {
            return this->m_mesh;
        }

        const mesh_t& get_submesh(std::size_t i) const
        {
            return this->m_mesh;
        }

        std::string get_submesh_name(std::size_t i) const
        {
            return "cell_array";
        }

        const std::size_t nb_submesh() const
        {
            return 1;
        }

    };

    template <class Options, class Mesh, class... T>
    class Hdf5_mesh_base: public Hdf5<Hdf5_mesh_base<Options, Mesh, T...>, Options, Mesh, T...>
    {
    public:
        using base_type = Hdf5<Hdf5_mesh_base<Options, Mesh, T...>, Options, Mesh, T...>;
        using mesh_t = Mesh;
        using mesh_id_t = typename Mesh::mesh_id_t;
        using ca_type = typename mesh_t::ca_type;

        Hdf5_mesh_base(const std::string& filename, const Options& options, const Mesh& mesh, const T&... fields)
        : base_type(filename, options, mesh, fields...)
        {}

        const ca_type& get_mesh() const
        {
            return this->m_mesh[mesh_id_t::cells];
        }

        const ca_type& get_submesh(std::size_t i) const
        {
            return this->m_mesh[static_cast<mesh_id_t>(i)];
        }

        std::string get_submesh_name(std::size_t i) const
        {
            return fmt::format("{}", static_cast<mesh_id_t>(i));
        }

        const std::size_t nb_submesh() const
        {
            return static_cast<std::size_t>(mesh_id_t::count);
        }

    };

    template <std::size_t dim, class TInterval, std::size_t max_size, class... T>
    void save(std::string name, const CellArray<dim, TInterval, max_size>& mesh, const T&... fields)
    {
        auto h5 = Hdf5_CellArray<Hdf5Options, CellArray<dim, TInterval, max_size>, T...>(name, Hdf5Options(), mesh, fields...);
        h5.save();
    }

    template <std::size_t dim, class TInterval, std::size_t max_size, class... T>
    void save(std::string name, const Hdf5Options& options, const CellArray<dim, TInterval, max_size>& mesh, const T&... fields)
    {
        auto h5 = Hdf5_CellArray<Hdf5Options, CellArray<dim, TInterval, max_size>, T...>(name, options, mesh, fields...);
        h5.save();
    }

    template <class D, class Config, class... T>
    void save(std::string name, const Mesh_base<D, Config>& mesh, const T&... fields)
    {
        auto h5 = Hdf5_mesh_base<Hdf5Options, Mesh_base<D, Config>, T...>(name, Hdf5Options(), mesh, fields...);
        h5.save();
    }

    template <class D, class Config, class... T>
    void save(std::string name, const Hdf5Options& options, const Mesh_base<D, Config>& mesh, const T&... fields)
    {
        auto h5 = Hdf5_mesh_base<Hdf5Options, Mesh_base<D, Config>, T...>(name, options, mesh, fields...);
        h5.save();
    }
} // namespace samurai