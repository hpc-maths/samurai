#pragma once

#include <fstream>
#include <functional>
#include <string>
#include <type_traits>

#include <pugixml.hpp>
#include <xtensor-io/xhighfive.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

#include "cell.hpp"
#include "algorithm.hpp"
#include "mr/mesh_type.hpp"
#include "mesh.hpp"
#include "utils.hpp"

namespace mure
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
        return xt::xtensor<double, 1>{0, 1};
    }

    inline auto get_element(std::integral_constant<std::size_t, 2>)
    {
        return xt::xtensor<double, 2>{{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    }

    inline auto get_element(std::integral_constant<std::size_t, 3>)
    {
        return xt::xtensor<double, 2>{{0, 0, 0}, {1, 0, 0}, {1, 1, 0},
                                      {0, 1, 0}, {0, 0, 1}, {1, 0, 1},
                                      {1, 1, 1}, {0, 1, 1}};
    }

    template<class Field, class SubMesh>
    auto extract_data(const Field& field, const SubMesh& submesh)
    {
        auto data_field = field.array();
        std::array<std::size_t, 2> shape = {submesh.nb_cells(), field.size};
        xt::xtensor<double, 2> data(shape);
        std::size_t index = 0;
        for_each_cell(submesh, [&](auto cell)
        {
            auto view = xt::view(data, index++);
            view = xt::view(data_field, cell.index);
        });

        return data;
    }

    template<class Mesh>
    auto extract_coords_and_connectivity(const Mesh& mesh)
    {
        constexpr std::size_t dim = Mesh::dim;
        std::size_t nb_cells = mesh.nb_cells();

        std::size_t nb_points_per_cell = std::pow(2, dim);
        auto range = xt::arange(nb_points_per_cell);

        xt::xtensor<std::size_t, 2> connectivity;
        connectivity.resize({nb_cells, nb_points_per_cell});

        xt::xtensor<double, 2> coords;
        coords.resize({nb_points_per_cell * nb_cells, 3});
        coords.fill(0);

        auto element = get_element(std::integral_constant<std::size_t, dim>{});

        std::size_t index = 0;
        for_each_cell(mesh, [&](auto cell)
        {
            auto coords_view = xt::view(coords,
                                        xt::range(nb_points_per_cell * index, nb_points_per_cell * (index + 1)),
                                        xt::range(0, dim));
            auto connectivity_view = xt::view(connectivity, index, xt::all());

            coords_view = xt::eval(cell.first_corner() + cell.length * element);
            connectivity_view = xt::eval(nb_points_per_cell * index + range);
            index++;
        });

        return std::make_pair(coords, connectivity);
    }

    template <class D, class Mesh, class... T>
    class Hdf5
    {
    public:

        using derived_type = D;
        using mesh_t = Mesh;
        static constexpr std::size_t dim = mesh_t::dim;

        Hdf5(std::string filename, const Mesh& mesh, const T&... fields);

        ~Hdf5();

        Hdf5(const Hdf5 &) = delete;
        Hdf5 &operator=(const Hdf5 &) = delete;

        Hdf5(Hdf5 &&) = default;
        Hdf5 &operator=(Hdf5 &&) = default;

        void save_mesh();

        template<class Field>
        inline void save_field(const Field& field);

        inline void save_fields()
        {
            save_fields_impl(std::make_index_sequence<sizeof...(T)>());
        }

        template <std::size_t... I>
        inline void save_fields_impl(std::index_sequence<I...>)
        {
            (void)std::initializer_list<int>{(save_field(std::get<I>(m_fields)), 0)...};
        }

        derived_type &derived_cast() & noexcept
        {
            return *static_cast<derived_type *>(this);
        }

        const derived_type &derived_cast() const &noexcept
        {
            return *static_cast<const derived_type *>(this);
        }

        derived_type derived_cast() && noexcept
        {
            return *static_cast<derived_type *>(this);
        }
        // template<class MRConfig, class Field>
        // inline void add_field_by_level(const Mesh<MRConfig>& mesh, const Field& field);

        // template<class MRConfig, class Field>
        // inline void _add_on_level(const Mesh<MRConfig>& mesh, const Field& field, std::size_t level);

    protected:
        mesh_t m_mesh;

    private:
        using fields_type = std::tuple<T...>;

        HighFive::File h5_file;
        std::string filename;
        pugi::xml_document doc;
        pugi::xml_node domain;
        std::ofstream xdmf_file;
        fields_type m_fields;
    };

    template <class D, class Mesh, class... T>
    inline Hdf5<D, Mesh, T...>::Hdf5(std::string filename, const Mesh& mesh, const T&... fields)
    : h5_file(filename + ".h5", HighFive::File::Overwrite)
    , filename(filename)
    , m_mesh(mesh)
    , m_fields(fields...)
    {
        doc.append_child(pugi::node_doctype).set_value("Xdmf SYSTEM \"Xdmf.dtd\"");
        auto xdmf = doc.append_child("Xdmf");
        domain = xdmf.append_child("Domain");
    }

    template <class D, class Mesh, class... T>
    inline Hdf5<D, Mesh, T...>::~Hdf5()
    {
        std::stringstream xdmf_str;
        xdmf_str << filename << ".xdmf";
        doc.save_file(xdmf_str.str().data());
    }

    template <class D, class Mesh, class... T>
    inline void Hdf5<D, Mesh, T...>::save_mesh()
    {
        auto mesh = this->derived_cast().get_mesh();
        xt::xtensor<std::size_t, 2> connectivity;
        xt::xtensor<double, 2> coords;
        std::tie(coords, connectivity) = extract_coords_and_connectivity(mesh);

        xt::dump(h5_file, "mesh/connectivity", connectivity);
        xt::dump(h5_file, "mesh/points", coords);

        auto grid = domain.append_child("Grid");

        auto topo = grid.append_child("Topology");
        topo.append_attribute("TopologyType") = element_type(dim).c_str();
        topo.append_attribute("NumberOfElements") = connectivity.shape()[0];

        auto topo_data = topo.append_child("DataItem");
        topo_data.append_attribute("Dimensions") = connectivity.size();
        topo_data.append_attribute("Format") = "HDF";
        topo_data.text() = (filename + ".h5:/mesh/connectivity").c_str();

        auto geom = grid.append_child("Geometry");
        geom.append_attribute("GeometryType") = "XYZ";

        auto geom_data = geom.append_child("DataItem");
        geom_data.append_attribute("Dimensions") = coords.size();
        geom_data.append_attribute("Format") = "HDF";
        geom_data.text() = (filename + ".h5:/mesh/points").c_str();
    }

    template <class D, class Mesh, class... T>
    template<class Field>
    inline void Hdf5<D, Mesh, T...>::save_field(const Field& field)
    {
        auto mesh = this->derived_cast().get_mesh();
        auto grid = domain.child("Grid");
        auto data = extract_data(field, mesh);

        for(std::size_t i = 0; i < field.size; ++i)
        {
            std::stringstream s;
            s << field.name() << "_" << i;
            xt::dump(h5_file, "fields/" + s.str(), xt::eval(xt::view(data, xt::all(), i)));

            auto attribute = grid.append_child("Attribute");
            attribute.append_attribute("Name") = s.str().c_str();
            attribute.append_attribute("Center") = "Cell";

            auto dataitem = attribute.append_child("DataItem");
            dataitem.append_attribute("Dimensions") = mesh.nb_cells();
            dataitem.append_attribute("Format") = "HDF";
            dataitem.text() = (filename + ".h5:/fields/" + s.str()).c_str();
        }
    }

    // template<class MRConfig, class Field>
    // inline void Hdf5::add_field_by_level(const Mesh<MRConfig>& mesh, const Field& field)
    // {
    //     constexpr std::size_t max_refinement_level = Mesh<MRConfig>::max_refinement_level;

    //     for (std::size_t level = 0; level <= max_refinement_level; ++level)
    //     {
    //         _add_on_level(mesh, field, level);
    //     }
    // }


    // template<class MRConfig, class Field>
    // inline void Hdf5::_add_on_level(const Mesh<MRConfig>& mesh, const Field& field, std::size_t level)
    // {
    //     std::size_t nb_points = std::pow(2, Mesh<MRConfig>::dim);
    //     constexpr std::size_t dim = Mesh<MRConfig>::dim;

    //     // std::array<std::string, 5> mesh_name{"cells", "cells_and_ghosts",
    //     //                                      "proj", "all", "union"};

    //     std::array<std::string, 6> mesh_name{"cells", "cells_and_ghosts",
    //                                             "proj", "all", "union", "overleaves"};


    //     auto range = xt::arange(nb_points);

    //     auto grid_parent = domain.append_child("Grid");
    //     grid_parent.append_attribute("Name") = ("level " + std::to_string(level)).c_str();
    //     grid_parent.append_attribute("GridType") = "Collection";

    //     for (std::size_t imesh_type = 0; imesh_type < 6; ++imesh_type)
    //     {
    //         auto mesh_type = static_cast<MeshType>(imesh_type);
    //         if (mesh.nb_cells(level, mesh_type) != 0)
    //         {
    //             xt::xtensor<std::size_t, 2> connectivity;
    //             xt::xtensor<double, 2> coords;
    //             std::tie(coords, connectivity) = extract_coords_and_connectivity(mesh[mesh_type][level]);

    //             std::stringstream ss1;
    //             ss1 << "level/" << level << "/mesh/" << imesh_type << "/connectivity";
    //             xt::dump(h5_file, ss1.str().data(), connectivity);
    //             std::stringstream ss2;
    //             ss2 << "level/" << level << "/mesh/" << imesh_type << "/points";
    //             xt::dump(h5_file, ss2.str().data(), coords);

    //             auto grid = grid_parent.append_child("Grid");
    //             grid.append_attribute("Name") = mesh_name[imesh_type].c_str();

    //             auto topo = grid.append_child("Topology");
    //             topo.append_attribute("TopologyType") = element_type(dim).c_str();
    //             topo.append_attribute("NumberOfElements") = connectivity.shape()[0];

    //             auto topo_data = topo.append_child("DataItem");
    //             topo_data.append_attribute("Dimensions") = connectivity.size();
    //             topo_data.append_attribute("Format") = "HDF";
    //             topo_data.text() = (filename + ".h5:" + ss1.str()).c_str();

    //             auto geom = grid.append_child("Geometry");
    //             geom.append_attribute("GeometryType") = "XYZ";

    //             auto geom_data = geom.append_child("DataItem");
    //             geom_data.append_attribute("Dimensions") = coords.size();
    //             geom_data.append_attribute("Format") = "HDF";
    //             geom_data.text() = (filename + ".h5:" + ss2.str()).c_str();

    //             // Modified to handle multiple components of the field
    //             auto mesh = field.mesh();
    //             for (std::size_t h = 0; h <field.size; ++h)
    //             {
    //                 std::stringstream new_field_name;
    //                 new_field_name << field.name()<< "_" << h;

    //                 std::stringstream ss;
    //                 ss << "level/" << level << "/" << imesh_type << "/fields/" << new_field_name.str();

    //                 auto all_fields = extract_data(field, mesh[mesh_type][level]);

    //                 xt::dump(h5_file, ss.str().data(), xt::eval(xt::view(all_fields, xt::all(), h)));

    //                 auto attribute = grid.append_child("Attribute");
    //                 attribute.append_attribute("Name") = new_field_name.str().c_str();
    //                 attribute.append_attribute("Center") = "Cell";

    //                 auto dataitem = attribute.append_child("DataItem");
    //                 dataitem.append_attribute("Dimensions") = mesh.nb_cells(level, mesh_type);
    //                 dataitem.append_attribute("Format") = "HDF";
    //                 dataitem.text() = ((filename + ".h5:/level/" + std::to_string(level)
    //                                 + "/" + std::to_string(imesh_type) + "/fields/"
    //                                 + new_field_name.str()).c_str());
    //             }
    //         }
    //     }
    // }

    template <class Mesh, class... T>
    class Hdf5_CellArray: public Hdf5<Hdf5_CellArray<Mesh, T...>, Mesh, T...>
    {
    public:
        using base_type = Hdf5<Hdf5_CellArray<Mesh, T...>, Mesh, T...>;
        using mesh_t = Mesh;

        Hdf5_CellArray(std::string filename, const Mesh& mesh, const T&... fields)
        : base_type(filename, mesh, fields...)
        {}

        const mesh_t& get_mesh()
        {
            return this->m_mesh;
        }
    };

    template <class Config>
    class Mesh;

    template <class Mesh, class... T>
    class Hdf5_MR: public Hdf5<Hdf5_MR<Mesh, T...>, Mesh, T...>
    {
    public:
        using base_type = Hdf5<Hdf5_MR<Mesh, T...>, Mesh, T...>;
        using mesh_t = Mesh;
        using mesh_id_t = typename Mesh::mesh_id_t;
        using ca_type = typename mesh_t::ca_type;

        Hdf5_MR(std::string filename, const Mesh& mesh, const T&... fields)
        : base_type(filename, mesh, fields...)
        {}

        const ca_type& get_mesh()
        {
            return this->m_mesh[mesh_id_t::cells];
        }
    };

    template <std::size_t dim, class TInterval, std::size_t max_size, class... T>
    void save(std::string name, const CellArray<dim, TInterval, max_size>& mesh, const T&... fields)
    {
        auto h5 = Hdf5_CellArray<CellArray<dim, TInterval, max_size>, T...>(name, mesh, fields...);
        h5.save_mesh();
        h5.save_fields();
    }

    template <class Config, class... T>
    void save(std::string name, const Mesh<Config>& mesh, const T&... fields)
    {
        auto h5 = Hdf5_MR<Mesh<Config>, T...>(name, mesh, fields...);
        h5.save_mesh();
        h5.save_fields();
    }

    template <class D, class Config, class... T>
    void save(std::string name, const Mesh_base<D, Config>& mesh, const T&... fields)
    {
        auto h5 = Hdf5_MR<Mesh_base<D, Config>, T...>(name, mesh, fields...);
        h5.save_mesh();
        h5.save_fields();
    }
}