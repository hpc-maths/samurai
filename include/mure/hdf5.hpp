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
#include "mr/mesh_type.hpp"

namespace mure
{
    template<class MRConfig>
    class Mesh;

    template<class MRConfig, class value_t = double>
    class Field;

    std::string element_type(std::size_t dim)
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

    auto get_element(std::integral_constant<std::size_t, 1>)
    {
        return xt::xtensor<double, 1>{0, 1};
    }

    auto get_element(std::integral_constant<std::size_t, 2>)
    {
        return xt::xtensor<double, 2>{{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    }

    auto get_element(std::integral_constant<std::size_t, 3>)
    {
        return xt::xtensor<double, 2>{{0, 0, 0}, {1, 0, 0}, {1, 1, 0},
                                      {0, 1, 0}, {0, 0, 1}, {1, 0, 1},
                                      {1, 1, 1}, {0, 1, 1}};
    }

    class Hdf5 {
      public:
        Hdf5(std::string filename, MeshType mesh_type = MeshType::cells)
            : h5_file(filename + ".h5", HighFive::File::Overwrite),
              filename(filename), mesh_type{mesh_type}
        {
            doc.append_child(pugi::node_doctype)
                .set_value("Xdmf SYSTEM \"Xdmf.dtd\"");
            auto xdmf = doc.append_child("Xdmf");
            domain = xdmf.append_child("Domain");
        }

        ~Hdf5()
        {
            std::stringstream xdmf_str;
            xdmf_str << filename << ".xdmf";
            doc.save_file(xdmf_str.str().data());
        }

        Hdf5(const Hdf5 &) = delete;
        Hdf5 &operator=(const Hdf5 &) = delete;

        Hdf5(Hdf5 &&) = default;
        Hdf5 &operator=(Hdf5 &&) = default;

        template<class MRConfig>
        void add_mesh(Mesh<MRConfig> const &mesh)
        {
            std::size_t nb_points = std::pow(2, Mesh<MRConfig>::dim);
            constexpr std::size_t dim = Mesh<MRConfig>::dim;

            auto range = xt::arange(nb_points);

            xt::xtensor<std::size_t, 2> connectivity;
            connectivity.resize({mesh.nb_cells(mesh_type), nb_points});

            xt::xtensor<double, 2> coords;
            coords.resize({nb_points * mesh.nb_cells(mesh_type), 3});
            coords.fill(0);

            auto element =
                get_element(std::integral_constant<std::size_t, dim>{});

            std::size_t index = 0;
            mesh.for_each_cell(
                [&](auto cell) {
                    auto coords_view = xt::view(
                        coords,
                        xt::range(nb_points * index, nb_points * (index + 1)),
                        xt::range(0, dim));
                    auto connectivity_view =
                        xt::view(connectivity, index, xt::all());

                    coords_view =
                        xt::eval(cell.first_corner() + cell.length() * element);
                    connectivity_view = xt::eval(nb_points * index + range);
                    index++;
                },
                mesh_type);
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

        template<class MRConfig>
        void add_mesh_by_level(Mesh<MRConfig> const &mesh)
        {
            std::size_t nb_points = std::pow(2, Mesh<MRConfig>::dim);
            constexpr std::size_t dim = Mesh<MRConfig>::dim;
            constexpr std::size_t max_refinement_level =
                Mesh<MRConfig>::max_refinement_level;

            auto range = xt::arange(nb_points);

            for (std::size_t level = 0; level <= max_refinement_level; ++level)
            {
                if (mesh.nb_cells(level, mesh_type) != 0)
                {
                    xt::xtensor<std::size_t, 2> connectivity;
                    connectivity.resize(
                        {mesh.nb_cells(level, mesh_type), nb_points});

                    xt::xtensor<double, 2> coords;
                    coords.resize(
                        {nb_points * mesh.nb_cells(level, mesh_type), 3});
                    coords.fill(0);

                    auto element =
                        get_element(std::integral_constant<std::size_t, dim>{});

                    std::size_t index = 0;
                    mesh.for_each_cell(
                        level,
                        [&](auto cell) {
                            auto coords_view =
                                xt::view(coords,
                                         xt::range(nb_points * index,
                                                   nb_points * (index + 1)),
                                         xt::range(0, dim));
                            auto connectivity_view =
                                xt::view(connectivity, index, xt::all());

                            coords_view = xt::eval(cell.first_corner() +
                                                   cell.length() * element);
                            connectivity_view =
                                xt::eval(nb_points * index + range);
                            index++;
                        },
                        mesh_type);
                    std::stringstream ss1;
                    ss1 << "level/" << level << "/mesh/connectivity";
                    xt::dump(h5_file, ss1.str().data(), connectivity);
                    std::stringstream ss2;
                    ss2 << "level/" << level << "/mesh/points";
                    xt::dump(h5_file, ss2.str().data(), coords);

                    // xdmf_file << "<Grid Name=\"level " << level << "\">\n";
                    // xdmf_file << "<Topology TopologyType=\"" <<
                    // element_type(dim) << "\" NumberOfElements=\"" <<
                    // connectivity.shape()[0] << "\">\n"; xdmf_file <<
                    // "<DataItem Dimensions=\"" << connectivity.size() << "\"
                    // Format=\"HDF\">\n"; xdmf_file << filename <<
                    // ".h5:/level/" << level << "/mesh/connectivity\n";
                    // xdmf_file << "</DataItem>\n";
                    // xdmf_file << "</Topology>\n";
                    // xdmf_file << "<Geometry GeometryType=\"XYZ\">\n";
                    // xdmf_file << "<DataItem Dimensions=\"" << coords.size()
                    // << "\" Format=\"HDF\">\n"; xdmf_file << filename <<
                    // ".h5:/level/" << level << "/mesh/points\n"; xdmf_file <<
                    // "</DataItem>\n"; xdmf_file << "</Geometry>\n"; xdmf_file
                    // << "</Grid>\n";
                }
            }
        }

        template<class MRConfig, class value_t>
        void add_field(Field<MRConfig, value_t> const &field)
        {
            xt::dump(h5_file, "fields/" + field.name(), field.data(mesh_type));
            auto grid = domain.child("Grid");
            auto attribute = grid.append_child("Attribute");
            attribute.append_attribute("Name") = field.name().c_str();
            attribute.append_attribute("Center") = "Cell";
            auto dataitem = attribute.append_child("DataItem");
            dataitem.append_attribute("Dimensions") = field.nb_cells(mesh_type);
            dataitem.append_attribute("Format") = "HDF";
            dataitem.text() =
                (filename + ".h5:/fields/" + field.name()).c_str();
        }

        template<class MRConfig, class value_t>
        void add_field_by_level(Mesh<MRConfig> const &mesh,
                                Field<MRConfig, value_t> const &field)
        {
            constexpr std::size_t max_refinement_level =
                Mesh<MRConfig>::max_refinement_level;

            for (std::size_t level = 0; level <= max_refinement_level; ++level)
            {
                _add_on_level(mesh, field, level);
            }
        }

        template<class MRConfig, class value_t>
        void _add_on_level(Mesh<MRConfig> const &mesh,
                           Field<MRConfig, value_t> const &field,
                           std::size_t level)
        {
            std::size_t nb_points = std::pow(2, Mesh<MRConfig>::dim);
            constexpr std::size_t dim = Mesh<MRConfig>::dim;

            std::array<std::string, 4> mesh_name{"cells", "cells_and_ghosts",
                                                 "proj", "all"};

            auto range = xt::arange(nb_points);

            auto grid_parent = domain.append_child("Grid");
            grid_parent.append_attribute("Name") =
                ("level " + std::to_string(level)).c_str();
            grid_parent.append_attribute("GridType") = "Collection";

            for (std::size_t mesh_type = 0; mesh_type < 4; ++mesh_type)
            {
                if (mesh.nb_cells(level, static_cast<MeshType>(mesh_type)) != 0)
                {
                    xt::xtensor<std::size_t, 2> connectivity;
                    connectivity.resize(
                        {mesh.nb_cells(level, static_cast<MeshType>(mesh_type)),
                         nb_points});

                    xt::xtensor<double, 2> coords;
                    coords.resize(
                        {nb_points * mesh.nb_cells(level, static_cast<MeshType>(
                                                              mesh_type)),
                         3});
                    coords.fill(0);

                    auto element =
                        get_element(std::integral_constant<std::size_t, dim>{});

                    std::size_t index = 0;
                    mesh.for_each_cell(
                        level,
                        [&](auto cell) {
                            auto coords_view =
                                xt::view(coords,
                                         xt::range(nb_points * index,
                                                   nb_points * (index + 1)),
                                         xt::range(0, dim));
                            auto connectivity_view =
                                xt::view(connectivity, index, xt::all());

                            coords_view = xt::eval(cell.first_corner() +
                                                   cell.length() * element);
                            connectivity_view =
                                xt::eval(nb_points * index + range);
                            index++;
                        },
                        static_cast<MeshType>(mesh_type));
                    std::stringstream ss1;
                    ss1 << "level/" << level << "/mesh/" << mesh_type
                        << "/connectivity";
                    xt::dump(h5_file, ss1.str().data(), connectivity);
                    std::stringstream ss2;
                    ss2 << "level/" << level << "/mesh/" << mesh_type
                        << "/points";
                    xt::dump(h5_file, ss2.str().data(), coords);

                    auto grid = grid_parent.append_child("Grid");
                    grid.append_attribute("Name") =
                        mesh_name[mesh_type].c_str();
                    auto topo = grid.append_child("Topology");
                    topo.append_attribute("TopologyType") =
                        element_type(dim).c_str();
                    topo.append_attribute("NumberOfElements") =
                        connectivity.shape()[0];
                    auto topo_data = topo.append_child("DataItem");
                    topo_data.append_attribute("Dimensions") =
                        connectivity.size();
                    topo_data.append_attribute("Format") = "HDF";
                    topo_data.text() = (filename + ".h5:" + ss1.str()).c_str();

                    auto geom = grid.append_child("Geometry");
                    geom.append_attribute("GeometryType") = "XYZ";
                    auto geom_data = geom.append_child("DataItem");
                    geom_data.append_attribute("Dimensions") = coords.size();
                    geom_data.append_attribute("Format") = "HDF";
                    geom_data.text() = (filename + ".h5:" + ss2.str()).c_str();

                    std::stringstream ss;
                    ss << "level/" << level << "/" << mesh_type << "/fields/"
                       << field.name();
                    xt::dump(h5_file, ss.str().data(),
                             field.data_on_level(
                                 level, static_cast<MeshType>(mesh_type)));

                    auto attribute = grid.append_child("Attribute");
                    attribute.append_attribute("Name") = field.name().c_str();
                    attribute.append_attribute("Center") = "Cell";
                    auto dataitem = attribute.append_child("DataItem");
                    dataitem.append_attribute("Dimensions") =
                        field.nb_cells(level, static_cast<MeshType>(mesh_type));
                    dataitem.append_attribute("Format") = "HDF";
                    dataitem.text() =
                        (filename + ".h5:/level/" + std::to_string(level) +
                         "/" + std::to_string(mesh_type) + "/fields/" +
                         field.name())
                            .c_str();
                }
            }
        }
        template<class MRConfig, class value_t>
        void
        add_field_by_level(Mesh<MRConfig> const &mesh,
                           std::vector<Field<MRConfig, value_t>> const &fields)
        {
            std::size_t nb_points = std::pow(2, Mesh<MRConfig>::dim);
            constexpr std::size_t dim = Mesh<MRConfig>::dim;
            constexpr std::size_t max_refinement_level =
                Mesh<MRConfig>::max_refinement_level;

            auto range = xt::arange(nb_points);

            for (std::size_t level = 0; level <= max_refinement_level; ++level)
            {
                if (mesh.nb_cells_for_level(level, mesh_type) != 0)
                {
                    xt::xtensor<std::size_t, 2> connectivity;
                    connectivity.resize(
                        {mesh.nb_cells_for_level(level, mesh_type), nb_points});

                    xt::xtensor<double, 2> coords;
                    coords.resize(
                        {nb_points * mesh.nb_cells_for_level(level, mesh_type),
                         3});
                    coords.fill(0);

                    auto element =
                        get_element(std::integral_constant<std::size_t, dim>{});

                    std::size_t index = 0;
                    mesh.for_each_cell_on_level(
                        level,
                        [&](auto cell) {
                            auto coords_view =
                                xt::view(coords,
                                         xt::range(nb_points * index,
                                                   nb_points * (index + 1)),
                                         xt::range(0, dim));
                            auto connectivity_view =
                                xt::view(connectivity, index, xt::all());

                            coords_view = xt::eval(cell.first_corner() +
                                                   cell.length() * element);
                            connectivity_view =
                                xt::eval(nb_points * index + range);
                            index++;
                        },
                        mesh_type);
                    std::stringstream ss1;
                    ss1 << "level/" << level << "/mesh/connectivity";
                    xt::dump(h5_file, ss1.str().data(), connectivity);
                    std::stringstream ss2;
                    ss2 << "level/" << level << "/mesh/points";
                    xt::dump(h5_file, ss2.str().data(), coords);

                    // xdmf_file << "<Grid Name=\"level " << level << "\">\n";
                    // xdmf_file << "<Topology TopologyType=\"" <<
                    // element_type(dim) << "\" NumberOfElements=\"" <<
                    // connectivity.shape()[0] << "\">\n"; xdmf_file <<
                    // "<DataItem Dimensions=\"" << connectivity.size() << "\"
                    // Format=\"HDF\">\n"; xdmf_file << filename <<
                    // ".h5:/level/" << level << "/mesh/connectivity\n";
                    // xdmf_file << "</DataItem>\n";
                    // xdmf_file << "</Topology>\n";
                    // xdmf_file << "<Geometry GeometryType=\"XYZ\">\n";
                    // xdmf_file << "<DataItem Dimensions=\"" << coords.size()
                    // << "\" Format=\"HDF\">\n"; xdmf_file << filename <<
                    // ".h5:/level/" << level << "/mesh/points\n"; xdmf_file <<
                    // "</DataItem>\n"; xdmf_file << "</Geometry>\n";

                    for (auto &field : fields)
                    {
                        std::stringstream ss;
                        ss << "level/" << level << "/fields/" << field.name();
                        xt::dump(h5_file, ss.str().data(),
                                 field.data_on_level(level, mesh_type));
                        // xdmf_file << "<Attribute Name='" << field.name() <<
                        // "' Center='Cell'>\n"; xdmf_file << "<DataItem
                        // Format='HDF' Dimensions='" <<
                        // field.nb_cells_on_level(level, mesh_type) << "
                        // 1'>\n"; xdmf_file << filename << ".h5:/level/" <<
                        // level << "/fields/" << field.name() << "\n";
                        // xdmf_file << "</DataItem>\n";
                        // xdmf_file << "</Attribute>\n";
                    }
                    // xdmf_file << "</Grid>\n";
                }
            }
        }

      private:
        HighFive::File h5_file;
        std::string filename;
        pugi::xml_document doc;
        pugi::xml_node domain;
        std::ofstream xdmf_file;
        MeshType mesh_type;
    };
}