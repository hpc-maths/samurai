#pragma once

#include <fstream>
#include <string>
#include <type_traits>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor-io/xhighfive.hpp>

#include "cell.hpp"
#include "mesh.hpp"

namespace mure
{
    template<class MRConfig>
    class Mesh;

    std::string element_type(std::size_t dim)
    {
        switch ( dim ) {
        case 1: return "Polyline";
        case 2: return "Quadrilateral";
        case 3: return "Hexahedron";
        default: break;
        }
    }

    template<std::size_t dim>
    auto get_element(std::integral_constant<std::size_t, dim>);

    auto get_element(std::integral_constant<std::size_t, 1>)
    {
        return xt::xtensor<double, 1>{0, 1};
    }

    auto get_element(std::integral_constant<std::size_t, 2>)
    {
        return xt::xtensor<double, 2>{{0, 0},
                                      {1, 0},
                                      {1, 1},
                                      {0, 1}};
    }

    auto get_element(std::integral_constant<std::size_t, 3>)
    {
        return xt::xtensor<double, 2>{{0, 0, 0},
                                      {1, 0, 0},
                                      {1, 1, 0},
                                      {0, 1, 0},
                                      {0, 0, 1},
                                      {1, 0, 1},
                                      {1, 1, 1},
                                      {0, 1, 1}};
    }

    class Hdf5
    {
    public:

        Hdf5(std::string filename, std::size_t mesh_type=0)
            : h5_file(filename + ".h5", HighFive::File::Overwrite),
              filename(filename), mesh_type{mesh_type}
        {
            xdmf_file.open(filename + ".xdmf");
            xdmf_file << "<?xml version=\"1.0\" ?>\n";
            xdmf_file << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
            xdmf_file << "<Xdmf>\n";
            xdmf_file << "<Domain>\n";
            xdmf_file << "<Grid>\n";
        }

        ~Hdf5()
        {
            xdmf_file << "</Grid>\n";
            xdmf_file << "</Domain>\n";
            xdmf_file << "</Xdmf>\n";
            xdmf_file.close();
        }

        Hdf5(const Hdf5&) = delete;
        Hdf5& operator=(const Hdf5&) = delete;

        Hdf5(Hdf5&&) = default;
        Hdf5& operator=(Hdf5&&) = default;

        template<class MRConfig>
        void add_mesh(Mesh<MRConfig> const& mesh)
        {
            std::size_t nb_points = std::pow(2, Mesh<MRConfig>::dim);
            constexpr std::size_t dim =  Mesh<MRConfig>::dim;

            auto range = xt::arange(nb_points);

            xt::xtensor<std::size_t, 2> connectivity;
            connectivity.resize({mesh.nb_cells(mesh_type), nb_points});

            xt::xtensor<double, 2> coords;
            coords.resize({nb_points*mesh.nb_cells(mesh_type), 3});
            coords.fill(0);

            auto element = get_element(std::integral_constant<std::size_t, dim>{});

            std::size_t index = 0;
            mesh.for_each_cell([&](auto cell)
            {
                auto coords_view = xt::view(coords,
                                            xt::range(nb_points*index, nb_points*(index+1)),
                                            xt::range(0, dim));
                auto connectivity_view = xt::view(connectivity, index, xt::all());

                coords_view = xt::eval(cell.first_corner() + cell.length()*element);
                connectivity_view = xt::eval(nb_points*index + range);
                index++;
            }, mesh_type);
            xt::dump(h5_file, "mesh/connectivity", connectivity);
            xt::dump(h5_file, "mesh/points", coords);

            xdmf_file << "<Topology TopologyType=\"" << element_type(dim) << "\" NumberOfElements=\"" << connectivity.shape()[0] << "\">\n";
            xdmf_file << "<DataItem Dimensions=\"" << connectivity.size() << "\" Format=\"HDF\">\n";
            xdmf_file << filename << ".h5:/mesh/connectivity\n";
            xdmf_file << "</DataItem>\n";
            xdmf_file << "</Topology>\n";
            xdmf_file << "<Geometry GeometryType=\"XYZ\">\n";
            xdmf_file << "<DataItem Dimensions=\"" << coords.size() << "\" Format=\"HDF\">\n";
            xdmf_file << filename << ".h5:/mesh/points\n";
            xdmf_file << "</DataItem>\n";
            xdmf_file << "</Geometry>\n";
        }

        template<class MRConfig>
        void add_mesh_by_level(Mesh<MRConfig> const& mesh)
        {
            std::size_t nb_points = std::pow(2, Mesh<MRConfig>::dim);
            constexpr std::size_t dim =  Mesh<MRConfig>::dim;
            constexpr std::size_t max_refinement_level =  Mesh<MRConfig>::max_refinement_level;

            auto range = xt::arange(nb_points);

            for(std::size_t level=0; level<=max_refinement_level; ++level)
            {
                if (mesh.nb_cells_for_level(level, mesh_type) != 0)
                {
                    xt::xtensor<std::size_t, 2> connectivity;
                    connectivity.resize({mesh.nb_cells_for_level(level, mesh_type), nb_points});

                    xt::xtensor<double, 2> coords;
                    coords.resize({nb_points*mesh.nb_cells_for_level(level, mesh_type), 3});
                    coords.fill(0);

                    auto element = get_element(std::integral_constant<std::size_t, dim>{});

                    std::size_t index = 0;
                    mesh.for_each_cell_on_level(level, [&](auto cell)
                    {
                        auto coords_view = xt::view(coords,
                                                    xt::range(nb_points*index, nb_points*(index+1)),
                                                    xt::range(0, dim));
                        auto connectivity_view = xt::view(connectivity, index, xt::all());

                        coords_view = xt::eval(cell.first_corner() + cell.length()*element);
                        connectivity_view = xt::eval(nb_points*index + range);
                        index++;
                    }, mesh_type);
                    std::stringstream ss1;
                    ss1 << "level/" << level << "/mesh/connectivity";
                    xt::dump(h5_file, ss1.str().data(), connectivity);
                    std::stringstream ss2;
                    ss2 << "level/" << level << "/mesh/points";
                    xt::dump(h5_file, ss2.str().data(), coords);

                    xdmf_file << "<Grid Name=\"level " << level << "\">\n";
                    xdmf_file << "<Topology TopologyType=\"" << element_type(dim) << "\" NumberOfElements=\"" << connectivity.shape()[0] << "\">\n";
                    xdmf_file << "<DataItem Dimensions=\"" << connectivity.size() << "\" Format=\"HDF\">\n";
                    xdmf_file << filename << ".h5:/level/" << level << "/mesh/connectivity\n";
                    xdmf_file << "</DataItem>\n";
                    xdmf_file << "</Topology>\n";
                    xdmf_file << "<Geometry GeometryType=\"XYZ\">\n";
                    xdmf_file << "<DataItem Dimensions=\"" << coords.size() << "\" Format=\"HDF\">\n";
                    xdmf_file << filename << ".h5:/level/" << level << "/mesh/points\n";
                    xdmf_file << "</DataItem>\n";
                    xdmf_file << "</Geometry>\n";
                    xdmf_file << "</Grid>\n";
                }
            }
        }

        template<class MRConfig>
        void add_field(Field<MRConfig> const& field)
        {
            xt::dump(h5_file, "fields/" + field.name(), field.data(mesh_type));
            xdmf_file << "<Attribute Name='" << field.name() << "' Center='Cell'>\n";
            xdmf_file << "<DataItem Format='HDF' Dimensions='" << field.nb_cells() << " 1'>\n";
            xdmf_file << filename << ".h5:/fields/" << field.name() << "\n";
            xdmf_file << "</DataItem>\n";
            xdmf_file << "</Attribute>\n";
        }

        template<class MRConfig, class value_t>
        void add_field_by_level(Mesh<MRConfig> const& mesh, Field<MRConfig, value_t> const& field)
        {
            std::size_t nb_points = std::pow(2, Mesh<MRConfig>::dim);
            constexpr std::size_t dim =  Mesh<MRConfig>::dim;
            constexpr std::size_t max_refinement_level =  Mesh<MRConfig>::max_refinement_level;

            auto range = xt::arange(nb_points);

            for(std::size_t level=0; level<=max_refinement_level; ++level)
            {
                if (mesh.nb_cells_for_level(level, mesh_type) != 0)
                {
                    xt::xtensor<std::size_t, 2> connectivity;
                    connectivity.resize({mesh.nb_cells_for_level(level, mesh_type), nb_points});

                    xt::xtensor<double, 2> coords;
                    coords.resize({nb_points*mesh.nb_cells_for_level(level, mesh_type), 3});
                    coords.fill(0);

                    auto element = get_element(std::integral_constant<std::size_t, dim>{});

                    std::size_t index = 0;
                    mesh.for_each_cell_on_level(level, [&](auto cell)
                    {
                        auto coords_view = xt::view(coords,
                                                    xt::range(nb_points*index, nb_points*(index+1)),
                                                    xt::range(0, dim));
                        auto connectivity_view = xt::view(connectivity, index, xt::all());

                        coords_view = xt::eval(cell.first_corner() + cell.length()*element);
                        connectivity_view = xt::eval(nb_points*index + range);
                        index++;
                    }, mesh_type);
                    std::stringstream ss1;
                    ss1 << "level/" << level << "/mesh/connectivity";
                    xt::dump(h5_file, ss1.str().data(), connectivity);
                    std::stringstream ss2;
                    ss2 << "level/" << level << "/mesh/points";
                    xt::dump(h5_file, ss2.str().data(), coords);

                    xdmf_file << "<Grid Name=\"level " << level << "\">\n";
                    xdmf_file << "<Topology TopologyType=\"" << element_type(dim) << "\" NumberOfElements=\"" << connectivity.shape()[0] << "\">\n";
                    xdmf_file << "<DataItem Dimensions=\"" << connectivity.size() << "\" Format=\"HDF\">\n";
                    xdmf_file << filename << ".h5:/level/" << level << "/mesh/connectivity\n";
                    xdmf_file << "</DataItem>\n";
                    xdmf_file << "</Topology>\n";
                    xdmf_file << "<Geometry GeometryType=\"XYZ\">\n";
                    xdmf_file << "<DataItem Dimensions=\"" << coords.size() << "\" Format=\"HDF\">\n";
                    xdmf_file << filename << ".h5:/level/" << level << "/mesh/points\n";
                    xdmf_file << "</DataItem>\n";
                    xdmf_file << "</Geometry>\n";

                    std::stringstream ss;
                    ss << "level/" << level << "/fields/" << field.name();
                    xt::dump(h5_file, ss.str().data(), field.data_on_level(level, mesh_type));
                    xdmf_file << "<Attribute Name='" << field.name() << "' Center='Cell'>\n";
                    xdmf_file << "<DataItem Format='HDF' Dimensions='" << field.nb_cells_on_level(level, mesh_type) << " 1'>\n";
                    xdmf_file << filename << ".h5:/level/" << level << "/fields/" << field.name() << "\n";
                    xdmf_file << "</DataItem>\n";
                    xdmf_file << "</Attribute>\n";
                    xdmf_file << "</Grid>\n";
                }
            }
        }

        template<class MRConfig, class value_t>
        void add_field_by_level(Mesh<MRConfig> const& mesh, std::vector<Field<MRConfig, value_t>> const& fields)
        {
            std::size_t nb_points = std::pow(2, Mesh<MRConfig>::dim);
            constexpr std::size_t dim =  Mesh<MRConfig>::dim;
            constexpr std::size_t max_refinement_level =  Mesh<MRConfig>::max_refinement_level;

            auto range = xt::arange(nb_points);

            for(std::size_t level=0; level<=max_refinement_level; ++level)
            {
                if (mesh.nb_cells_for_level(level, mesh_type) != 0)
                {
                    xt::xtensor<std::size_t, 2> connectivity;
                    connectivity.resize({mesh.nb_cells_for_level(level, mesh_type), nb_points});

                    xt::xtensor<double, 2> coords;
                    coords.resize({nb_points*mesh.nb_cells_for_level(level, mesh_type), 3});
                    coords.fill(0);

                    auto element = get_element(std::integral_constant<std::size_t, dim>{});

                    std::size_t index = 0;
                    mesh.for_each_cell_on_level(level, [&](auto cell)
                    {
                        auto coords_view = xt::view(coords,
                                                    xt::range(nb_points*index, nb_points*(index+1)),
                                                    xt::range(0, dim));
                        auto connectivity_view = xt::view(connectivity, index, xt::all());

                        coords_view = xt::eval(cell.first_corner() + cell.length()*element);
                        connectivity_view = xt::eval(nb_points*index + range);
                        index++;
                    }, mesh_type);
                    std::stringstream ss1;
                    ss1 << "level/" << level << "/mesh/connectivity";
                    xt::dump(h5_file, ss1.str().data(), connectivity);
                    std::stringstream ss2;
                    ss2 << "level/" << level << "/mesh/points";
                    xt::dump(h5_file, ss2.str().data(), coords);

                    xdmf_file << "<Grid Name=\"level " << level << "\">\n";
                    xdmf_file << "<Topology TopologyType=\"" << element_type(dim) << "\" NumberOfElements=\"" << connectivity.shape()[0] << "\">\n";
                    xdmf_file << "<DataItem Dimensions=\"" << connectivity.size() << "\" Format=\"HDF\">\n";
                    xdmf_file << filename << ".h5:/level/" << level << "/mesh/connectivity\n";
                    xdmf_file << "</DataItem>\n";
                    xdmf_file << "</Topology>\n";
                    xdmf_file << "<Geometry GeometryType=\"XYZ\">\n";
                    xdmf_file << "<DataItem Dimensions=\"" << coords.size() << "\" Format=\"HDF\">\n";
                    xdmf_file << filename << ".h5:/level/" << level << "/mesh/points\n";
                    xdmf_file << "</DataItem>\n";
                    xdmf_file << "</Geometry>\n";

                    for (auto& field: fields)
                    {
                        std::stringstream ss;
                        ss << "level/" << level << "/fields/" << field.name();
                        xt::dump(h5_file, ss.str().data(), field.data_on_level(level, mesh_type));
                        xdmf_file << "<Attribute Name='" << field.name() << "' Center='Cell'>\n";
                        xdmf_file << "<DataItem Format='HDF' Dimensions='" << field.nb_cells_on_level(level, mesh_type) << " 1'>\n";
                        xdmf_file << filename << ".h5:/level/" << level << "/fields/" << field.name() << "\n";
                        xdmf_file << "</DataItem>\n";
                        xdmf_file << "</Attribute>\n";
                    }
                    xdmf_file << "</Grid>\n";
                }
            }
        }

    private:
        std::string filename;
        HighFive::File h5_file;
        std::ofstream xdmf_file;
        std::size_t mesh_type;
    };

}