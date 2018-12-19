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

        Hdf5(std::string filename)
            : h5_file(filename + ".h5", HighFive::File::Overwrite),
              filename(filename)
        {
            xdmf_file.open(filename + ".xdmf");
            xdmf_file << "<?xml version=\"1.0\" ?>\n";
            xdmf_file << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
            xdmf_file << "<Xdmf>\n";
        }

        ~Hdf5()
        {
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
            connectivity.resize({mesh.nb_cells(), nb_points});

            xt::xtensor<double, 2> coords;
            coords.resize({nb_points*mesh.nb_cells(), 3});

            auto element = get_element(std::integral_constant<std::size_t, dim>{});

            std::size_t index = 0;
            for(std::size_t level=0; level<std::size_t(Mesh<MRConfig>::max_refinement_level); ++level)
            {
                auto level_array = mesh.get_cells(level);
                level_array.for_each_interval_in_x([&](auto& index_yz, auto& interval)
                {
                    for(int ix=interval.start; ix<interval.end; ++ix)
                    {
                        Cell<int, Mesh<MRConfig>::dim> cell{level,
                                                            xt::concatenate(xt::xtuple(xt::xarray<int>{ix}, index_yz))};
                        auto coords_view = xt::view(coords,
                                                    xt::range(nb_points*index, nb_points*(index+1)),
                                                    xt::range(0, dim));
                        auto connectivity_view = xt::view(connectivity, index, xt::all());

                        coords_view = xt::eval(cell.first_corner() + cell.length()*element);
                        connectivity_view = xt::eval(nb_points*index + range);
                        index++;
                    }
                }
                );
            }
            xt::dump(h5_file, "mesh/connectivity", connectivity);
            xt::dump(h5_file, "mesh/points", coords);

            xdmf_file << "<Domain>\n";
            xdmf_file << "<Grid>\n";
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
            xdmf_file << "</Grid>\n";
            xdmf_file << "</Domain>\n";
            xdmf_file << "</Xdmf>\n";
        }
    private:
        std::string filename;
        HighFive::File h5_file;
        std::ofstream xdmf_file;
    };

}