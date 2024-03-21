#include <algorithm>
#include <fmt/format.h>
#include <string>

#include <fstream>

#define WITH_STATS
#if defined(WITH_STATS)
#include <nlohmann/json.hpp>
using json = nlohmann::json;
#endif

namespace samurai
{
#if defined(WITH_STATS)
    struct Statistics
    {
        Statistics( const std::string & filename, int save_all = 10)
            : _outfile( filename )
            , save_all(save_all)
            , icurrent(0)
        {
            std::cerr << "\t> Constucteur [Statistics] " << std::endl;
        }

        template <class Mesh>
        void operator()( std::string test_case, Mesh& mesh )
        {

            std::cerr << "\t> Constucteur [operator()] " << std::endl;

            icurrent++;
            using mesh_id_t = typename Mesh::mesh_id_t;
            auto ca         = mesh[mesh_id_t::cells];
            std::size_t dim = Mesh::dim;

            std::size_t min_level = ca.min_level();
            std::size_t max_level = ca.max_level();

            size_t n_neighbours = mesh.mpi_neighbourhood().size();
            size_t n_cells_l    = mesh.nb_cells( Mesh::mesh_id_t::cells );
            size_t n_cells_g    = mesh.nb_cells( Mesh::mesh_id_t::reference ) - mesh.nb_cells( Mesh::mesh_id_t::cells );

            auto comp = [](const auto& a, const auto& b)
            {
                return a.size() < b.size();
            };

            json by_level;
            for (std::size_t l = min_level; l <= max_level; ++l)
            {
                json result;
                result["cells"] = ca[l].nb_cells();

                for (std::size_t d = 0; d < dim; ++d)
                {
                    auto dim_str                           = fmt::format("axis-{}", d);
                    result[dim_str]["number of intervals"] = ca[l][d].size();

                    auto minmax                                  = std::minmax_element(ca[l][d].cbegin(), ca[l][d].cend(), comp);
                    result[dim_str]["cells per interval"]["min"] = (minmax.first)->size();
                    result[dim_str]["cells per interval"]["max"] = (minmax.second)->size();
                }

                for (std::size_t d = 1; d < dim; ++d)
                {
                    auto dim_str = fmt::format("axis-{}", d);
                    auto offsets = ca[l].offsets(1);
                    std::adjacent_difference(offsets.begin(), offsets.end(), offsets.begin());
                    auto minmax = std::minmax_element(offsets.cbegin() + 1, offsets.cend());

                    result[dim_str]["number of intervals per component"]["min"] = *minmax.first;
                    result[dim_str]["number of intervals per component"]["max"] = *minmax.second;
                }

                by_level[fmt::format("{:02}", l)] = result;
            }

            if (stats.contains(test_case))
            {
                stats[test_case].push_back({
                    {"min_level", min_level},
                    {"max_level", max_level},
                    {"n_neighbours", n_neighbours},
                    {"n_cells_g", n_cells_g}, // ghost ?
                    {"n_cells_l", n_cells_l}, // leaves
                    {"by_level",  by_level }
                });
            }
            else
            {
                json out = json::array();
                out.push_back({
                    {"min_level", min_level},
                    {"max_level", max_level},
                    {"n_neighbours", n_neighbours},
                    {"n_cells_g", n_cells_g}, // ghost ?
                    {"n_cells_l", n_cells_l}, // leaves
                    {"by_level",  by_level }
                });
                stats[test_case] = out;
            }

            if (icurrent == save_all)
            {
                std::ofstream ofile( _outfile );
                ofile << std::setw(4) << stats << std::endl;
                icurrent = 0;
            }
        }

        ~Statistics()
        {
            std::ofstream file( _outfile );
            file << std::setw(4) << stats;
        }

        std::string _outfile;
        json stats;
        std::size_t icurrent;
        int save_all;
    };

#else
    template <class Mesh>
    void statistics(const std::string& test_case, const Mesh& mesh){};
#endif
}
