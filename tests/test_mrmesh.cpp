#include <gtest/gtest.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xfixed.hpp>

#include <samurai/samurai.hpp>
#include <samurai/mr/mesh.hpp>

#include <vector>

namespace samurai
{

    /*
    * test MR Mesh;
    */
    TEST(mrmesh, test_nbcells){

        constexpr int dim = 2;

        using Config = samurai::MRConfig<dim>;
        using Mesh_t = samurai::MRMesh<Config>;

        samurai::CellList<dim> cl;
        cl[0][{0}].add_interval({0, 4});
        cl[0][{1}].add_interval({0, 1});
        cl[0][{1}].add_interval({3, 4});
        cl[0][{2}].add_interval({0, 1});
        cl[0][{2}].add_interval({3, 4});
        cl[0][{3}].add_interval({0, 3});

        cl[1][{2}].add_interval({2, 6});
        cl[1][{3}].add_interval({2, 6});
        cl[1][{4}].add_interval({2, 4});
        cl[1][{4}].add_interval({5, 6});
        cl[1][{5}].add_interval({2, 6});
        cl[1][{6}].add_interval({6, 8});
        cl[1][{7}].add_interval({6, 7});

        cl[2][{8}].add_interval({8, 10});
        cl[2][{8}].add_interval({8, 10});
        cl[2][{14}].add_interval({14, 16});
        cl[2][{15}].add_interval({14, 16});

        Mesh_t mesh( cl, 2, 4 );

        ASSERT_EQ( mesh.min_level(), 2 ); 
        ASSERT_EQ( mesh.max_level(), 4 );

        std::vector<size_t> nCellPerLevel = { 11, 18, 8 };
        for( size_t ilvl=mesh.min_level(); ilvl<=mesh.max_level(); ++ilvl ){
            std::cerr << "\t> NbCells level (" << ilvl << ") : " << mesh.nb_cells( 0 ) << std::endl;
            ASSERT_EQ( mesh.nb_cells( ilvl - mesh.min_level() ), nCellPerLevel[ ilvl - mesh.min_level() ] );
        }

        std::cerr << "\t> Number of cells : " << mesh.nb_cells() << std::endl;


    }

}