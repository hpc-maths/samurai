#include <gtest/gtest.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xfixed.hpp>

#include <samurai/samurai.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/interval.hpp>

#include <vector>

namespace samurai
{

    /*
    * test MR Mesh;
    */
    TEST(mrmesh, test_nbcells2D){

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
        cl[2][{9}].add_interval({8, 10});
        cl[2][{14}].add_interval({14, 16});
        cl[2][{15}].add_interval({14, 16});

        Mesh_t mesh( cl, 0, 2 );

        ASSERT_EQ( mesh.min_level(), 0 ); 
        ASSERT_EQ( mesh.max_level(), 2 );

        std::vector<size_t> nCellPerLevel_withGhost = { 36, 48, 32 }; // including ghost
        for( size_t ilvl=mesh.min_level(); ilvl<=mesh.max_level(); ++ilvl ){
            ASSERT_EQ( mesh.nb_cells( mesh.min_level() ), 
                       nCellPerLevel_withGhost[ mesh.min_level() ] );
        }

        std::vector<size_t> nCellPerLevel_leaves = { 11, 18, 8 }; // not including ghost
        for( size_t ilvl=mesh.min_level(); ilvl<=mesh.max_level(); ++ilvl ){
            ASSERT_EQ( mesh.nb_cells( mesh.min_level(), samurai::MRMeshId::cells ), 
                       nCellPerLevel_leaves[ mesh.min_level() ] );
        }

    }

    TEST(mrmesh, test_exist2D){

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
        cl[2][{9}].add_interval({8, 10});
        cl[2][{14}].add_interval({14, 16});
        cl[2][{15}].add_interval({14, 16});

        Mesh_t mesh( cl, 0, 2 );

        Interval<int, int> i{0, 3, 0};
        
        // mesh.exists( samurai::MRMeshId::cells, 1, )
    }

    TEST(mrmesh, test_merge){

        constexpr int dim = 2;

        using Config = samurai::MRConfig<dim>;
        using Mesh_t = samurai::MRMesh<Config>;
        using CellArray_t = Mesh_t::ca_type;

        samurai::CellList<dim> cl;
        cl[0][{0}].add_interval({0, 4});
        cl[0][{1}].add_interval({0, 1});
        cl[0][{1}].add_interval({3, 4});
        cl[1][{2}].add_interval({2, 6});

        Mesh_t mesh( cl, 0, 2 );
        ASSERT_EQ( mesh.nb_cells( Mesh_t::mesh_id_t::cells ), 10 );

        samurai::CellList<dim> to_add_cl;
        to_add_cl[1][{3}].add_interval({2, 6});

        CellArray_t to_add_ca = { to_add_cl, false };

        mesh.merge( to_add_ca );

        // technically not sufficient to be sure the merge was done properly
        ASSERT_EQ( mesh.nb_cells( Mesh_t::mesh_id_t::cells ), 14 );

    }

    TEST(mrmesh, test_remove){

        constexpr int dim = 2;

        using Config = samurai::MRConfig<dim>;
        using Mesh_t = samurai::MRMesh<Config>;
        using CellArray_t = Mesh_t::ca_type;

        samurai::CellList<dim> cl;
        cl[0][{0}].add_interval({0, 4});
        cl[0][{1}].add_interval({0, 1});
        cl[0][{1}].add_interval({3, 4});
        cl[1][{2}].add_interval({2, 6});
        cl[1][{3}].add_interval({2, 6});

        Mesh_t mesh( cl, 0, 2 );
        ASSERT_EQ( mesh.nb_cells( Mesh_t::mesh_id_t::cells ), 14 );

        samurai::CellList<dim> to_rm_cl;
        to_rm_cl[0][{1}].add_interval({3,4}); 
        to_rm_cl[1][{3}].add_interval({2, 6});
        
        CellArray_t to_rm_ca = { to_rm_cl, false };

        mesh.remove( to_rm_ca );

        // technically not sufficient to be sure the remove was done properly
        ASSERT_EQ( mesh.nb_cells( Mesh_t::mesh_id_t::cells ), 9 );

    }



}