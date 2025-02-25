#include <gtest/gtest.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xfixed.hpp>

#include <samurai/samurai.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/interval.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>

#ifdef SAMURAI_WITH_MPI
#include <samurai/load_balancing.hpp>
#endif

#include <vector>

namespace samurai {

#ifdef SAMURAI_WITH_MPI
    /*
    * test cmptLoad;
    */
    TEST(loadBalance, cmptLoad){

        constexpr int dim = 2;

        using Config = samurai::MRConfig<dim>;
        using Mesh_t = samurai::MRMesh<Config>;

        samurai::CellList<dim> cl;
        cl[0][{0}].add_interval({0, 4});
        cl[0][{1}].add_interval({0, 1});
        cl[0][{1}].add_interval({3, 4});
        cl[1][{2}].add_interval({2, 6});
        cl[2][{6}].add_interval({4, 6});
        cl[2][{6}].add_interval({10, 12});

        Mesh_t mesh( cl, 0, 2 );

        auto ncells = cmptLoad<BalanceElement_t::CELL>( mesh );
        auto ninter = cmptLoad<BalanceElement_t::INTERVAL>( mesh );

        ASSERT_EQ( ncells, 14 );
        ASSERT_EQ( ninter,  6 );

    }

    /**
     * Be aware that the cmptInterface require 2:1 balance for this test
    */
    TEST(loadBalance, cmptInterface){
        constexpr int dim = 2;
        
        using Config = samurai::MRConfig<dim>;
        using Mesh_t = samurai::MRMesh<Config>;

        samurai::CellList<dim> cl_a, cl_b;

        {
            cl_a[0][{0}].add_interval({0, 4});   // level 0 
            
            cl_a[1][{2}].add_interval({0, 2});   // level 1 
            cl_a[1][{3}].add_interval({0, 2});   // level 1
            cl_a[1][{2}].add_interval({6, 8});   // level 1 
            cl_a[1][{3}].add_interval({6, 8});   // level 1

            cl_a[1][{2}].add_interval({2, 6});   // level 1
            cl_a[2][{6}].add_interval({4, 6});   // level 2
            cl_a[2][{6}].add_interval({10, 12}); // level 2 
        }

        {
            // cl_b[0][{2}].add_interval({0, 4});   // level 0 
            cl_b[0][{3}].add_interval({0, 4});   // level 0
            
            cl_b[1][{4}].add_interval({0, 8});   // level 1 
            cl_b[1][{5}].add_interval({0, 8});   // level 1

            cl_b[1][{3}].add_interval({3, 5});   // level 1
            cl_b[2][{7}].add_interval({4, 6});   // level 2
            cl_b[2][{7}].add_interval({10, 12}); // level 2 
        }

        Mesh_t mesh_a ( cl_a, 0, 2 ), mesh_b (cl_b, 0, 2 );

        auto interface = cmptInterface<dim, Direction_t::FACE, 1>( mesh_a, mesh_b );

        ASSERT_EQ( interface.nb_cells(), 10 );

    }

    TEST(loadBalance, cellExists){
        constexpr int dim = 2;
        
        using Config = samurai::MRConfig<dim, 1>;
        using Mesh_t = samurai::MRMesh<Config>;

        samurai::CellList<dim> cl_a, cl_b;

        {
            cl_a[0][{0}].add_interval({0, 4});   // level 0 
            
            cl_a[1][{2}].add_interval({0, 2});   // level 1 
            cl_a[1][{3}].add_interval({0, 2});   // level 1
            cl_a[1][{2}].add_interval({6, 8});   // level 1 
            cl_a[1][{3}].add_interval({6, 8});   // level 1

            cl_a[1][{2}].add_interval({2, 6});   // level 1
            cl_a[2][{6}].add_interval({4, 6});   // level 2
            cl_a[2][{6}].add_interval({10, 12}); // level 2 
        }

        Mesh_t mesh ( cl_a, 0, 2 );

        {
            xt::xtensor_fixed<int, xt::xshape<Mesh_t::dim>> ij = {{3, 2}};
            ASSERT_TRUE( samurai::cellExists( mesh, Mesh_t::mesh_id_t::cells, static_cast<size_t>(1), ij ) );
        }

        {
            xt::xtensor_fixed<int, xt::xshape<Mesh_t::dim>> ij = {{0, 1}};
            ASSERT_FALSE( samurai::cellExists( mesh, Mesh_t::mesh_id_t::cells, static_cast<size_t>(0), ij ) );
        }

        {
            xt::xtensor_fixed<int, xt::xshape<Mesh_t::dim>> ij = {{0, 2}};
            ASSERT_TRUE( samurai::cellExists( mesh, Mesh_t::mesh_id_t::reference, static_cast<size_t>(0), ij ) );
        }

        auto coords      = make_field<int, dim, false>("coordinates", mesh);
        auto level_field = make_field<std::size_t, 1, false>("level", mesh);
        for_each_cell(mesh[Mesh_t::mesh_id_t::reference], [&](auto& cell) {
                            if constexpr ( dim == 1 )
                            {
                                coords[cell] = cell.indices[0];
                            }
                            else
                            {
                                coords[cell] = cell.indices;
                            }
                            level_field[cell] = cell.level;
                        });

        samurai::save( "./", "test-lb-cellExists", {true, true}, mesh, coords, level_field );

    }

#endif
}
