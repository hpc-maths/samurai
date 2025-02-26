#include <gtest/gtest.h>

#include <samurai/hdf5.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/subset/subset_op.hpp>

#include <vector>

namespace samurai
{

    template <int dim>
    inline samurai::CellList<dim> getInitState()
    {
        assert(dim == 2);

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

        return cl;
    }

    /*
     * test MR Mesh;
     */
    TEST(mrmesh, test_nbcells2D)
    {
        constexpr int dim = 2;

        using Config = samurai::MRConfig<dim>;
        using Mesh_t = samurai::MRMesh<Config>;

        Mesh_t mesh(getInitState<dim>(), 0, 2);

        samurai::save("./", "test_mrmesh_cells_and_ghosts", mesh);

        ASSERT_EQ(mesh.min_level(), 0);
        ASSERT_EQ(mesh.max_level(), 2);

        /**
         * Need fix, this does not work.
         */
        // std::vector<size_t> nCellPerLevel_withGhost = {36, 48, 32}; // including ghost
        // for (size_t ilvl = mesh.min_level(); ilvl <= mesh.max_level(); ++ilvl)
        // {
        //     std::cerr << "cells_and_ghosts [" << ilvl << "]: " << mesh.nb_cells(ilvl, samurai::MRMeshId::cells_and_ghosts) << std::endl;
        //     // ASSERT_EQ(mesh.nb_cells(ilvl), nCellPerLevel_withGhost[ilvl]);
        // }

        std::vector<size_t> nCellPerLevel_proj = {5, 2, 0}; // proj cells
        for (size_t ilvl = mesh.min_level(); ilvl <= mesh.max_level(); ++ilvl)
        {
            // std::cerr << "proj_cells [" << ilvl << "]: " << mesh.nb_cells(ilvl, samurai::MRMeshId::proj_cells) << std::endl;
            ASSERT_EQ(mesh.nb_cells(ilvl, samurai::MRMeshId::proj_cells), nCellPerLevel_proj[ilvl]);
        }

        std::vector<size_t> nCellPerLevel_leaves = {11, 18, 8}; // not including ghost
        for (size_t ilvl = mesh.min_level(); ilvl <= mesh.max_level(); ++ilvl)
        {
            ASSERT_EQ(mesh.nb_cells(ilvl, samurai::MRMeshId::cells), nCellPerLevel_leaves[ilvl]);
        }
    }

    TEST(mrmesh, test_exist2D)
    {
        constexpr int dim = 2;

        using Config     = samurai::MRConfig<dim>;
        using Mesh_t     = samurai::MRMesh<Config>;
        using Interval_t = typename Mesh_t::interval_t;

        Mesh_t mesh(getInitState<dim>(), 0, 2);

        Interval_t i{0, 3, 0};

        // const auto val = mesh.exists(samurai::MRMeshId::cells, 1, i);
    }

    TEST(mrmesh, test_merge)
    {
        constexpr int dim = 2;

        using Config      = samurai::MRConfig<dim>;
        using Mesh_t      = samurai::MRMesh<Config>;
        using CellArray_t = Mesh_t::ca_type;

        samurai::CellList<dim> cl;
        cl[0][{0}].add_interval({0, 4});
        cl[0][{1}].add_interval({0, 1});
        cl[0][{1}].add_interval({3, 4});
        cl[1][{2}].add_interval({2, 6});

        Mesh_t mesh(cl, 0, 2);
        ASSERT_EQ(mesh.nb_cells(Mesh_t::mesh_id_t::cells), 10);

        samurai::CellList<dim> to_add_cl;
        to_add_cl[1][{3}].add_interval({2, 6});

        CellArray_t to_add_ca = {to_add_cl, false};

        mesh.merge(to_add_ca);

        // technically not sufficient to be sure the merge was done properly
        ASSERT_EQ(mesh.nb_cells(Mesh_t::mesh_id_t::cells), 14);
    }

    TEST(mrmesh, test_remove)
    {
        constexpr int dim = 2;

        using Config      = samurai::MRConfig<dim>;
        using Mesh_t      = samurai::MRMesh<Config>;
        using CellArray_t = Mesh_t::ca_type;

        samurai::CellList<dim> cl;
        cl[0][{0}].add_interval({0, 4});
        cl[0][{1}].add_interval({0, 1});
        cl[0][{1}].add_interval({3, 4});
        cl[1][{2}].add_interval({2, 6});
        cl[1][{3}].add_interval({2, 6});

        Mesh_t mesh(cl, 0, 2);
        ASSERT_EQ(mesh.nb_cells(Mesh_t::mesh_id_t::cells), 14);

        samurai::CellList<dim> to_rm_cl;
        to_rm_cl[0][{1}].add_interval({3, 4});
        to_rm_cl[1][{3}].add_interval({2, 6});

        CellArray_t to_rm_ca = {to_rm_cl, false};

        mesh.remove(to_rm_ca);

        // technically not sufficient to be sure the remove was done properly
        ASSERT_EQ(mesh.nb_cells(Mesh_t::mesh_id_t::cells), 9);
    }

    TEST(mrmesh, test_construct_subdomain)
    {
        constexpr int dim = 2;

        using Config     = samurai::MRConfig<dim>;
        using Mesh_t     = samurai::MRMesh<Config>;

        samurai::CellList<dim> cl;
        cl[0][{0}].add_interval({0, 4});
        cl[0][{1}].add_interval({0, 1});
        cl[0][{1}].add_interval({3, 4});
        cl[0][{2}].add_interval({0, 1});
        cl[0][{2}].add_interval({3, 4});

        cl[1][{2}].add_interval({2, 6});
        cl[1][{3}].add_interval({2, 6});
        cl[1][{4}].add_interval({2, 4});
        cl[1][{4}].add_interval({5, 6});
        cl[1][{5}].add_interval({2, 6});
        cl[1][{6}].add_interval({6, 8});
        cl[1][{7}].add_interval({6, 7});

        Mesh_t mesh( cl, 0, 1);

        bool intersect_found = false;
        const auto & subdom = mesh.subdomain();
        for(std::size_t level=mesh.min_level(); level<mesh.max_level(); ++level ){
            auto intersect = difference( mesh[Mesh_t::mesh_id_t::cells][level], subdom ).on(level);
            size_t ninter = 0;
            intersect( [&]([[maybe_unused]]const auto & interval, [[maybe_unused]]const auto & index ){
                ninter += 1;
            });

            if( ninter >  0) { intersect_found = true; }
        }

        ASSERT_FALSE( intersect_found );

    }

    TEST(mrmesh, test_construct_union)
    {
        constexpr int dim = 2;

        using Config = samurai::MRConfig<dim>;
        using Mesh_t = samurai::MRMesh<Config>;
        using cl_t   = Mesh_t::cl_type;
        using ca_t   = Mesh_t::ca_type;

        Mesh_t mesh( getInitState<dim>(), 0, 2);

        auto un = mesh.get_union();

        ASSERT_EQ( un[ 0 ].nb_cells(), 5);
        {
            cl_t cl;
            cl[0][{1}].add_interval({1,3});
            cl[0][{2}].add_interval({1,3});
            cl[0][{3}].add_interval({3,4});
            ca_t ca = { cl, false };
            auto tmp_ = samurai::difference( ca[0], un[0] );
            
            size_t ni = 0;
            tmp_( [&]([[maybe_unused]]const auto & interval, [[maybe_unused]]const auto & index){
                ni +=1;
            });
            ASSERT_EQ( ni, 0);
        }

        ASSERT_EQ( un[ 1 ].nb_cells(), 2);
        {
            cl_t cl;
            cl[1][{4}].add_interval({4,5});
            cl[1][{7}].add_interval({7,8});
            ca_t ca = { cl, false };
            auto tmp_ = samurai::difference( ca[1], un[1] );
            
            size_t ni = 0;
            tmp_( [&]([[maybe_unused]]const auto & interval, [[maybe_unused]]const auto & index){
                ni +=1;
            });
            ASSERT_EQ( ni, 0);
        }

        ASSERT_EQ( un[ 2 ].nb_cells(), 0);


    }

}