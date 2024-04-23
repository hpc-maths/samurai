#include <gtest/gtest.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xfixed.hpp>

#include <samurai/hilbert.hpp>
#include <samurai/morton.hpp>

#include <vector>

namespace samurai
{
    using Coord_2D_t = xt::xtensor_fixed<uint32_t, xt::xshape<2>>;
    using Coord_3D_t = xt::xtensor_fixed<uint32_t, xt::xshape<3>>;

    /*
     * test computation of morton indexes 2D
     */
    TEST(sfc, morton2D_getKey)
    {
        constexpr int dim = 2;

        SFC<Morton> morton;

        Coord_2D_t ij = {0, 0};
        auto key1     = morton.getKey<dim>(ij);
        ASSERT_EQ(key1, 0);

        ij        = {2, 1};
        auto key2 = morton.getKey<dim>(ij);
        ASSERT_EQ(key2, 6);

        ij        = {5, 6};
        auto key3 = morton.getKey<dim>(ij);
        ASSERT_EQ(key3, 57);

        ij        = {0, 4};
        auto key4 = morton.getKey<dim>(ij);
        ASSERT_EQ(key4, 32);

        ij        = {4, 0};
        auto key5 = morton.getKey<dim>(ij);
        ASSERT_EQ(key5, 16);
    }

    /*
     * test computation of (i,j) logical coordinate from morton indexes 2D
     */
    TEST(sfc, morton2D_getCoordinates)
    {
        constexpr int dim = 2;

        SFC<Morton> morton;

        auto ij = morton.getCoordinates<dim>(static_cast<SFC_key_t>(0));
        ASSERT_EQ(ij(0), 0);
        ASSERT_EQ(ij(1), 0);

        auto ij2 = morton.getCoordinates<dim>(static_cast<SFC_key_t>(31));
        ASSERT_EQ(ij2(0), 7);
        ASSERT_EQ(ij2(1), 3);

        auto ij3 = morton.getCoordinates<dim>(static_cast<SFC_key_t>(51));
        ASSERT_EQ(ij3(0), 5);
        ASSERT_EQ(ij3(1), 5);

        auto ij4 = morton.getCoordinates<dim>(static_cast<SFC_key_t>(39));
        ASSERT_EQ(ij4(0), 3);
        ASSERT_EQ(ij4(1), 5);
    }

    /*
     * test computation of morton indexes 3D
     */
    TEST(sfc, morton3D_getKey)
    {
        constexpr int dim = 3;

        SFC<Morton> morton;

        Coord_3D_t ijk = {0, 0, 0};
        auto key1      = morton.getKey<dim>(ijk);
        ASSERT_EQ(key1, 0);

        ijk       = {1, 1, 0};
        auto key2 = morton.getKey<dim>(ijk);
        ASSERT_EQ(key2, 3);

        ijk       = {0, 0, 1};
        auto key3 = morton.getKey<dim>(ijk);
        ASSERT_EQ(key3, 4);

        ijk       = {5, 9, 1};
        auto key4 = morton.getKey<dim>(ijk);
        ASSERT_EQ(key4, 1095);
    }

    /*
     * test computation of (i,j,k) logical coordinates from morton indexes 3D
     */
    TEST(sfc, morton3D_getCoordinates)
    {
        constexpr int dim = 3;

        SFC<Morton> morton;

        auto ijk = morton.getCoordinates<dim>(static_cast<SFC_key_t>(1095));
        ASSERT_EQ(ijk(0), 5);
        ASSERT_EQ(ijk(1), 9);
        ASSERT_EQ(ijk(2), 1);
    }

    /*
     * test computation of hilbert key for 2D
     */
    TEST(sfc, hilbert2D_getKey)
    {
        constexpr int dim = 2;

        SFC<Hilbert> hilbert;

        Coord_2D_t ij = {0, 0};
        ASSERT_EQ(hilbert.getKey<dim>(ij), static_cast<SFC_key_t>(0));

        ij = {1, 1};
        ASSERT_EQ(hilbert.getKey<dim>(ij), static_cast<SFC_key_t>(2));

        ij = {1, 2};
        ASSERT_EQ(hilbert.getKey<dim>(ij), static_cast<SFC_key_t>(7));

        ij = {3, 1};
        ASSERT_EQ(hilbert.getKey<dim>(ij), static_cast<SFC_key_t>(12));

        ij = {3, 4};
        ASSERT_EQ(hilbert.getKey<dim>(ij), static_cast<SFC_key_t>(53));

        ij = {3, 5};
        ASSERT_EQ(hilbert.getKey<dim>(ij), static_cast<SFC_key_t>(52));

        ij = {3, 6};
        ASSERT_EQ(hilbert.getKey<dim>(ij), static_cast<SFC_key_t>(51));
    }

    /*
     * test computation of hilbert key & back to {i,j,k}
     */
    // TEST(tests_hilbert, test_consistency_hilbert3D){

    //     {
    //         Logical_Pos_t pt1 {0, 0 ,0};
    //         auto hk1 = getHilbertKey3D<uint64_t>( pt1, 0 );
    //         auto pt1_r = getLogicalFromHilberKey3D( hk1, 0 );
    //         ASSERT_EQ( pt1.i, pt1_r.i );
    //         ASSERT_EQ( pt1.j, pt1_r.j );
    //         ASSERT_EQ( pt1.k, pt1_r.k );
    //     }

    //     {
    //         Logical_Pos_t pt1 {2, 3, 1};
    //         auto hk1 = getHilbertKey3D<uint64_t>( pt1, 2 );
    //         auto pt1_r = getLogicalFromHilberKey3D( hk1, 2 );
    //         ASSERT_EQ( pt1.i, pt1_r.i );
    //         ASSERT_EQ( pt1.j, pt1_r.j );
    //         ASSERT_EQ( pt1.k, pt1_r.k );
    //     }

    //     {
    //         Logical_Pos_t pt1 {8, 14, 28};
    //         auto hk1 = getHilbertKey3D<uint64_t>( pt1, 6 );
    //         auto pt1_r = getLogicalFromHilberKey3D( hk1, 6 );
    //         ASSERT_EQ( pt1.i, pt1_r.i );
    //         ASSERT_EQ( pt1.j, pt1_r.j );
    //         ASSERT_EQ( pt1.k, pt1_r.k );
    //     }

    // }

    /*
     * test computation of morton indexes 3D
     */
    // TEST(tests_hilbert, test_consistency_hilbert3D_old){

    // {
    //     auto hk1 = get_hilbert_key_3D( {0, 0 ,0}, 0 );
    //     auto pt1_r = get_logical_from_hilbert_3D( hk1, 0, 3);
    //     ASSERT_EQ( 0, pt1_r[0] );
    //     ASSERT_EQ( 0, pt1_r[1] );
    //     ASSERT_EQ( 0, pt1_r[2] );
    // }

}