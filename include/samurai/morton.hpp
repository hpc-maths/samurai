#pragma once

#include <vector>
#include <cstdint>

#include "sfc.hpp"

class Morton : public SFC<Morton> {

    private:

        /*
        * Split bit by 3, version for 2D, used to compute morton index
        *
        * Parameters:
        *             logical_a : coordonnées logiques (i ou j)
        *
        * return :
        *              x        : unsigned 64bits with splited by 3
        */
        inline SFC_key_t splitBy3_2D( uint32_t logical_a ) const {

            SFC_key_t x = logical_a & 0xffffffff;
            x = (x | x << 16) & 0xffff0000ffff;
            x = (x | x << 8) & 0xff00ff00ff00ff;
            x = (x | x << 4) & 0xf0f0f0f0f0f0f0f;
            x = (x | x << 2) & 0x3333333333333333;
            x = (x | x << 1) & 0x5555555555555555;

            return x;
        }

        /*
        * Split bit by 3, version for 3D, used to compute morton index
        *
        * Parameters:
        *             logical_a : coordonnées logiques (i ou j ou k)
        *
        * return :
        *              x        : unsigned 64bits with splited by 3
        */
        inline SFC_key_t splitBy3_3D(uint32_t logical_a) const {

            SFC_key_t x = logical_a & 0x1fffff;

            x = (x | x << 32) & 0x1f00000000ffff;
            x = (x | x << 16) & 0x1f0000ff0000ff;
            x = (x | x <<  8) & 0x100f00f00f00f00f;
            x = (x | x <<  4) & 0x10c30c30c30c30c3;
            x = (x | x <<  2) & 0x1249249249249249;

            return x;

        }


    public:

        /*
        * Return the morton index from logical coordinate (i,j) or (i,j,k)
        *
        * Parameters:
        *             lc : structure containing logical coordinate
        *
        * Return :
        *             key    : morton key
        */
        template<class Coord_t>
        inline SFC_key_t getKey_2D_impl( const Coord_t & lc ) const {
            SFC_key_t la_clef = 0;
            la_clef |= splitBy3_2D( lc( 0 ) ) | splitBy3_2D( lc( 1 ) ) << 1;
            return la_clef;
        }

        template<class Coord_t>
        inline SFC_key_t getKey_3D_impl( const Coord_t & lc ) const {
            SFC_key_t la_clef = 0;
            la_clef |= splitBy3_3D( lc( 0 ) ) | splitBy3_3D( lc( 1 ) ) << 1 | splitBy3_3D( lc( 2 ) ) << 2;
            return la_clef;
        }

        /*
        * Return the logical coordinate (i,j) or (i,j,k) from the morton index
        *
        * Parameters:
        *             clef (in): morton index
        *
        * Return :
        *             lc  (out): 2D/3D logical coordinates
        */
        inline auto getCoordinates_2D_impl( const SFC_key_t & clef ) const {

            xt::xtensor_fixed<uint32_t, xt::xshape<2>> lc = { 0, 0 };

            // Extract coord i
            SFC_key_t keyi = clef >> 0;
            keyi &= 0x5555555555555555;
            keyi = (keyi ^ (keyi >> 1))  & 0x3333333333333333;
            keyi = (keyi ^ (keyi >> 2))  & 0x0f0f0f0f0f0f0f0f;
            keyi = (keyi ^ (keyi >> 4))  & 0x00ff00ff00ff00ff;
            keyi = (keyi ^ (keyi >> 8))  & 0x0000ffff0000ffff;
            keyi = (keyi ^ (keyi >> 16)) & 0x00000000ffffffff;
            lc( 0 ) = static_cast<uint32_t>( keyi );

            // extract coord j
            SFC_key_t keyj = clef >> 1;
            keyj &= 0x5555555555555555;
            keyj = (keyj ^ (keyj >> 1))  & 0x3333333333333333;
            keyj = (keyj ^ (keyj >> 2))  & 0x0f0f0f0f0f0f0f0f;
            keyj = (keyj ^ (keyj >> 4))  & 0x00ff00ff00ff00ff;
            keyj = (keyj ^ (keyj >> 8))  & 0x0000ffff0000ffff;
            keyj = (keyj ^ (keyj >> 16)) & 0x00000000ffffffff;
            lc( 1 ) = static_cast<uint32_t>( keyj );

            return lc;
        }

        inline auto getCoordinates_3D_impl( const SFC_key_t & clef ) const {

            xt::xtensor_fixed<uint32_t, xt::xshape<3>> lc = { 0, 0, 0 };

            // Extract coord i
            SFC_key_t keyi = clef >> 0;
            keyi &= 0x1249249249249249;
            keyi = (keyi ^ (keyi >> 2))  & 0x30c30c30c30c30c3;
            keyi = (keyi ^ (keyi >> 4))  & 0xf00f00f00f00f00f;
            keyi = (keyi ^ (keyi >> 8))  & 0x00ff0000ff0000ff;
            keyi = (keyi ^ (keyi >> 16)) & 0x00ff00000000ffff;
            keyi = (keyi ^ (keyi >> 32)) & 0x1fffff;
            lc( 0 ) = static_cast<uint32_t>( keyi ); // assert for overflow ?

            SFC_key_t keyj = clef >> 1;
            keyj &= 0x1249249249249249;
            keyj = (keyj ^ (keyj >> 2))  & 0x30c30c30c30c30c3;
            keyj = (keyj ^ (keyj >> 4))  & 0xf00f00f00f00f00f;
            keyj = (keyj ^ (keyj >> 8))  & 0x00ff0000ff0000ff;
            keyj = (keyj ^ (keyj >> 16)) & 0x00ff00000000ffff;
            keyj = (keyj ^ (keyj >> 32)) & 0x1fffff;
            lc( 1 ) = static_cast<uint32_t>( keyj );

            SFC_key_t keyk = clef >> 2;
            keyk &= 0x1249249249249249;
            keyk = (keyk ^ (keyk >> 2))  & 0x30c30c30c30c30c3;
            keyk = (keyk ^ (keyk >> 4))  & 0xf00f00f00f00f00f;
            keyk = (keyk ^ (keyk >> 8))  & 0x00ff0000ff0000ff;
            keyk = (keyk ^ (keyk >> 16)) & 0x00ff00000000ffff;
            keyk = (keyk ^ (keyk >> 32)) & 0x1fffff;
            lc( 2 ) = static_cast<uint32_t>( keyk );

            return lc;
        }

};
