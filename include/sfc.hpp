#pragma once

#include <cstdint>

struct Logical_coord{
    uint32_t i, j, k;
};

template<class SFC_Flavor>
class SFC {

    public:
        using SFC_key_t = uint64_t;
        
        /**
         * 
         * Interface to compute Space Filling Curve 1D index from
         * 2D/3D logical coordinates of cell.
         * 
        */
        template<int dim, typename...Args>
        inline SFC_key_t getKey( const Logical_coord & lc, const Args&... kw ) const{
            // return SFC_Flavor::getKey_impl( lc );
            if constexpr ( dim == 2 ) return static_cast<const SFC_Flavor*>(this)->getKey_2D_impl( lc, kw... );
            if constexpr ( dim == 3 ) return static_cast<const SFC_Flavor*>(this)->getKey_3D_impl( lc, kw... );
        };

        /**
         * 
         * Interface to compute logical coordinate from Space Filling Curve 1D index.
         * 
        */
        template<int dim>
        inline Logical_coord getCoordinates( const SFC_key_t & clef ) const{
            // return SFC_Flavor::getKey_impl( lc );
            if constexpr ( dim == 2 ) return static_cast<const SFC_Flavor*>(this)->getCoordinates_2D_impl( clef );
            if constexpr ( dim == 3 ) return static_cast<const SFC_Flavor*>(this)->getCoordinates_3D_impl( clef );
        };

};