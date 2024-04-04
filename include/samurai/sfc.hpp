#pragma once

#include <cstdint>

#include "assertLogTrace.hpp"

using SFC_key_t = uint64_t;

template<class SFC_Flavor>
class SFC {

    public:
        
        /**
         * 
         * Interface to compute Space Filling Curve 1D index from
         * 2D/3D logical coordinates of cell.
         * 
        */
        template<int dim, class Coord_t>
        inline SFC_key_t getKey( const Coord_t & lc ) const{
            if constexpr ( dim == 2 ) return static_cast<const SFC_Flavor*>( this )->getKey_2D_impl( lc );
            if constexpr ( dim == 3 ) return static_cast<const SFC_Flavor*>( this )->getKey_3D_impl( lc );
        }

        /**
         * 
         * Interface to compute logical coordinate from Space Filling Curve 1D index.
         * 
        */
        template<int dim>
        inline auto getCoordinates( const SFC_key_t & clef ) const{
            if constexpr ( dim == 2 ) return static_cast<const SFC_Flavor*>( this )->getCoordinates_2D_impl( clef );
            if constexpr ( dim == 3 ) return static_cast<const SFC_Flavor*>( this )->getCoordinates_3D_impl( clef );
        }

        inline auto getName() const -> std::string {
            return static_cast<const SFC_Flavor*>( this )->getName();
        }
};