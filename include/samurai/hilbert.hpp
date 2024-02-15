#pragma once

#include <vector>
#include "sfc.hpp"

class Hilbert : public SFC<Hilbert> {

    private:
        const std::vector<std::vector<std::vector<int>>> _2D_STATE = {{{1, 0, 2, 3}, {0, 0, 2, 2}},
                                                                      {{0, 3, 2, 1}, {1, 3, 1, 3}},
                                                                      {{2, 1, 0, 3}, {3, 1, 3, 1}},
                                                                      {{0, 1, 3, 2}, {2, 2, 0, 0}}};

        const std::vector<std::vector<std::vector<int>>> _3D_STATE = {{{1,2,0,6,11,4,5,6,10,4,7,10},
                                                                       {0,0,0,2,4,6,4,6,2,2,4,6}},
                                                                      {{2,6,9,0,11,4,7,1,3,4,2,3},
                                                                       {1,7,3,3,3,5,7,7,5,1,5,1}},
                                                                      {{3,0,10,6,0,8,5,6,1,8,11,2},
                                                                       {3,1,7,1,5,1,3,5,3,5,7,7}},
                                                                      {{2,7,9,11,7,8,3,10,1,8,2,6},
                                                                       {2,6,4,0,2,2,0,4,4,6,6,0}},
                                                                      {{4,8,1,9,5,0,1,9,10,2,7,10},
                                                                       {7,3,1,5,7,7,5,1,1,3,3,5}},
                                                                      {{5,8,1,0,9,6,1,4,3,7,5,3},
                                                                       {6,4,2,4,0,4,6,0,6,0,2,2}},
                                                                      {{3,0,11,9,0,10,11,9,5,2,8,4},
                                                                       {4,2,6,6,6,0,2,2,0,4,0,4}},
                                                                      {{5,7,11,8,7,6,11,10,9,3,5,4},
                                                                       {5,5,5,7,1,3,1,3,7,7,1,3}}};

        const std::vector<std::vector<std::vector<int>>> _3D_STATE_r = {{{1,2,0,11,9,10,3,4,5,7,8,6},
                                                                         {0,0,0,3,5,6,3,5,6,5,6,3}},
                                                                        {{2,0,1,6,7,8,11,9,10,4,5,3},
                                                                         {1,2,4,2,7,2,7,4,4,1,7,1}},
                                                                        {{2,0,1,6,7,8,11,9,10,4,5,3},
                                                                         {3,6,5,0,3,3,6,6,0,0,5,5}},
                                                                        {{3,8,9,0,11,6,5,10,1,2,7,4},
                                                                         {2,4,1,1,1,7,2,7,2,4,4,7}},
                                                                        {{3,8,9,0,11,6,5,10,1,2,7,4},
                                                                         {6,5,3,5,0,5,0,3,3,6,0,6}},
                                                                        {{5,7,11,9,0,4,1,6,3,8,2,10},
                                                                         {7,7,7,4,2,1,4,2,1,2,1,4}},
                                                                        {{5,7,11,9,0,4,1,6,3,8,2,10},
                                                                         {5,3,6,6,6,0,5,0,5,3,3,0}},
                                                                        {{4,6,10,8,5,0,7,1,9,3,11,2},
                                                                         {4,1,2,7,4,4,1,1,7,7,2,2}}};


    public:

        /*
        * Return the hilbert index from logical coordinate (i,j) or (i,j,k)
        *
        * Parameters:
        *             lc    : structure containing logical coordinate
        *             level : used for multiresolution
        * Return :
        *             key    : hilbert key
        */
        inline SFC_key_t getKey_2D_impl( const Logical_coord & lc, int lvl ) const {

            SFC_key_t la_clef = 0;
            
            constexpr int dim = 2;
            constexpr int twotondim = 1 << dim;

            int ind_bits[lvl][dim];
            int h_digits[lvl];

            SFC_key_t xyz[dim] = { lc.i, lc.j };

            // Set ind_bits for current point
            for (int ibit = 0; ibit < lvl; ++ibit) {
                for( size_t idim=0; idim<dim; ++idim ) {
                    ind_bits[ibit][idim] = (xyz[idim] >> ibit) & 1;
                }
            }

            // Compute Hilbert key bits
            int cur_state = 0;
            int new_state;
            for(int ibit=lvl-1; ibit>-1; --ibit) {

                // Compute s_digit by interleaving bits
                int s_digit = 0;
                for(size_t idim=0; idim<dim; ++idim)
                    s_digit += (ind_bits[ibit][idim]) << (dim - 1 - idim);

                // Compute the new state from the state diagram
                new_state = _2D_STATE[s_digit][0][cur_state];
                h_digits[ibit] = _2D_STATE[s_digit][1][cur_state];

                cur_state = new_state;
            }

            // Assemble the point's key
            for(int ibit=0; ibit<lvl; ++ibit) {
                la_clef += std::pow( twotondim, ibit ) * h_digits[ibit];
            }


            return la_clef;
        }

        inline SFC_key_t getKey_3D_impl( const Logical_coord & lc, int lvl ) const {
            SFC_key_t la_clef = 0;

            constexpr int dim = 3;

            int twotondim = 1 << dim;
            int ind_bits[lvl][dim];
            int h_digits[lvl];

            SFC_key_t xyz[dim] = { lc.i, lc.j, lc.k };

            // Set ind_bits for current point
            for (int ibit = 0; ibit < lvl; ++ibit) {
                for(size_t idim=0; idim<dim; ++idim) {
                    ind_bits[ibit][idim] = ( xyz[idim] >> ibit ) & 1;
                }
            }

            // Compute Hilbert key bits
            int cur_state = 0;
            int new_state;
            for(int ibit=lvl-1; ibit>-1; --ibit) {

                // Compute s_digit by interleaving bits
                int s_digit = 0;
                for(size_t idim=0; idim<dim; ++idim)
                    s_digit += (ind_bits[ibit][idim]) << ( dim - 1 - idim );

                // Compute the new state from the state diagram
                new_state = _3D_STATE[s_digit][0][cur_state];
                h_digits[ibit] = _3D_STATE[s_digit][1][cur_state];

                cur_state = new_state;
            }

            // Assemble the point's key
            for(int ibit=0; ibit<lvl; ++ibit) {
                la_clef += std::pow(twotondim, ibit) * h_digits[ibit];
            }

            return la_clef;
        }


};

// /*
//  * Return the logical coordinate from a hilbert key encoded in long double.
//  *
//  * Parameters:
//  *             long double hkey : hilbert key on float128
//  *             order            : hilbert order (level)
//  *             dim              : dimension should be 3D here
//  */
// inline std::vector<int> get_logical_from_hilbert_3D(long double hkey, int order, int dim){

//     int twotondim = 1 << dim;
//     std::vector<int> ind_array(dim);
//     std::vector<int> h_digits(order);
//     std::vector<std::vector<int>> ind_bits( order, std::vector<int> (dim));

//     // Compute Hilbert key bits
//     for(int ibit=0; ibit<order; ++ibit){
//         h_digits[ibit] = int( static_cast<long int>((hkey / std::pow(twotondim, ibit))) % twotondim);
//         hkey -= h_digits[ibit] * std::pow(twotondim, ibit);
//     }

//     // Compute indices bits
//     int cur_state = 0, new_state, s_digit = 0;
//     for(int ibit=order-1; ibit>-1; --ibit) {

//         // Compute the new s_digit from the state diagram
//         new_state = _3D_STATE_r[h_digits[ibit]][0][cur_state];
//         s_digit   = _3D_STATE_r[h_digits[ibit]][1][cur_state];
//         cur_state = new_state;

//         // Compute ind_bitd
//         for (int idim = 0; idim < dim; ++idim) {
//             ind_bits[ibit][idim] = (s_digit >> (dim - 1 - idim)) & 1;
//         }
//     }

//     // Set indices for current key
//     for(int ibit=0; ibit<order; ++ibit)
//         for(int idim=0; idim<dim; ++idim)
//             ind_array[idim] += ind_bits[ibit][idim] << ibit;

//     return ind_array;
// }

// /*
//  * Return the logical coordinate from a hilbert key encoded in long double.
//  *
//  * Parameters:
//  *             long double hkey : hilbert key on float128
//  *             order            : hilbert order (level)
//  *             dim              : dimension should be 3D here
//  */
// inline
// Logical_Pos_t getLogicalFromHilberKey3D( uint64_t hkey, int level ){

//     constexpr int dim = 3;

//     int twotondim = 1 << dim;
//     uint32_t ind_array[dim] = {0, 0, 0};
//     int h_digits[level];
//     int ind_bits[level][dim];

//     // Compute Hilbert key bits
//     for(int ibit=0; ibit<level; ++ibit){
//         h_digits[ibit] = int( static_cast<long int>( (hkey / std::pow( twotondim, ibit )) ) % twotondim);
//         hkey -= h_digits[ibit] * std::pow(twotondim, ibit);
//     }

//     // Compute indices bits
//     int cur_state = 0, new_state, s_digit = 0;
//     for(int ibit=level-1; ibit>-1; --ibit) {

//         // Compute the new s_digit from the state diagram
//         new_state = _HILBERT_3D_STATE_r[h_digits[ibit]][0][cur_state];
//         s_digit = _HILBERT_3D_STATE_r[h_digits[ibit]][1][cur_state];
//         cur_state = new_state;

//         // Compute ind_bitd
//         for (int idim = 0; idim < dim; ++idim) {
//             ind_bits[ibit][idim] = (s_digit >> (dim - 1 - idim)) & 1;
//         }
//     }

//     // Set indices for current key
//     for(int ibit=0; ibit<level; ++ibit)
//         for(int idim=0; idim<dim; ++idim)
//             ind_array[idim] += ind_bits[ibit][idim] << ibit;

//     return Logical_Pos_t { ind_array[0], ind_array[1], ind_array[2] };
// }