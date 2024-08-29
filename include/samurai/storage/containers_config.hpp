// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

//----------------------------------------------------------------------------//
// This file contains the configuration of the actual storage containers used //
// according to the compilation variables.                                    //
//----------------------------------------------------------------------------//

// #define SAMURAI_FIELD_CONTAINER_EIGEN3

// clang-format off
#include "std/algebraic_array.hpp"
#ifdef SAMURAI_FIELD_CONTAINER_EIGEN3
    // #define EIGEN_DEFAULT_DENSE_INDEX_TYPE int64_t
    #include "eigen/eigen.hpp"
    #include "eigen/eigen_static.hpp"

    #define FLUX_CONTAINER_eigen
    #define TMP_STATIC_MATRIX_CONTAINER_eigen
#else
    #include "xtensor/xtensor.hpp"
#endif
#include "xtensor/xtensor_static.hpp"

// clang-format on

namespace samurai
{
    //-----------------//
    // Field container //
    //-----------------//

#if defined(SAMURAI_FIELD_CONTAINER_EIGEN3)

    template <class value_type, std::size_t size = 1, bool SOA = false>
    using field_data_storage_t = eigen_container<value_type, size, SOA>;

    template <class value_type, std::size_t size, bool SOA>
    using local_field_data_t = eigen_collapsable_static_array<value_type, size, SOA>;

#else // SAMURAI_FIELD_CONTAINER_XTENSOR

    template <class value_type, std::size_t size = 1, bool SOA = false>
    using field_data_storage_t = xtensor_container<value_type, size, SOA>;

    template <class value_type, std::size_t size, bool>
    using local_field_data_t = xtensor_collapsable_static_array<value_type, size>;

#endif

    //--------------//
    // Static array //
    //--------------//

    template <class value_type, std::size_t size, bool SOA>
#if defined(FLUX_CONTAINER_array)
    using Array = StdArrayWrapper<value_type, size>;
#elif defined(FLUX_CONTAINER_eigen)
    using Array              = eigen_static_array<value_type, size, SOA>;
#else // FLUX_CONTAINER_xtensor
    using Array = xtensor_static_array<value_type, size>;
#endif

    template <class value_type, std::size_t size, bool SOA>
    using CollapsArray = CollapsableArray<Array<value_type, size, SOA>, value_type, size>;

    //---------------//
    // Static matrix //
    //---------------//

#if defined(TMP_STATIC_MATRIX_CONTAINER_eigen)

    template <class value_type, std::size_t rows, std::size_t cols>
    using Matrix = eigen_static_matrix<value_type, rows, cols>;

    // template <class value_type, std::size_t rows, std::size_t cols>
    // using CollapsMatrix = eigen_collapsable_static_matrix<value_type, rows, cols>;

#else // TMP_STATIC_MATRIX_CONTAINER_xtensor

    template <class value_type, std::size_t rows, std::size_t cols>
    using Matrix = xtensor_static_matrix<value_type, rows, cols>;

#endif

    template <class value_type, std::size_t rows, std::size_t cols>
    using CollapsMatrix = CollapsableMatrix<Matrix<value_type, rows, cols>, value_type, rows, cols>;

}
