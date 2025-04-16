// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

//----------------------------------------------------------------------------//
// This file contains the configuration of the actual storage containers used //
// according to the compilation variables.                                    //
//----------------------------------------------------------------------------//

// clang-format off
#include "std/algebraic_array.hpp"
#if defined(SAMURAI_FIELD_CONTAINER_EIGEN3)
    // #define EIGEN_DEFAULT_DENSE_INDEX_TYPE int64_t
    #include "eigen/eigen.hpp"
#else
    //#if XTENSOR_VERSION_MINOR < 26
        #include "xtensor/xtensor.hpp"
    //#else
    //    #include "xtensor/containers/xtensor.hpp"
    //#endif
#endif
#include "xtensor/xtensor_static.hpp"

#if defined(SAMURAI_FIELD_CONTAINER_EIGEN3) || defined(SAMURAI_FLUX_CONTAINER_EIGEN3) || defined(SAMURAI_STATIC_MAT_CONTAINER_EIGEN3)
    #include "eigen/eigen_static.hpp"
#endif

// clang-format on

#if XTENSOR_VERSION_MINOR < 26
#else
using namespace xt;
#endif

namespace samurai
{
    //-----------------//
    // Field container //
    //-----------------//

#if defined(SAMURAI_FIELD_CONTAINER_EIGEN3)

    template <class value_type, std::size_t size = 1, bool SOA = false, bool can_collapse = true>
    using field_data_storage_t = eigen_container<value_type, size, SOA, can_collapse>;

    template <class value_type, std::size_t size, bool SOA = false, bool can_collapse = true>
    using local_field_data_t = eigen_collapsable_static_array<value_type, size, SOA, can_collapse>;

    template <class T>
    using default_view_t = Eigen::IndexedView<T, Eigen::internal::ArithmeticSequenceRange<16777215, -1, 16777215>, Eigen::internal::SingleRange<0>>;
#else // SAMURAI_FIELD_CONTAINER_XTENSOR

    template <class value_type, std::size_t size = 1, bool SOA = false, bool can_collapse = true>
    using field_data_storage_t = xtensor_container<value_type, size, SOA, can_collapse>;

    template <class value_type, std::size_t size, bool SOA = false, bool can_collapse = true>
    using local_field_data_t = xtensor_collapsable_static_array<value_type, size, can_collapse>;

    template <class T>
    using default_view_t = xt::xview<T&, xt::xstepped_range<long>>;
#endif

    //--------------//
    // Static array //
    //--------------//

    template <class value_type, std::size_t size, bool SOA = false>
#if defined(SAMURAI_FIELD_CONTAINER_EIGEN3)
    using Array = eigen_static_array<value_type, size, SOA>;
#else // SAMURAI_FIELD_CONTAINER_XTENSOR
    using Array = xtensor_static_array<value_type, size>;
#endif

    template <class value_type, std::size_t size, bool SOA = false, bool can_collapse = true>
    using CollapsArray = CollapsableArray<Array<value_type, size, SOA>, value_type, size, can_collapse>;

    //----------------//
    // Flux container //
    //----------------//

    template <class value_type, std::size_t size>
#if defined(SAMURAI_FLUX_CONTAINER_ARRAY)
    using flux_array_t    = StdArrayWrapper<value_type, size>;
    using flux_index_type = std::size_t;
#elif defined(SAMURAI_FLUX_CONTAINER_EIGEN3)
    using flux_array_t    = eigen_static_array<value_type, size, false>;
    using flux_index_type = Eigen::Index;
#else // SAMURAI_FLUX_CONTAINER_XTENSOR
    using flux_array_t    = xtensor_static_array<value_type, size>;
    using flux_index_type = std::size_t;
#endif

    template <class value_type, std::size_t size, bool can_collapse>
    using CollapsFluxArray = CollapsableArray<flux_array_t<value_type, size>, value_type, size, can_collapse>;

    //---------------//
    // Static matrix //
    //---------------//

#if defined(SAMURAI_STATIC_MAT_CONTAINER_EIGEN3)
    template <class value_type, std::size_t rows, std::size_t cols>
    using Matrix = eigen_static_matrix<value_type, rows, cols>;

    // template <class value_type, std::size_t rows, std::size_t cols>
    // using CollapsMatrix = eigen_collapsable_static_matrix<value_type, rows, cols>;

#else // SAMURAI_STATIC_MAT_CONTAINER_XTENSOR
    template <class value_type, std::size_t rows, std::size_t cols>
    using Matrix = xtensor_static_matrix<value_type, rows, cols>;
#endif

    template <class value_type, std::size_t rows, std::size_t cols, bool can_collapse>
    using CollapsMatrix = CollapsableMatrix<Matrix<value_type, rows, cols>, value_type, rows, cols, can_collapse>;

}
