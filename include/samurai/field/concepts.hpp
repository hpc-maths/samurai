// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <type_traits>

#include <xtensor/containers/xtensor.hpp>

namespace samurai
{
    template <class mesh_t, class value_t>
    class ScalarField;
    template <class mesh_t, class value_t, std::size_t n_comp, bool SOA>
    class VectorField;

    template <class T>
    constexpr bool field_like_helper = false;

    template <class mesh_t, class value_t>
    constexpr bool field_like_helper<ScalarField<mesh_t, value_t>> = true;

    template <class mesh_t, class value_t, std::size_t n_comp, bool SOA>
    constexpr bool field_like_helper<VectorField<mesh_t, value_t, n_comp, SOA>> = true;

    template <class T>
    concept field_like = field_like_helper<std::remove_cvref_t<T>>;

    namespace detail
    {
        template <class Field>
        struct inner_field_types;
    }

    // Base concept for valid mesh and value type parameters (shared by scalar and vector fields)
    // Checks that D is a field_like type with valid mesh and value types
    template <class D>
    concept valid_field_mesh_and_value = field_like<D> && requires {
        typename detail::inner_field_types<std::remove_cvref_t<D>>::mesh_t;
        typename detail::inner_field_types<std::remove_cvref_t<D>>::value_type;
    };

    template <class T>
    constexpr bool xtensor_like_helper = false;

    template <class EC, std::size_t N, xt::layout_type L, class Tag>
    constexpr bool xtensor_like_helper<xt::xtensor_container<EC, N, L, Tag>> = true;

    template <class T>
    concept is_xtensor_container = xtensor_like_helper<std::remove_cvref_t<T>>;
} // namespace samurai
