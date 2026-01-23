// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <type_traits>

namespace samurai
{
    template <class mesh_t, class value_t>
    class ScalarField;
    template <class mesh_t, class value_t, std::size_t n_comp, bool SOA>
    class VectorField;

    template <class T>
    inline constexpr bool field_like_helper = false;

    template <class mesh_t, class value_t>
    inline constexpr bool field_like_helper<ScalarField<mesh_t, value_t>> = true;

    template <class mesh_t, class value_t, std::size_t n_comp, bool SOA>
    inline constexpr bool field_like_helper<VectorField<mesh_t, value_t, n_comp, SOA>> = true;

    template <class T>
    concept field_like = field_like_helper<std::remove_cvref_t<T>>;
} // namespace samurai
