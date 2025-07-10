// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

namespace samurai
{
    template <class mesh_t, class value_t, std::size_t n_comp, bool SOA>
    class VectorField;

    template <class mesh_t, class value_t>
    class ScalarField;

    template <class T>
    struct is_field_impl : std::false_type
    {
    };

    template <class mesh_t, class value_t, std::size_t n_comp, bool SOA>
    struct is_field_impl<VectorField<mesh_t, value_t, n_comp, SOA>> : std::true_type
    {
    };

    template <class mesh_t, class value_t>
    struct is_field_impl<ScalarField<mesh_t, value_t>> : std::true_type
    {
    };

    template <class T>
    inline constexpr bool is_field_v = is_field_impl<std::decay_t<T>>::value;

    template <typename T>
    concept IsField = is_field_v<T>;
}
