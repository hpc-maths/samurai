// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

namespace samurai
{
    // MESH CONCEPTS
    //////////////////////////////////////////////////////////////
    template <std::size_t dim, class TInterval>
    class LevelCellArray;

    template <std::size_t dim, class TInterval, std::size_t max_size>
    class CellArray;

    template <class D, class Config>
    class Mesh_base;

    template <class Config>
    class UniformMesh;

    template <class T>
    struct is_mesh_impl : std::false_type
    {
    };

    template <class T>
        requires std::is_base_of_v<Mesh_base<std::decay_t<T>, typename std::decay_t<T>::config>, std::decay_t<T>>
    struct is_mesh_impl<T> : std::true_type
    {
    };

    template <class Config>
    struct is_mesh_impl<UniformMesh<Config>> : std::true_type
    {
    };

    template <std::size_t dim, class TInterval>
    struct is_mesh_impl<LevelCellArray<dim, TInterval>> : std::true_type
    {
    };

    template <std::size_t dim, class TInterval, std::size_t max_size>
    struct is_mesh_impl<CellArray<dim, TInterval, max_size>> : std::true_type
    {
    };

    template <class T>
    constexpr bool is_mesh_impl_v{is_mesh_impl<std::decay_t<T>>::value};

    template <typename T>
    concept IsMesh = is_mesh_impl_v<T>;

    // FIELD CONCEPTS
    //////////////////////////////////////////////////////////////

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
