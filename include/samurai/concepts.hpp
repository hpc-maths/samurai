// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <type_traits>

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

    template <class Mesh>
    class hold;

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

    template <class Mesh>
    struct is_mesh_impl<hold<Mesh>> : std::bool_constant<is_mesh_impl_v<Mesh>>
    {
    };

    template <class T>
    constexpr bool mesh_like_helper = is_mesh_impl_v<std::remove_cvref_t<T>>;

    template <class T>
    concept mesh_like = mesh_like_helper<T>;
}
