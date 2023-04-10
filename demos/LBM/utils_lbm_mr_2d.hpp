// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

// I do many separate functions because the return type
// is not necessarely the same between directions and I want to avoid
// using a template, which indeed comes back to the same than this.
template <class Mesh>
auto get_adjacent_boundary_east(Mesh& mesh, std::size_t level, typename Mesh::mesh_id_t type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};

    std::size_t coeff = 1 << (mesh.max_level() - level); // When we are not at the finest
                                                         // level, we must translate more

    return samurai::intersection(
        samurai::difference(samurai::difference(samurai::difference(mesh.domain(), samurai::translate(mesh.domain(), -coeff * xp)),
                                                samurai::difference(mesh.domain(),
                                                                    samurai::translate(mesh.domain(),
                                                                                       -coeff * yp))),          // Removing NE
                            samurai::difference(mesh.domain(), samurai::translate(mesh.domain(), coeff * yp))), // Removing SE
        mesh[type][level]);                                                                                     //.on(level);
}

template <class Mesh>
auto get_adjacent_boundary_north(Mesh& mesh, std::size_t level, typename Mesh::mesh_id_t type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};

    std::size_t coeff = 1 << (mesh.max_level() - level);

    return samurai::intersection(
        samurai::difference(samurai::difference(samurai::difference(mesh.domain(), samurai::translate(mesh.domain(), -coeff * yp)),
                                                samurai::difference(mesh.domain(),
                                                                    samurai::translate(mesh.domain(),
                                                                                       -coeff * xp))),          // Removing NE
                            samurai::difference(mesh.domain(), samurai::translate(mesh.domain(), coeff * xp))), // Removing NW
        mesh[type][level]);                                                                                     //.on(level);
}

template <class Mesh>
auto get_adjacent_boundary_west(Mesh& mesh, std::size_t level, typename Mesh::mesh_id_t type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};

    std::size_t coeff = 1 << (mesh.max_level() - level);

    return samurai::intersection(
        samurai::difference(samurai::difference(samurai::difference(mesh.domain(), samurai::translate(mesh.domain(), coeff * xp)),
                                                samurai::difference(mesh.domain(),
                                                                    samurai::translate(mesh.domain(),
                                                                                       -coeff * yp))),          // Removing NW
                            samurai::difference(mesh.domain(), samurai::translate(mesh.domain(), coeff * yp))), // Removing SW
        mesh[type][level]);                                                                                     //.on(level);
}

template <class Mesh>
auto get_adjacent_boundary_south(Mesh& mesh, std::size_t level, typename Mesh::mesh_id_t type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};

    std::size_t coeff = 1 << (mesh.max_level() - level);

    return samurai::intersection(
        samurai::difference(samurai::difference(samurai::difference(mesh.domain(), samurai::translate(mesh.domain(), coeff * yp)),
                                                samurai::difference(mesh.domain(),
                                                                    samurai::translate(mesh.domain(),
                                                                                       -coeff * xp))),          // Removing SE
                            samurai::difference(mesh.domain(), samurai::translate(mesh.domain(), coeff * xp))), // Removing SW
        mesh[type][level]);                                                                                     //.on(level);
}

template <class Mesh>
auto get_adjacent_boundary_northeast(Mesh& mesh, std::size_t level, typename Mesh::mesh_id_t type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};
    const xt::xtensor_fixed<int, xt::xshape<2>> d11{1, 1};

    std::size_t coeff = 1 << (mesh.max_level() - level);

    return samurai::intersection(
        samurai::difference(samurai::difference(samurai::difference(mesh.domain(), samurai::translate(mesh.domain(), -coeff * d11)),
                                                samurai::translate(mesh.domain(),
                                                                   -coeff * yp)), // Removing vertical strip
                            samurai::translate(mesh.domain(),
                                               -coeff * xp)), // Removing horizontal strip
        mesh[type][level]);                                   //.on(level);
}

template <class Mesh>
auto get_adjacent_boundary_northwest(Mesh& mesh, std::size_t level, typename Mesh::mesh_id_t type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};
    const xt::xtensor_fixed<int, xt::xshape<2>> d1m1{1, -1};

    std::size_t coeff = 1 << (mesh.max_level() - level);

    return samurai::intersection(
        samurai::difference(samurai::difference(samurai::difference(mesh.domain(), samurai::translate(mesh.domain(), coeff * d1m1)),
                                                samurai::translate(mesh.domain(),
                                                                   -coeff * yp)), // Removing vertical strip
                            samurai::translate(mesh.domain(),
                                               coeff * xp)), // Removing horizontal strip
        mesh[type][level]);                                  //.on(level);
}

template <class Mesh>
auto get_adjacent_boundary_southwest(Mesh& mesh, std::size_t level, typename Mesh::mesh_id_t type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};
    const xt::xtensor_fixed<int, xt::xshape<2>> d11{1, 1};

    std::size_t coeff = 1 << (mesh.max_level() - level);

    return samurai::intersection(
        samurai::difference(samurai::difference(samurai::difference(mesh.domain(), samurai::translate(mesh.domain(), coeff * d11)),
                                                samurai::translate(mesh.domain(),
                                                                   coeff * yp)), // Removing vertical strip
                            samurai::translate(mesh.domain(),
                                               coeff * xp)), // Removing horizontal strip
        mesh[type][level]);                                  //.on(level);
}

template <class Mesh>
auto get_adjacent_boundary_southeast(Mesh& mesh, std::size_t level, typename Mesh::mesh_id_t type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};
    const xt::xtensor_fixed<int, xt::xshape<2>> d1m1{1, -1};

    std::size_t coeff = 1 << (mesh.max_level() - level);

    return samurai::intersection(
        samurai::difference(samurai::difference(samurai::difference(mesh.domain(), samurai::translate(mesh.domain(), -coeff * d1m1)),
                                                samurai::translate(mesh.domain(),
                                                                   coeff * yp)), // Removing vertical strip
                            samurai::translate(mesh.domain(),
                                               -coeff * xp)), // Removing horizontal strip
        mesh[type][level]);                                   //.on(level);
}
