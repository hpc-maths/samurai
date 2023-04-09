// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <xtensor/xfixed.hpp>

#include <samurai/operators_base.hpp>
#include <samurai/subset/subset_op.hpp>

template <class TInterval>
class update_boundary_D2Q4_flat_op : public samurai::field_operator_base<TInterval>
{
  public:

    INIT_OPERATOR(update_boundary_D2Q4_flat_op)

    template <class T, class stencil_t>
    inline void operator()(samurai::Dim<1>, T& field, const stencil_t& stencil) const
    {
        field(level, i) = field(level, i - stencil[0]);
    }

    template <class T, class stencil_t>
    inline void operator()(samurai::Dim<2>, T& field, const stencil_t& stencil) const
    {
        field(level, i, j) = field(level, i - stencil[0], j - stencil[1]);
    }
};

template <class T, class stencil_t>
inline auto update_boundary_D2Q4_flat(T&& field, stencil_t&& stencil)
{
    return samurai::make_field_operator_function<update_boundary_D2Q4_flat_op>(std::forward<T>(field), std::forward<stencil_t>(stencil));
}

template <class TInterval>
class update_boundary_D2Q4_linear_op : public samurai::field_operator_base<TInterval>
{
  public:

    INIT_OPERATOR(update_boundary_D2Q4_linear_op)

    template <class T, class stencil_t>
    inline void operator()(samurai::Dim<2>, T& field, const stencil_t& stencil) const
    {
        field(level, i, j) = 2 * field(level, i - stencil[0], j - stencil[1]) - field(level, i - 2 * stencil[0], j - 2 * stencil[1]);
    }
};

template <class T, class stencil_t>
inline auto update_boundary_D2Q4_linear(T&& field, stencil_t&& stencil)
{
    return samurai::make_field_operator_function<update_boundary_D2Q4_linear_op>(std::forward<T>(field), std::forward<stencil_t>(stencil));
}

template <class Field>
inline void update_bc_1D_constant_extension(Field& field, std::size_t level)
{
    const xt::xtensor_fixed<int, xt::xshape<1>> xp{1};
    auto& mesh       = field.mesh();
    using mesh_id_t  = typename decltype(mesh)::mesh_id_t;
    size_t max_level = mesh.max_level();

    const std::size_t j = max_level - level;

    // E first rank (not projected on the level for future use)
    auto east_1 = samurai::intersection(samurai::difference(samurai::translate(mesh.domain(), (1 << j) * xp), mesh.domain()),
                                        mesh[mesh_id_t::reference][level]);

    // E second rank
    auto east_2 = samurai::difference(
        samurai::intersection(samurai::difference(samurai::translate(mesh.domain(), 2 * (1 << j) * xp), mesh.domain()),
                              mesh[mesh_id_t::reference][level]),
        east_1);
    // The order is important becase the second rank shall take the values
    // stored in the first rank
    east_1.on(level).apply_op(update_boundary_D2Q4_flat(field, xp));
    east_2.on(level).apply_op(update_boundary_D2Q4_flat(field,
                                                        xp)); // By not multiplying by 2 it takes the values in the
                                                              // first rank

    // W first rank
    auto west_1 = samurai::intersection(samurai::difference(samurai::translate(mesh.domain(), (-1) * (1 << j) * xp), mesh.domain()),
                                        mesh[mesh_id_t::reference][level]);

    // W second rank
    auto west_2 = samurai::difference(
        samurai::intersection(samurai::difference(samurai::translate(mesh.domain(), 2 * (-1) * (1 << j) * xp), mesh.domain()),
                              mesh[mesh_id_t::reference][level]),
        west_1);
    west_1.on(level).apply_op(update_boundary_D2Q4_flat(field, (-1) * xp));
    west_2.on(level).apply_op(update_boundary_D2Q4_flat(field, (-1) * xp));
}

template <class Field>
inline void update_bc_D2Q4_3_Euler_constant_extension(Field& field, std::size_t level)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};

    const xt::xtensor_fixed<int, xt::xshape<2>> pp{1, 1};
    const xt::xtensor_fixed<int, xt::xshape<2>> pm{1, -1};

    auto& mesh            = field.mesh();
    using mesh_id_t       = typename decltype(mesh)::mesh_id_t;
    std::size_t max_level = mesh.max_level();

    // for (std::size_t level = min_level - 1; level <= max_level; ++level)
    {
        const std::size_t j = max_level - level;

        // E first rank (not projected on the level for future use)
        auto east_1 = samurai::intersection(samurai::difference(samurai::translate(mesh.domain(), (1 << j) * xp), mesh.domain()),
                                            mesh[mesh_id_t::reference][level]);

        // E second rank
        auto east_2 = samurai::difference(
            samurai::intersection(samurai::difference(samurai::translate(mesh.domain(), 2 * (1 << j) * xp), mesh.domain()),
                                  mesh[mesh_id_t::reference][level]),
            east_1);
        // The order is important becase the second rank shall take the values
        // stored in the first rank
        east_1.on(level).apply_op(update_boundary_D2Q4_flat(field, xp));
        east_2.on(level).apply_op(update_boundary_D2Q4_flat(field, xp)); // By not multiplying by 2 it takes the
                                                                         // values in the first rank

        // W first rank
        auto west_1 = samurai::intersection(samurai::difference(samurai::translate(mesh.domain(), (-1) * (1 << j) * xp), mesh.domain()),
                                            mesh[mesh_id_t::reference][level]);

        // W second rank
        auto west_2 = samurai::difference(
            samurai::intersection(samurai::difference(samurai::translate(mesh.domain(), 2 * (-1) * (1 << j) * xp), mesh.domain()),
                                  mesh[mesh_id_t::reference][level]),
            west_1);
        west_1.on(level).apply_op(update_boundary_D2Q4_flat(field, (-1) * xp));
        west_2.on(level).apply_op(update_boundary_D2Q4_flat(field, (-1) * xp));

        // N first rank
        auto north_1 = samurai::intersection(samurai::difference(samurai::translate(mesh.domain(), (1 << j) * yp), mesh.domain()),
                                             mesh[mesh_id_t::reference][level]);

        // N second rank
        auto north_2 = samurai::difference(
            samurai::intersection(samurai::difference(samurai::translate(mesh.domain(), 2 * (1 << j) * yp), mesh.domain()),
                                  mesh[mesh_id_t::reference][level]),
            north_1);
        north_1.on(level).apply_op(update_boundary_D2Q4_flat(field, yp));
        north_2.on(level).apply_op(update_boundary_D2Q4_flat(field, yp));

        // S first rank
        auto south_1 = samurai::intersection(samurai::difference(samurai::translate(mesh.domain(), (-1) * (1 << j) * yp), mesh.domain()),
                                             mesh[mesh_id_t::reference][level]);

        // S second rank
        auto south_2 = samurai::difference(
            samurai::intersection(samurai::difference(samurai::translate(mesh.domain(), 2 * (-1) * (1 << j) * yp), mesh.domain()),
                                  mesh[mesh_id_t::reference][level]),
            south_1);
        south_1.on(level).apply_op(update_boundary_D2Q4_flat(field, (-1) * yp));
        south_2.on(level).apply_op(update_boundary_D2Q4_flat(field, (-1) * yp));

        auto east  = samurai::union_(east_1, east_2);
        auto west  = samurai::union_(west_1, west_2);
        auto north = samurai::union_(north_1, north_2);
        auto south = samurai::union_(south_1, south_2);

        auto north_east = samurai::difference(
            samurai::intersection(samurai::difference(samurai::translate(mesh.domain(), 2 * (1 << j) * pp), mesh.domain()),
                                  mesh[mesh_id_t::reference][level]),
            samurai::union_(east, north));

        north_east.on(level).apply_op(update_boundary_D2Q4_flat(field, 2 * pp)); // Come back inside

        auto south_east = samurai::difference(
            samurai::intersection(samurai::difference(samurai::translate(mesh.domain(), 2 * (1 << j) * pm), mesh.domain()),
                                  mesh[mesh_id_t::reference][level]),
            samurai::union_(east, south));

        south_east.on(level).apply_op(update_boundary_D2Q4_flat(field, 2 * pm)); // Come back inside

        auto north_west = samurai::difference(
            samurai::intersection(samurai::difference(samurai::translate(mesh.domain(), 2 * (-1) * (1 << j) * pm), mesh.domain()),
                                  mesh[mesh_id_t::reference][level]),
            samurai::union_(west, north));

        north_west.on(level).apply_op(update_boundary_D2Q4_flat(field, 2 * (-1) * pm)); // Come back inside

        auto south_west = samurai::difference(
            samurai::intersection(samurai::difference(samurai::translate(mesh.domain(), 2 * (-1) * (1 << j) * pp), mesh.domain()),
                                  mesh[mesh_id_t::reference][level]),
            samurai::union_(south, west));

        south_west.on(level).apply_op(update_boundary_D2Q4_flat(field, 2 * (-1) * pp)); // Come back inside
    }
}

template <class Field>
inline void update_bc_D2Q4_3_Euler_constant_extension_uniform(Field& field)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};

    const xt::xtensor_fixed<int, xt::xshape<2>> pp{1, 1};
    const xt::xtensor_fixed<int, xt::xshape<2>> pm{1, -1};

    auto& mesh      = field.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;

    // E first rank (not projected on the level for future use)
    auto east_1 = samurai::intersection(samurai::difference(samurai::translate(mesh[mesh_id_t::cells], xp), mesh[mesh_id_t::cells]),
                                        mesh[mesh_id_t::reference]);

    // E second rank
    auto east_2 = samurai::difference(
        samurai::intersection(samurai::difference(samurai::translate(mesh[mesh_id_t::cells], 2 * xp), mesh[mesh_id_t::cells]),
                              mesh[mesh_id_t::reference]),
        east_1);

    // The order is important becase the second rank shall take the values
    // stored in the first rank
    east_1.apply_op(update_boundary_D2Q4_flat(field, xp));
    east_2.apply_op(update_boundary_D2Q4_flat(field,
                                              xp)); // By not multiplying by 2 it takes the values in the first rank

    // W first rank
    auto west_1 = samurai::intersection(samurai::difference(samurai::translate(mesh[mesh_id_t::cells], (-1) * xp), mesh[mesh_id_t::cells]),
                                        mesh[mesh_id_t::reference]);

    // W second rank
    auto west_2 = samurai::difference(
        samurai::intersection(samurai::difference(samurai::translate(mesh[mesh_id_t::cells], 2 * (-1) * xp), mesh[mesh_id_t::cells]),
                              mesh[mesh_id_t::reference]),
        west_1);
    west_1.apply_op(update_boundary_D2Q4_flat(field, (-1) * xp));
    west_2.apply_op(update_boundary_D2Q4_flat(field, (-1) * xp));

    // N first rank
    auto north_1 = samurai::intersection(samurai::difference(samurai::translate(mesh[mesh_id_t::cells], yp), mesh[mesh_id_t::cells]),
                                         mesh[mesh_id_t::reference]);

    // N second rank
    auto north_2 = samurai::difference(
        samurai::intersection(samurai::difference(samurai::translate(mesh[mesh_id_t::cells], 2 * yp), mesh[mesh_id_t::cells]),
                              mesh[mesh_id_t::reference]),
        north_1);
    north_1.apply_op(update_boundary_D2Q4_flat(field, yp));
    north_2.apply_op(update_boundary_D2Q4_flat(field, yp));

    // S first rank
    auto south_1 = samurai::intersection(samurai::difference(samurai::translate(mesh[mesh_id_t::cells], (-1) * yp), mesh[mesh_id_t::cells]),
                                         mesh[mesh_id_t::reference]);

    // S second rank
    auto south_2 = samurai::difference(
        samurai::intersection(samurai::difference(samurai::translate(mesh[mesh_id_t::cells], 2 * (-1) * yp), mesh[mesh_id_t::cells]),
                              mesh[mesh_id_t::reference]),
        south_1);
    south_1.apply_op(update_boundary_D2Q4_flat(field, (-1) * yp));
    south_2.apply_op(update_boundary_D2Q4_flat(field, (-1) * yp));

    auto east  = samurai::union_(east_1, east_2);
    auto west  = samurai::union_(west_1, west_2);
    auto north = samurai::union_(north_1, north_2);
    auto south = samurai::union_(south_1, south_2);

    auto north_east = samurai::difference(
        samurai::intersection(samurai::difference(samurai::translate(mesh[mesh_id_t::cells], 2 * pp), mesh[mesh_id_t::cells]),
                              mesh[mesh_id_t::reference]),
        samurai::union_(east, north));

    north_east.apply_op(update_boundary_D2Q4_flat(field, 2 * pp)); // Come back inside

    auto south_east = samurai::difference(
        samurai::intersection(samurai::difference(samurai::translate(mesh[mesh_id_t::cells], 2 * pm), mesh[mesh_id_t::cells]),
                              mesh[mesh_id_t::reference]),
        samurai::union_(east, south));

    south_east.apply_op(update_boundary_D2Q4_flat(field, 2 * pm)); // Come back inside

    auto north_west = samurai::difference(
        samurai::intersection(samurai::difference(samurai::translate(mesh[mesh_id_t::cells], 2 * (-1) * pm), mesh[mesh_id_t::cells]),
                              mesh[mesh_id_t::reference]),
        samurai::union_(west, north));

    north_west.apply_op(update_boundary_D2Q4_flat(field, 2 * (-1) * pm)); // Come back inside

    auto south_west = samurai::difference(
        samurai::intersection(samurai::difference(samurai::translate(mesh[mesh_id_t::cells], 2 * (-1) * pp), mesh[mesh_id_t::cells]),
                              mesh[mesh_id_t::reference]),
        samurai::union_(south, west));

    south_west.apply_op(update_boundary_D2Q4_flat(field, 2 * (-1) * pp)); // Come back inside
}

template <class Field>
inline void update_bc_D2Q4_3_Euler_linear_extension(Field& field, std::size_t level)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};

    const xt::xtensor_fixed<int, xt::xshape<2>> pp{1, 1};
    const xt::xtensor_fixed<int, xt::xshape<2>> pm{1, -1};

    auto& mesh       = field.mesh();
    using mesh_id_t  = typename decltype(mesh)::mesh_id_t;
    size_t max_level = mesh.max_level();

    {
        const std::size_t j = max_level - level;

        // E first rank (not projected on the level for future use)
        auto east_1 = samurai::intersection(samurai::difference(samurai::translate(mesh.domain(), (1 << j) * xp), mesh.domain()),
                                            mesh[mesh_id_t::reference][level]);

        // E second rank
        auto east_2 = samurai::difference(
            samurai::intersection(samurai::difference(samurai::translate(mesh.domain(), 2 * (1 << j) * xp), mesh.domain()),
                                  mesh[mesh_id_t::reference][level]),
            east_1);
        // The order is important becase the second rank shall take the values
        // stored in the first rank
        east_1.on(level).apply_op(update_boundary_D2Q4_linear(field, xp));
        east_2.on(level).apply_op(update_boundary_D2Q4_linear(field, xp)); // By not multiplying by 2 it takes
                                                                           // the values in the first rank

        // W first rank
        auto west_1 = samurai::intersection(samurai::difference(samurai::translate(mesh.domain(), (-1) * (1 << j) * xp), mesh.domain()),
                                            mesh[mesh_id_t::reference][level]);

        // W second rank
        auto west_2 = samurai::difference(
            samurai::intersection(samurai::difference(samurai::translate(mesh.domain(), 2 * (-1) * (1 << j) * xp), mesh.domain()),
                                  mesh[mesh_id_t::reference][level]),
            west_1);
        west_1.on(level).apply_op(update_boundary_D2Q4_linear(field, (-1) * xp));
        west_2.on(level).apply_op(update_boundary_D2Q4_linear(field, (-1) * xp));

        // N first rank
        auto north_1 = samurai::intersection(samurai::difference(samurai::translate(mesh.domain(), (1 << j) * yp), mesh.domain()),
                                             mesh[mesh_id_t::reference][level]);

        // N second rank
        auto north_2 = samurai::difference(
            samurai::intersection(samurai::difference(samurai::translate(mesh.domain(), 2 * (1 << j) * yp), mesh.domain()),
                                  mesh[mesh_id_t::reference][level]),
            north_1);
        north_1.on(level).apply_op(update_boundary_D2Q4_linear(field, yp));
        north_2.on(level).apply_op(update_boundary_D2Q4_linear(field, yp));

        // S first rank
        auto south_1 = samurai::intersection(samurai::difference(samurai::translate(mesh.domain(), (-1) * (1 << j) * yp), mesh.domain()),
                                             mesh[mesh_id_t::reference][level]);

        // S second rank
        auto south_2 = samurai::difference(
            samurai::intersection(samurai::difference(samurai::translate(mesh.domain(), 2 * (-1) * (1 << j) * yp), mesh.domain()),
                                  mesh[mesh_id_t::reference][level]),
            south_1);
        south_1.on(level).apply_op(update_boundary_D2Q4_linear(field, (-1) * yp));
        south_2.on(level).apply_op(update_boundary_D2Q4_linear(field, (-1) * yp));

        auto east  = samurai::union_(east_1, east_2);
        auto west  = samurai::union_(west_1, west_2);
        auto north = samurai::union_(north_1, north_2);
        auto south = samurai::union_(south_1, south_2);

        auto north_east = samurai::difference(
            samurai::intersection(samurai::difference(samurai::translate(mesh.domain(), 2 * (1 << j) * pp), mesh.domain()),
                                  mesh[mesh_id_t::reference][level]),
            samurai::union_(east, north));

        north_east.on(level).apply_op(update_boundary_D2Q4_flat(field, 2 * pp)); // Come back inside

        auto south_east = samurai::difference(
            samurai::intersection(samurai::difference(samurai::translate(mesh.domain(), 2 * (1 << j) * pm), mesh.domain()),
                                  mesh[mesh_id_t::reference][level]),
            samurai::union_(east, south));

        south_east.on(level).apply_op(update_boundary_D2Q4_flat(field, 2 * pm)); // Come back inside

        auto north_west = samurai::difference(
            samurai::intersection(samurai::difference(samurai::translate(mesh.domain(), 2 * (-1) * (1 << j) * pm), mesh.domain()),
                                  mesh[mesh_id_t::reference][level]),
            samurai::union_(west, north));

        north_west.on(level).apply_op(update_boundary_D2Q4_flat(field, 2 * (-1) * pm)); // Come back inside

        auto south_west = samurai::difference(
            samurai::intersection(samurai::difference(samurai::translate(mesh.domain(), 2 * (-1) * (1 << j) * pp), mesh.domain()),
                                  mesh[mesh_id_t::reference][level]),
            samurai::union_(south, west));

        south_west.on(level).apply_op(update_boundary_D2Q4_flat(field, 2 * (-1) * pp)); // Come back inside
    }
}
