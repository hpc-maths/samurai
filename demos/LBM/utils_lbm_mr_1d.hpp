// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

// Attention : the number 2 as second template parameter does not mean
// that we are dealing with two fields!!!!
template <class Field, class interval_t>
xt::xtensor<double, 2> prediction_all(const Field& f,
                                      std::size_t level_g,
                                      std::size_t level,
                                      const interval_t& k,
                                      std::map<std::tuple<std::size_t, std::size_t, interval_t>, xt::xtensor<double, 2>>& mem_map)
{
    constexpr std::size_t nvel = Field::size; // Number of velocities
    // That is used to employ _ with xtensor
    using namespace xt::placeholders;

    auto it = mem_map.find({level_g, level, k});

    if (it != mem_map.end() && k.size() == (std::get<2>(it->first)).size())
    {
        return it->second;
    }
    else
    {
        auto& mesh      = f.mesh();
        using mesh_id_t = typename decltype(mesh)::mesh_id_t;

        std::vector<std::size_t> shape_x = {k.size(), nvel};
        xt::xtensor<double, 2> out       = xt::empty<double>(shape_x);

        auto mask = mesh.exists(mesh_id_t::cells_and_ghosts,
                                level_g + level,
                                k); // Check if we are on a leaf or a ghost (CHECK IF IT IS OK)

        xt::xtensor<double, 2> mask_all = xt::empty<double>(shape_x);

        for (int h_field = 0; h_field < nvel; ++h_field)
        {
            xt::view(mask_all, xt::all(), h_field) = mask;
        }

        // Recursion finished
        if (xt::all(mask))
        {
            return xt::eval(f(0, nvel, level_g + level, k));
        }

        // If we cannot stop here
        auto kg = k >> 1;
        kg.step = 1;

        xt::xtensor<double, 2> val = xt::empty<double>(shape_x);

        auto earth = xt::eval(prediction_all(f, level_g, level - 1, kg, mem_map));
        auto W     = xt::eval(prediction_all(f, level_g, level - 1, kg - 1, mem_map));
        auto E     = xt::eval(prediction_all(f, level_g, level - 1, kg + 1, mem_map));

        // This is to deal with odd/even indices in the x direction
        std::size_t start_even = (k.start & 1) ? 1 : 0;
        std::size_t start_odd  = (k.start & 1) ? 0 : 1;
        std::size_t end_even   = (k.end & 1) ? kg.size() : kg.size() - 1;
        std::size_t end_odd    = (k.end & 1) ? kg.size() - 1 : kg.size();

        xt::view(val, xt::range(start_even, _, 2)) = xt::view(earth + 1. / 8 * (W - E), xt::range(start_even, _));
        xt::view(val, xt::range(start_odd, _, 2))  = xt::view(earth - 1. / 8 * (W - E), xt::range(_, end_odd));

        xt::masked_view(out, !mask_all) = xt::masked_view(val, !mask_all);

        for (int k_mask = 0, k_int = k.start; k_int < k.end; ++k_mask, ++k_int)
        {
            if (mask[k_mask])
            {
                xt::view(out, k_mask) = xt::view(f(0, nvel, level_g + level, {k_int, k_int + 1}), 0);
            }
        }

        // It is crucial to use insert and not []
        // in order not to update the value in case of duplicated (same key)
        mem_map.insert(std::make_pair(std::tuple<std::size_t, std::size_t, interval_t>{level_g, level, k}, out));
        return out;
    }
}
