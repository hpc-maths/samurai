// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <xtensor/xmasked_view.hpp>
#include <xtensor/xtensor.hpp>

#include "../cell_flag.hpp"
#include "../subset/subset_op.hpp"
#include "utils.hpp"

namespace samurai
{
    ///////////////////////
    // graduate operator //
    ///////////////////////

    template <std::size_t dim, class TInterval>
    class graduate_op : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(graduate_op)

        template <std::size_t d, class T, class Stencil>
        inline void operator()(Dim<d>, T& tag, const Stencil& s) const
        {
            using namespace xt::placeholders;

            auto tag_func = [&](auto& i_f)
            {
                auto mask = tag(level, i_f - s[0], index - view(s, xt::range(1, _))) & static_cast<int>(CellFlag::refine);
                auto i_c  = i_f >> 1;
                apply_on_masked(tag(level - 1, i_c, index >> 1),
                                mask,
                                [](auto& e)
                                {
                                    e |= static_cast<int>(CellFlag::refine);
                                });

                auto mask2 = tag(level, i_f - s[0], index - view(s, xt::range(1, _))) & static_cast<int>(CellFlag::keep);
                apply_on_masked(tag(level - 1, i_c, index >> 1),
                                mask2,
                                [](auto& e)
                                {
                                    e |= static_cast<int>(CellFlag::keep);
                                });
            };

            if (auto i_even = i.even_elements(); i_even.is_valid())
            {
                tag_func(i_even);
            }

            if (auto i_odd = i.odd_elements(); i_odd.is_valid())
            {
                tag_func(i_odd);
            }
        }
    };

    template <class T, class Stencil>
    inline auto graduate(T& tag, const Stencil& s)
    {
        return make_field_operator_function<graduate_op>(tag, s);
    }

    template <class Tag, class Stencil>
    void graduation(Tag& tag, const Stencil& stencil)
    {
        auto& mesh      = tag.mesh();
        using mesh_t    = typename Tag::mesh_t;
        using mesh_id_t = typename mesh_t::mesh_id_t;

        std::size_t max_level = mesh.max_level();

        constexpr int ghost_width = mesh_t::config::graduation_width; // cppcheck-suppress unreadVariable

        for (std::size_t level = max_level; level > 0; --level)
        {
            /**
             *
             *        |-----|-----| |-----|-----|
             *                                    --------------->
             *                                                             K
             *        |===========|-----------| |===========|-----------|
             */

            auto ghost_subset = intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::reference][level - 1]).on(level - 1);

            ghost_subset.apply_op(tag_to_keep<0>(tag));

            /**
             *                 R                                 K     R     K
             *        |-----|-----|=====|   ---------------> |-----|-----|=====|
             *
             */

            auto subset_2 = intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]);

            subset_2.apply_op(tag_to_keep<ghost_width>(tag, CellFlag::refine));

            /**
             *      K     C                          K     K
             *   |-----|-----|   -------------->  |-----|-----|
             *
             *   |-----------|
             *
             */

            auto keep_subset = intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]).on(level - 1);
            keep_subset.apply_op(keep_children_together(tag));

            /**
             * Case 1
             * ======
             *                   R     K R     K
             *                |-----|-----|   --------------> |-----|-----| C or
             * K                                                 R
             *   |-----------| |-----------|
             *
             * Case 2
             * ======
             *                   K     K K     K
             *                |-----|-----|   --------------> |-----|-----| C K
             *   |-----------| |-----------|
             *
             */
            assert(stencil.shape()[1] == Tag::dim);
            for (std::size_t i = 0; i < stencil.shape()[0]; ++i)
            {
                auto s      = xt::view(stencil, i);
                auto subset = intersection(translate(mesh[mesh_id_t::cells][level], s), mesh[mesh_id_t::cells][level - 1]).on(level);
                subset.apply_op(graduate(tag, s));
            }
        }
    }
}
