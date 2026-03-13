// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../cell_flag.hpp"
#include "../operators_base.hpp"
#include "../utils.hpp"

namespace samurai
{
    template <std::size_t dim, class TInterval>
    class mr_criteria_op : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(mr_criteria_op)

        template <class T1, class T2>
        SAMURAI_INLINE void operator()(Dim<dim>, const T1& detail, T2& tag, double eps, double regularity) const
        {
            std::size_t fine_level = level + 1;

            auto& mesh     = tag.mesh();
            auto min_level = mesh.min_level();
            auto max_level = mesh.max_level();

            const auto* data = detail.data();
            auto* tag_data   = tag.data();

            auto fine_eps      = pow(2.0, regularity) * eps;
            auto coarse_eps    = fine_eps / (1 << dim);
            auto coarse_offset = memory_offset(mesh, {level, i.start, index});

            std::array<std::size_t, 1ULL << dim> fine_offsets;
            std::size_t ind = 0;
            static_nested_loop<dim - 1, 0, 2>(
                [&](const auto& stencil)
                {
                    auto new_index        = 2 * index + stencil;
                    fine_offsets[ind]     = memory_offset(detail.mesh(), {fine_level, 2 * i.start, new_index});
                    fine_offsets[ind + 1] = fine_offsets[ind] + 1;
                    ind += 2;
                });

            for (std::size_t ii = 0; ii < i.size(); ++ii)
            {
                if (fine_level > min_level)
                {
                    bool cond = false;
                    for (std::size_t n = 0; n < T1::n_comp; ++n)
                    {
                        if (std::abs(data[(coarse_offset + ii) * T1::n_comp + n]) > coarse_eps)
                        {
                            cond = true;
                            break;
                        }
                    }

                    if (!cond)
                    {
                        auto coarsen_check = std::apply(
                            [&](auto... offsets)
                            {
                                auto check_comp = [&](auto offset)
                                {
                                    for (std::size_t n = 0; n < T1::n_comp; ++n)
                                    {
                                        if (std::abs(data[(offset + 2 * ii) * T1::n_comp + n]) > eps)
                                        {
                                            return false;
                                        }
                                    }
                                    return true;
                                };
                                return (check_comp(offsets) && ...);
                            },
                            fine_offsets);

                        if (coarsen_check)
                        {
                            std::apply(
                                [&](auto... offsets)
                                {
                                    ((tag_data[offsets + 2 * ii] = static_cast<std::uint8_t>(CellFlag::coarsen)), ...);
                                },
                                fine_offsets);
                        }
                    }
                }

                if (fine_level < max_level)
                {
                    std::apply(
                        [&](auto... offsets)
                        {
                            auto tag2refine = [&](auto offset)
                            {
                                for (std::size_t n = 0; n < T1::n_comp; ++n)
                                {
                                    if (std::abs(data[(offset + 2 * ii) * T1::n_comp + n]) > fine_eps)
                                    {
                                        return true;
                                    }
                                }
                                return false;
                            };
                            ((tag_data[offsets + 2 * ii] |= (tag2refine(offsets)) * static_cast<std::uint8_t>(CellFlag::refine)), ...);
                        },
                        fine_offsets);
                }
            }
        }
    };

    template <class... CT>
    SAMURAI_INLINE auto mr_criteria(CT&&... e)
    {
        return make_field_operator_function<mr_criteria_op>(std::forward<CT>(e)...);
    }

} // namespace samurai
