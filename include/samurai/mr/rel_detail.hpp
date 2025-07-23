// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <array>
#include <limits>

namespace samurai
{
    namespace detail
    {
        void set_inv_max_field(auto& inv_max_fields, const auto& field, std::size_t dec = 0)
            requires(std::decay_t<decltype(field)>::is_scalar)
        {
            auto& mesh = field.mesh();

            for_each_cell(mesh,
                          [&](const auto& cell)
                          {
                              inv_max_fields[dec] = std::max(inv_max_fields[dec], std::abs(field[cell]));
                          });
        }

        void set_inv_max_field(auto& inv_max_fields, const auto& field, std::size_t dec = 0)
            requires(!std::decay_t<decltype(field)>::is_scalar)
        {
            auto& mesh = field.mesh();

            for_each_cell(mesh,
                          [&](const auto& cell)
                          {
                              for (std::size_t i = 0; i < field.n_comp; ++i)
                              {
                                  inv_max_fields[dec + i] = std::max(inv_max_fields[dec + i], std::abs(field[cell][i]));
                              }
                          });
        }

        template <class... TFields>
        void set_inv_max_field(auto& inv_max_fields, const Field_tuple<TFields...>& field)
        {
            auto f = [](auto& inv_max_fields, const auto& field, auto& dec)
            {
                set_inv_max_field(inv_max_fields, field, dec);
                dec += field.n_comp;
            };

            std::size_t dec = 0;
            std::apply(
                [&](const auto&... args)
                {
                    (f(inv_max_fields, args, dec), ...);
                },
                field.elements());
        }
    }

    auto compute_relative_detail(auto& detail, const auto& fields)
    {
        using detail_t   = typename std::decay_t<decltype(detail)>;
        using value_t    = typename detail_t::value_type;
        using local_type = std::array<value_t, detail_t::n_comp>;

        local_type inv_max_fields;
        inv_max_fields.fill(std::numeric_limits<value_t>::min());

        detail::set_inv_max_field(inv_max_fields, fields);

#ifdef SAMURAI_WITH_MPI
        mpi::communicator world;
        mpi::all_reduce(world, mpi::inplace(inv_max_fields.data()), inv_max_fields.size(), mpi::maximum<value_t>());
#endif

        for (std::size_t i = 0; i < inv_max_fields.size(); ++i)
        {
            if (inv_max_fields[i] < std::numeric_limits<value_t>::epsilon())
            {
                inv_max_fields[i] = 1.0;
            }
            inv_max_fields[i] = 1. / inv_max_fields[i];
        }

        auto inv_max_fields_xt = xt::adapt(inv_max_fields);
        detail.array() *= inv_max_fields_xt;
    }
}