// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <array>
#include <limits>

#include <xtensor/containers/xadapt.hpp>

namespace samurai
{
    namespace detail
    {
        // Per-component max of |field| over the leaf cells. Iterated per interval
        // over the raw field data (the interval's index is the flat cell offset,
        // as used by field[cell]) rather than per cell with for_each_cell, which
        // builds a Cell object for every cell - the dominant cost of the previous
        // version. max is order-independent, so the result is bit-identical.
        void set_inv_max_field(auto& inv_max_fields, const auto& field, std::size_t dec = 0)
            requires(std::decay_t<decltype(field)>::is_scalar)
        {
            const auto* data = field.data();
            for_each_interval(field.mesh(),
                              [&](std::size_t, const auto& i, const auto&)
                              {
                                  for (auto x = i.start; x < i.end; ++x)
                                  {
                                      const auto flat     = static_cast<std::size_t>(i.index + x);
                                      inv_max_fields[dec] = std::max(inv_max_fields[dec], std::abs(data[flat]));
                                  }
                              });
        }

        void set_inv_max_field(auto& inv_max_fields, const auto& field, std::size_t dec = 0)
            requires(!std::decay_t<decltype(field)>::is_scalar)
        {
            constexpr std::size_t nc = std::decay_t<decltype(field)>::n_comp;
            const auto* data         = field.data();
            for_each_interval(field.mesh(),
                              [&](std::size_t, const auto& i, const auto&)
                              {
                                  for (auto x = i.start; x < i.end; ++x)
                                  {
                                      const auto flat = static_cast<std::size_t>(i.index + x) * nc;
                                      for (std::size_t c = 0; c < nc; ++c)
                                      {
                                          inv_max_fields[dec + c] = std::max(inv_max_fields[dec + c], std::abs(data[flat + c]));
                                      }
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

        // Raw, cache-streaming normalization instead of xtensor's broadcast
        // `detail.array() *= inv_max_fields_xt`: the latter does not vectorize and
        // dominated the whole detail computation (~5 s out of 7 s on the euler_2d
        // demo). Same result, bit-identical.
        constexpr std::size_t nc = detail_t::n_comp;
        auto* dd                 = detail.data();
        const std::size_t n      = detail.array().size();
        for (std::size_t k = 0; k < n; k += nc)
        {
            for (std::size_t c = 0; c < nc; ++c)
            {
                dd[k + c] *= inv_max_fields[c];
            }
        }
    }
}
