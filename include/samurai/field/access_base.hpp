
// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "concepts.hpp"
#include "debug_utils.hpp"

namespace samurai
{
    namespace detail
    {
        template <class Field>
        struct inner_field_types;

        template <class Field>
        struct field_data_access;

        template <class D>
            requires valid_field_mesh_and_value<D>
        class field_data_access_base
        {
          public:

            using derived_type                  = D;
            using inner_types                   = inner_field_types<derived_type>;
            static constexpr std::size_t dim    = inner_types::dim;
            using data_type                     = typename inner_types::data_type;
            using local_data_type               = typename inner_types::local_data_type;
            using value_type                    = typename inner_types::value_type;
            using interval_t                    = typename inner_types::interval_t;
            using interval_value_t              = typename inner_types::interval_value_t;
            using index_t                       = typename inner_types::index_t;
            using size_type                     = typename inner_types::size_type;
            using cell_t                        = typename inner_types::cell_t;
            static constexpr auto static_layout = data_type::static_layout;

            derived_type& derived_cast() & noexcept
            {
                return *static_cast<derived_type*>(this);
            }

            const derived_type& derived_cast() const& noexcept
            {
                return *static_cast<const derived_type*>(this);
            }

            derived_type derived_cast() && noexcept
            {
                return std::move(*static_cast<derived_type*>(this));
            }

            auto& storage() noexcept
            {
                return m_storage;
            }

            const auto& storage() const noexcept
            {
                return m_storage;
            }

            void fill(value_type v)
            {
                m_storage.data().fill(v);
                this->derived_cast().ghosts_updated() = false;
            }

            template <class... T>
            inline auto operator()(const std::size_t level, const interval_t& interval, const T... index)
            {
                auto interval_tmp = this->derived_cast().get_interval(level, interval, index...);
                return view(m_storage, {interval_tmp.index + interval.start, interval_tmp.index + interval.end, interval.step});
            }

            template <class... T>
            inline auto operator()(const std::size_t level, const interval_t& interval, const T... index) const
            {
                auto interval_tmp = this->derived_cast().get_interval(level, interval, index...);
                auto data = view(m_storage, {interval_tmp.index + interval.start, interval_tmp.index + interval.end, interval.step});
                check_nan(data, "field access");
                return data;
            }

            inline auto operator()(const std::size_t level,
                                   const interval_t& interval,
                                   const xt::xtensor_fixed<interval_value_t, xt::xshape<dim - 1>>& index)
            {
                auto interval_tmp = this->derived_cast().get_interval(level, interval, index);
                return view(m_storage, {interval_tmp.index + interval.start, interval_tmp.index + interval.end, interval.step});
            }

            template <class... T>
            inline auto operator()(const std::size_t level,
                                   const interval_t& interval,
                                   const xt::xtensor_fixed<interval_value_t, xt::xshape<dim - 1>>& index) const
            {
                auto interval_tmp = this->derived_cast().get_interval(level, interval, index);
                auto data = view(m_storage, {interval_tmp.index + interval.start, interval_tmp.index + interval.end, interval.step});

                check_nan(data, "field access (xtensor_fixed index)");
                return data;
            }

            void resize()
            {
                m_storage.resize(static_cast<size_type>(this->derived_cast().mesh().nb_cells()));
                this->derived_cast().ghosts_updated() = false;
#ifdef SAMURAI_CHECK_NAN
                if constexpr (std::is_floating_point_v<value_type>)
                {
                    this->derived_cast().storage().data().fill(std::nan(""));
                }
#endif
            }

          private:

            data_type m_storage;
        };
    }
}
