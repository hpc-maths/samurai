// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

namespace samurai
{
    // ------------------------------------------------------------------------
    // class Field_tuple
    // ------------------------------------------------------------------------

    template <class TField, class... TFields>
    class Field_tuple
    {
      public:

        using tuple_type             = std::tuple<TField&, TFields&...>;
        using tuple_type_without_ref = std::tuple<TField, TFields...>;
        using common_t               = detail::common_type_t<TField, TFields...>;
        using mesh_t                 = typename TField::mesh_t;
        using mesh_id_t              = typename mesh_t::mesh_id_t;
        using size_type              = typename TField::size_type;

        Field_tuple(TField& field, TFields&... fields)
            : m_fields(field, fields...)
        {
        }

        const auto& mesh() const
        {
            return std::get<0>(m_fields).mesh();
        }

        auto& mesh()
        {
            return std::get<0>(m_fields).mesh();
        }

        const auto& elements() const
        {
            return m_fields;
        }

        auto& elements()
        {
            return m_fields;
        }

      private:

        tuple_type m_fields;
    };
}
