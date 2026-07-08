// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <tuple>

#include "tuple_field.hpp"

namespace samurai
{

    template <class TField, class... TFields>
    auto& get_elements(Field_tuple<TField, TFields...>& ft)
    {
        return ft.elements();
    }

    template <class TField>
    auto get_elements(TField& f)
    {
        return std::tie(f);
    }
}
