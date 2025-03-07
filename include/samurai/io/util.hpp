// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

namespace samurai
{
    template <class Field, class SubMesh>
    auto extract_data(const Field& field, const SubMesh& submesh)
    {
        using size_type                       = typename Field::inner_types::size_type;
        std::array<std::size_t, 2> data_shape = {submesh.nb_cells(), static_cast<std::size_t>(field.size)};
        xt::xtensor<typename Field::value_type, 2> data(data_shape);

        if (submesh.nb_cells() != 0)
        {
            std::size_t index = 0;
            for_each_cell(submesh,
                          [&](auto cell)
                          {
                              if constexpr (Field::size == 1)
                              {
                                  data(index, 0) = field[cell];
                              }
                              else
                              {
                                  for (size_type i = 0; i < field.size; ++i)
                                  {
                                      data(index, i) = field[cell][i];
                                  }
                              }
                              index++;
                          });
        }
        return data;
    }
}
