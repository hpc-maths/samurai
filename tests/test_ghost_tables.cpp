// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// The tables of update_ghost_mr against the set expressions they replace: on an
// adapted mesh, the field projection and the prediction of the ghosts of every
// level must give the same values, bit for bit, for a scalar and a vector field
// updated together.

#include <cmath>
#include <cstddef>
#include <cstring>

#include <gtest/gtest.h>

#include <samurai/algorithm/update.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>

namespace samurai
{
    namespace
    {
        // Every stored value, ghosts included, set to a value that differs from
        // one storage slot to the next.
        template <class Field>
        void fill_all(Field& f, double seed)
        {
            auto* data       = f.data();
            const auto count = f.array().size();
            for (std::size_t k = 0; k < count; ++k)
            {
                data[k] = std::sin(seed + 0.37 * static_cast<double>(k));
            }
        }

        template <class Field>
        bool same_bits(const Field& a, const Field& b)
        {
            return std::memcmp(a.data(), b.data(), a.array().size() * sizeof(*a.data())) == 0;
        }

        template <std::size_t dim>
        void check_tables()
        {
            const std::size_t max_level = dim == 3 ? 5 : 7;
            auto config = mesh_config<dim>().min_level(2).max_level(max_level).max_stencil_size(2).disable_minimal_ghost_width();
            auto mesh   = mra::make_mesh(Box<double, dim>(xt::zeros<double>({dim}), xt::ones<double>({dim})), config);

            // a blob that refines part of the domain, so that every level has ghosts
            auto u = make_scalar_field<double>("u", mesh);
            for_each_cell(mesh,
                          [&](const auto& cell)
                          {
                              double r2 = 0;
                              for (std::size_t d = 0; d < dim; ++d)
                              {
                                  r2 += (cell.center(d) - 0.4) * (cell.center(d) - 0.4);
                              }
                              u[cell] = std::tanh(50 * (std::sqrt(r2) - 0.2));
                          });
            make_bc<Dirichlet<1>>(u, 0.);
            make_MRAdapt(u)(mra_config().epsilon(1e-3));

            // the comparison below must have ghosts to compare
            const auto& tables = u.mesh().ghost_tables();
            ASSERT_FALSE(tables.empty());
            std::size_t n_projected = 0;
            std::size_t n_predicted = 0;
            for (const auto& t : tables.projection)
            {
                n_projected += t.dest.size();
            }
            for (const auto& t : tables.prediction)
            {
                n_predicted += t.dest.size();
            }
            ASSERT_GT(n_projected, 0u);
            ASSERT_GT(n_predicted, 0u);

            auto s = make_scalar_field<double>("s", u.mesh());
            auto v = make_vector_field<double, 3>("v", u.mesh());
            fill_all(s, 0.1);
            fill_all(v, 0.7);

            for (std::size_t level = 1; level <= u.mesh().max_level(); ++level)
            {
                auto s_sets  = s;
                auto v_sets  = v;
                auto s_table = s;
                auto v_table = v;
                detail::project_ghosts_with_sets(level, s_sets, v_sets);
                detail::project_ghosts(level, s_table, v_table);
                EXPECT_TRUE(same_bits(s_sets, s_table) && same_bits(v_sets, v_table)) << "field projection to level " << level - 1;

                s_sets  = s;
                v_sets  = v;
                s_table = s;
                v_table = v;
                detail::predict_ghosts_with_sets(level, s_sets, v_sets);
                detail::predict_ghosts(level, s_table, v_table);
                EXPECT_TRUE(same_bits(s_sets, s_table) && same_bits(v_sets, v_table)) << "prediction at level " << level;
            }
        }
    }

    TEST(ghost_tables, same_values_as_sets_1d)
    {
        check_tables<1>();
    }

    TEST(ghost_tables, same_values_as_sets_2d)
    {
        check_tables<2>();
    }

    TEST(ghost_tables, same_values_as_sets_3d)
    {
        check_tables<3>();
    }
}
