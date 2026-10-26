#include <memory>

#include <gtest/gtest.h>

#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>

namespace samurai
{
    auto init_mesh()
    {
        constexpr std::size_t dim   = 2;
        auto box                    = samurai::Box<double, dim>({0., 0.}, {1., 1.});
        const std::size_t min_level = 2;
        const std::size_t max_level = 10;

        auto mesh_cfg = mesh_config<dim>().min_level(min_level).max_level(max_level);
        return mra::make_mesh(box, mesh_cfg);
    }

    void init_field(auto& u, double factor = 1.0)
    {
        auto& mesh = u.mesh();
        u.resize();

        samurai::for_each_cell(
            mesh,
            [&](auto& cell)
            {
                auto center           = cell.center();
                const double radius   = .2;
                const double x_center = 0.3;
                const double y_center = 0.3;
                if (((center[0] - x_center) * (center[0] - x_center) + (center[1] - y_center) * (center[1] - y_center)) <= radius * radius)
                {
                    u[cell] = factor;
                }
                else
                {
                    u[cell] = 0;
                }
            });

        if constexpr (std::decay_t<decltype(u)>::is_scalar)
        {
            samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);
        }
        else
        {
            samurai::make_bc<samurai::Dirichlet<1>>(u, 0., 0.);
        }
    }

    auto scalar_test(bool relative_detail = false)
    {
        auto mesh  = init_mesh();
        auto field = samurai::make_scalar_field<double>("field", mesh);
        init_field(field);
        auto adapt      = samurai::make_MRAdapt(field);
        auto mra_config = samurai::mra_config().relative_detail(relative_detail);
        adapt(mra_config);
    }

    auto vector_test(bool relative_detail = false)
    {
        auto mesh  = init_mesh();
        auto field = samurai::make_vector_field<double, 2>("field", mesh);
        init_field(field);
        auto adapt      = samurai::make_MRAdapt(field);
        auto mra_config = samurai::mra_config().relative_detail(relative_detail);
        adapt(mra_config);
    }

    auto tuple_test(bool relative_detail = false)
    {
        auto mesh    = init_mesh();
        auto field_1 = samurai::make_scalar_field<double>("field_1", mesh);
        auto field_2 = samurai::make_vector_field<double, 2>("field_2", mesh);
        init_field(field_1);
        init_field(field_2, 2.0);
        auto adapt      = samurai::make_MRAdapt(field_1, field_2);
        auto mra_config = samurai::mra_config().relative_detail(relative_detail);
        adapt(mra_config);
    }

    TEST(MRA, scalar)
    {
        scalar_test(true);
        scalar_test(false);
    }

    TEST(MRA, vector)
    {
        vector_test(true);
        vector_test(false);
    }

    TEST(MRA, tuple)
    {
        tuple_test(true);
        tuple_test(false);
    }

    // Building an adaptation with a user prediction function is a normal call: it must not write to stdout (#570).
    TEST(MRA, custom_prediction_fn_does_not_print)
    {
        auto mesh  = init_mesh();
        auto field = samurai::make_scalar_field<double>("field", mesh);
        init_field(field);
        auto prediction_fn = [](auto& new_field, const auto& old_field)
        {
            return samurai::prediction<1, true>(new_field, old_field);
        };

        testing::internal::CaptureStdout();
        auto adapt      = samurai::make_MRAdapt(prediction_fn, field);
        auto mra_config = samurai::mra_config();
        adapt(mra_config);
        const std::string out = testing::internal::GetCapturedStdout();

        EXPECT_EQ(out, "");
    }

    namespace
    {
        // Prediction function that counts how many times the adaptation calls it.
        struct counting_prediction
        {
            int calls = 0;

            auto operator()(auto& new_field, const auto& old_field)
            {
                ++calls;
                return samurai::prediction<1, true>(new_field, old_field);
            }
        };

        // Same, but move-only: it cannot be copied into the adaptation.
        struct move_only_prediction
        {
            std::unique_ptr<int> calls;

            auto operator()(auto& new_field, const auto& old_field)
            {
                ++*calls;
                return samurai::prediction<1, true>(new_field, old_field);
            }
        };
    }

    // A prediction function passed as an lvalue is held by reference: the
    // caller's object is the one that gets called.
    TEST(MRA, lvalue_prediction_fn_is_held_by_reference)
    {
        auto mesh  = init_mesh();
        auto field = samurai::make_scalar_field<double>("field", mesh);
        init_field(field);

        counting_prediction prediction_fn;
        auto adapt      = samurai::make_MRAdapt(prediction_fn, field);
        auto mra_config = samurai::mra_config();
        adapt(mra_config);

        EXPECT_GT(prediction_fn.calls, 0);
    }

    // A prediction function passed as an rvalue is moved into the adaptation,
    // so a move-only one is accepted.
    TEST(MRA, rvalue_prediction_fn_is_moved_in)
    {
        auto mesh  = init_mesh();
        auto field = samurai::make_scalar_field<double>("field", mesh);
        init_field(field);

        auto calls             = std::make_unique<int>(0);
        const int* calls_value = calls.get();
        auto adapt             = samurai::make_MRAdapt(move_only_prediction{std::move(calls)}, field);
        auto mra_config        = samurai::mra_config();
        adapt(mra_config);

        EXPECT_GT(*calls_value, 0);
    }
}
