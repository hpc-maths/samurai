#include <algorithm>

#include <gtest/gtest.h>

#include <samurai/field.hpp>
#include <samurai/stencil_field.hpp>
#include <samurai/uniform_mesh.hpp>
#include <samurai/utils.hpp>

namespace samurai
{
    TEST(utils, is_field_function)
    {
        Box<double, 1> box{{0}, {1}};
        using Config = UniformConfig<1>;
        auto mesh    = UniformMesh<Config>(box, 3);

        auto u = make_scalar_field<double>("u", mesh);

        static_assert(detail::is_field_function<decltype(5 + u)>::value);
        static_assert(detail::is_field_function<decltype(upwind(1, u))>::value);
    }

    TEST(utils, compute_n_comp)
    {
        Box<double, 1> box{{0}, {1}};
        using Config = UniformConfig<1>;
        auto mesh    = UniformMesh<Config>(box, 3);

        auto u1 = make_scalar_field<int>("u", mesh);
        auto u2 = make_vector_field<double, 3>("u", mesh);

        static_assert(detail::compute_n_comp<decltype(u1), decltype(u2)>() == 4);

        static_assert(std::is_same<detail::common_type_t<decltype(u1), decltype(u2)>, double>::value);
    }
}
