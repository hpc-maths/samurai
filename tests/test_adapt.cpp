#include <gtest/gtest.h>

#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>

namespace samurai
{

    template <typename T>
    class adapt_test : public ::testing::Test
    {
    };

    using adapt_test_types = ::testing::
        Types<std::integral_constant<std::size_t, 1>, std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>>;

    TYPED_TEST_SUITE(adapt_test, adapt_test_types, );

    TYPED_TEST(adapt_test, mutliple_fields)
    {
        // ::samurai::initialize();

        static constexpr std::size_t dim = TypeParam::value;
        using config                     = MRConfig<dim>;
        auto mesh                        = MRMesh<config>({xt::zeros<double>({dim}), xt::ones<double>({dim})}, 2, 4);
        auto u_1                         = make_field<double, 1>("u_1", mesh);
        auto u_2                         = make_field<double, 3, true>("u_2", mesh);
        auto u_3                         = make_field<double, 2>("u_3", mesh);

        auto adapt = make_MRAdapt(u_1, u_2, u_3);
        adapt(1e-4, 2);

        // ::samurai::finalize();
    }
}
