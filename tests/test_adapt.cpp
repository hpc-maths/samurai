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

#if __GNUC__ == 11
    // The 3d case is broken for gcc-11.4 with gtest for release build type and only in gtest. It works outside...
    using adapt_test_types = ::testing::Types<std::integral_constant<std::size_t, 1>, std::integral_constant<std::size_t, 2>>; //,
                                                                                                                               // std::integral_constant<std::size_t,
                                                                                                                               // 3>>;
#else
    using adapt_test_types = ::testing::
        Types<std::integral_constant<std::size_t, 1>, std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>>;
#endif

    TYPED_TEST_SUITE(adapt_test, adapt_test_types, );

    TYPED_TEST(adapt_test, mutliple_fields)
    {
        ::samurai::initialize();

        static constexpr std::size_t dim = TypeParam::value;
        using config                     = MRConfig<dim>;
        using box_t                      = Box<double, dim>;
        auto mesh_cfg                    = mesh_config<dim>().min_level(2).max_level(4);
        auto mesh                        = MRMesh<config>(mesh_cfg, box_t{xt::zeros<double>({dim}), xt::ones<double>({dim})});
        auto u_1                         = make_scalar_field<double>("u_1", mesh);
        auto u_2                         = make_vector_field<double, 3, true>("u_2", mesh);
        auto u_3                         = make_vector_field<double, 2>("u_3", mesh);

        auto adapt      = make_MRAdapt(u_1, u_2, u_3);
        auto mra_config = samurai::mra_config().regularity(2.);
        adapt(mra_config);
        ::samurai::finalize();
    }
}
