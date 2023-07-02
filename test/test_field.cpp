#include <algorithm>

#include <gtest/gtest.h>

#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/uniform_mesh.hpp>

namespace samurai
{

    TEST(field, copy_from_const)
    {
        samurai::Box<double, 1> box{{0}, {1}};
        using Config       = samurai::UniformConfig<1>;
        auto mesh          = samurai::UniformMesh<Config>(box, 3);
        const auto u_const = samurai::make_field<double, 1>("uc", mesh);

        auto u = u_const;

        auto m              = samurai::holder(mesh);
        const auto u_const1 = samurai::make_field<double, 1>("uc", m);
        auto u1             = u_const1;
    }
}
