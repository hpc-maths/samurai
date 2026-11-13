#include <gtest/gtest.h>

#include <samurai/samurai_config.hpp>

namespace samurai
{
    namespace
    {
        SAMURAI_ALWAYS_INLINE double twice(double x)
        {
            return 2 * x;
        }

        template <class T>
        SAMURAI_ALWAYS_INLINE T thrice(T x)
        {
            return 3 * x;
        }

        struct Scale
        {
            double factor;

            SAMURAI_ALWAYS_INLINE double operator()(double x) const
            {
                return factor * x;
            }

            static SAMURAI_ALWAYS_INLINE double half(double x)
            {
                return 0.5 * x;
            }
        };
    }

    // The macro must be usable on free functions, templates and member functions with every compiler.
    TEST(always_inline, applies_to_functions_templates_and_members)
    {
        EXPECT_EQ(twice(2.), 4.);
        EXPECT_EQ(thrice(2), 6);
        EXPECT_EQ(Scale{4.}(2.), 8.);
        EXPECT_EQ(Scale::half(2.), 1.);
    }
}
