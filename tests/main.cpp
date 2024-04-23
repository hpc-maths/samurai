#include <gtest/gtest.h>
#include <samurai/samurai.hpp>

int main(int argc, char* argv[])
{
    samurai::initialize();

    ::testing::InitGoogleTest(&argc, argv);

    int ret = RUN_ALL_TESTS();

    samurai::finalize();

    return ret;
}
