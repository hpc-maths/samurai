#include <samurai/samurai.hpp>

int main(int argc, char** argv)
{
    samurai::initialize("Simple example", argc, argv);

    samurai::finalize();
    return 0;
}
