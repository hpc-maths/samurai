#include <samurai/samurai.hpp>

int main(int argc, char* argv[])
{
    samurai::initialize("MySamuraiProject", argc, argv);
    samurai::finalize();
    return 0;
}
