#include <samurai/box.hpp>
#include <samurai/samurai.hpp>

int main(int argc, char* argv[])
{
    samurai::initialize(argc, argv);

    static constexpr std::size_t dim = 2;

    samurai::Box<double, dim> box({-1.0, -1.0}, {1.0, 1.0});

    samurai::finalize();
    return 0;
}
