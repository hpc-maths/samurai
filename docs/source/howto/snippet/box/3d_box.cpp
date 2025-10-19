#include <samurai/box.hpp>

int main()
{
    static constexpr std::size_t dim = 3;

    samurai::Box<double, dim> box({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});

    return 0;
}
