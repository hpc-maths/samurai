#include <samurai/amr/mesh.hpp>
#include <samurai/box.hpp>

int main()
{
    static constexpr std::size_t dim = 2;
    using config_t                   = samurai::amr::Config<dim>;

    samurai::Box<double, dim> box({0.0, 0.0}, {1.0, 1.0});
    samurai::amr::Mesh<config_t> mesh(box, 4, 2, 5); // start level 4, min level 2, max level 5

    return 0;
}
