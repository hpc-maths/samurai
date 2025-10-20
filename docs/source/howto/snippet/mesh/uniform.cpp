#include <samurai/box.hpp>
#include <samurai/uniform_mesh.hpp>

int main()
{
    static constexpr std::size_t dim = 2;
    using config_t                   = samurai::UniformConfig<dim>;

    samurai::Box<double, dim> box({0.0, 0.0}, {1.0, 1.0});
    samurai::UniformMesh<config_t> mesh(box, 4); // 4 is the level of refinement

    return 0;
}
