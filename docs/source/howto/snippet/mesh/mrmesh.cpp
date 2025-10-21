#include <samurai/box.hpp>
#include <samurai/mr/mesh.hpp>

int main()
{
    static constexpr std::size_t dim = 2;

    samurai::Box<double, dim> box({0.0, 0.0}, {1.0, 1.0});

    auto config = samurai::mesh_config<dim>().min_level(2).max_level(5);
    auto mesh   = samurai::make_MRMesh(config, box);

    return 0;
}
