#include <samurai/amr/mesh.hpp>
#include <samurai/box.hpp>

int main()
{
    static constexpr std::size_t dim = 2;

    samurai::Box<double, dim> box({0.0, 0.0}, {1.0, 1.0});

    auto config = samurai::mesh_config<dim>().min_level(2).max_level(5).start_level(4);
    auto mesh   = samurai::amr::make_Mesh(box, config);

    return 0;
}
