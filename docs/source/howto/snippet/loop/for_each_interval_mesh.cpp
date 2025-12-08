#include <samurai/algorithm.hpp>
#include <samurai/box.hpp>
#include <samurai/mr/mesh.hpp>

int main()
{
    static constexpr std::size_t dim = 2;

    samurai::Box<double, dim> box({0.0, 0.0}, {1.0, 1.0});
    auto config = samurai::mesh_config<dim>().min_level(2).max_level(5);
    auto mesh   = samurai::make_MRMesh(box, config);

    samurai::for_each_interval(mesh,
                               [&](std::size_t level, const auto& interval, const auto& index)
                               {
                                   auto y = index[0];
                                   std::cout << "Level: " << level << ", x: " << interval << ", y: " << y << std::endl;
                               });

    return 0;
}
