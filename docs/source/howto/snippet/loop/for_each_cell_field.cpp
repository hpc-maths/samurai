#include <samurai/algorithm.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>

int main()
{
    static constexpr std::size_t dim = 2;

    samurai::Box<double, dim> box({0.0, 0.0}, {1.0, 1.0});
    auto config = samurai::mesh_config<dim>().min_level(2).max_level(5);
    auto mesh   = samurai::make_MRMesh(config, box);

    auto field = samurai::make_scalar_field<double>("u", mesh);

    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               auto x = cell.center(0);
                               auto y = cell.center(1);

                               field[cell] = std::exp(-((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)) * 20.0);
                           });

    return 0;
}
