#include <samurai/algorithm.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>

int main()
{
    static constexpr std::size_t dim = 2;

    samurai::Box<double, dim> box({-1.0, -1.0}, {1.0, 1.0});
    auto config = samurai::mesh_config<dim>().min_level(0).max_level(2);
    auto mesh   = samurai::make_MRMesh(config, box);

    auto field = samurai::make_scalar_field<double>("u", mesh);

    samurai::for_each_interval(mesh,
                               [&](std::size_t level, const auto& i, const auto& index)
                               {
                                   auto j = index[0];
                                   auto x = mesh.cell_length(level) * (xt::arange(i.start, i.end) + 0.5) + mesh.origin_point()[0];
                                   auto y = mesh.cell_length(level) * (j + 0.5) + mesh.origin_point()[1];

                                   field(level, i, j) = xt::exp(-((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)) * 20.0);

                                   std::cout << "Level: " << level << ", x: " << x << ", y: " << y << std::endl;
                               });

    return 0;
}
