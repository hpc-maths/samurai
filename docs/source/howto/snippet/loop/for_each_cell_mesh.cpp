#include <samurai/algorithm.hpp>
#include <samurai/box.hpp>
#include <samurai/mr/mesh.hpp>

int main()
{
    static constexpr std::size_t dim = 2;
    using config_t                   = samurai::MRConfig<dim>;

    samurai::Box<double, dim> box({0.0, 0.0}, {1.0, 1.0});
    samurai::MRMesh<config_t> mesh(box, 2, 5); // min level 2, max level 5

    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               std::cout << "Cell level: " << cell.level << ", center: " << cell.center() << std::endl;
                           });

    return 0;
}
