#include <samurai/algorithm.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/timers.hpp>

int main(int argc, char** argv)
{
    static constexpr std::size_t dim = 2;
    using config_t                   = samurai::MRConfig<dim>;

    samurai::initialize("Custom timer example", argc, argv);

    samurai::Box<double, dim> box({0.0, 0.0}, {1.0, 1.0});
    samurai::MRMesh<config_t> mesh(box, 2, 5); // min level 2, max level 5

    auto field = samurai::make_scalar_field<double>("u", mesh);

    samurai::times::timers.start("init field");

    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               auto x = cell.center(0);
                               auto y = cell.center(1);

                               field[cell] = std::exp(-((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)) * 20.0);
                           });

    samurai::times::timers.stop("init field");

    samurai::finalize();
    return 0;
}
