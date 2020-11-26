
#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>

#include "../step_3/init_sol.hpp"
#include "../step_3/mesh.hpp"

#include "AMR_criterion.hpp"
#include "update_mesh.hpp"

int main()
{
    constexpr std::size_t dim = 1;
    std::size_t start_level = 8;
    std::size_t min_level = 2;
    std::size_t max_level = 8;

    samurai::Box<double, dim> box({-3}, {3});
    Mesh<MeshConfig<dim>> mesh(box, start_level, min_level, max_level);

    auto phi = init_sol(mesh);

    ////////////////////////////////
    while(true)
    {
        auto tag = samurai::make_field<std::size_t, 1>("tag", mesh);

        AMR_criterion(phi, tag);

        samurai::save("step4_criterion", mesh, phi, tag);

        if (update_mesh(phi, tag))
        {
            break;
        };
    }

    auto level = samurai::make_field<int, 1>("level", mesh);
    samurai::for_each_interval(mesh[MeshID::cells], [&](std::size_t l, const auto& i, auto)
    {
        level(l, i) = l;
    });

    samurai::save("step4", mesh, phi, level);
    ////////////////////////////////

    return 0;
}

