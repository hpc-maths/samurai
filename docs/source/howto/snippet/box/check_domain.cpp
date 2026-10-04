#include <iostream>

#include <samurai/domain_builder.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>

int main(int argc, char* argv[])
{
    samurai::initialize(argc, argv);

    static constexpr std::size_t dim = 2;

    samurai::DomainBuilder<dim> domain({0.0, 0.0}, {2.0, 1.0});
    domain.add({0.0, 1.0}, {1.0, 2.0});

    auto config = samurai::mesh_config<dim>();

    config.min_level(2).max_level(4);
    auto mesh = samurai::mra::make_mesh(domain, config);

    using mesh_id_t = typename decltype(mesh)::mesh_id_t;
    std::cout << mesh.nb_cells(mesh_id_t::cells) << std::endl;

    samurai::save("l_shaped_domain", mesh);

    samurai::finalize();
    return 0;
}
