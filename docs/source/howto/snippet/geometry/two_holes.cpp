#include <iostream>

#include <samurai/domain_builder.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>

int main(int argc, char* argv[])
{
    samurai::initialize(argc, argv);

    static constexpr std::size_t dim = 2;

    // A 2 x 1 channel with two square obstacles
    samurai::DomainBuilder<dim> domain({0.0, 0.0}, {2.0, 1.0});
    domain.remove({0.25, 0.25}, {0.5, 0.5});
    domain.remove({1.25, 0.5}, {1.5, 0.75});

    auto config = samurai::mesh_config<dim>();
    config.min_level(2).max_level(5);
    auto mesh = samurai::mra::make_mesh(domain, config);

    using mesh_id_t = typename decltype(mesh)::mesh_id_t;
    std::size_t n   = mesh.nb_cells(mesh_id_t::cells);
    double dx       = mesh.cell_length(mesh.max_level());
    double area     = static_cast<double>(n) * dx * dx;
    std::cout << "cells: " << n << '\n';
    std::cout << "cell length: " << dx << '\n';
    std::cout << "covered area: " << area << '\n';

    samurai::save("two_holes", mesh);

    samurai::finalize();
    return 0;
}
