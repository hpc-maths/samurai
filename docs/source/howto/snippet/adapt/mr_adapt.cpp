#include <cmath>
#include <iostream>

#include <samurai/algorithm.hpp>
#include <samurai/algorithm/update.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>

// Print the number of cells of each level of the mesh
template <class Mesh>
void print_cells_per_level(const Mesh& mesh)
{
    using mesh_id_t = typename Mesh::mesh_id_t;
    for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
    {
        std::cout << "  level " << level << ": " << mesh.nb_cells(level, mesh_id_t::cells) << " cells\n";
    }
    std::cout << "  total: " << mesh.nb_cells(mesh_id_t::cells) << " cells" << std::endl;
}

int main(int argc, char** argv)
{
    samurai::initialize("Multiresolution adaptation", argc, argv);
    SAMURAI_PARSE(argc, argv);

    static constexpr std::size_t dim = 2;

    // Create the mesh and the field
    samurai::Box<double, dim> box({0.0, 0.0}, {1.0, 1.0});
    auto config = samurai::mesh_config<dim>().min_level(3).max_level(7);
    auto mesh   = samurai::mra::make_mesh(box, config);

    auto u = samurai::make_scalar_field<double>("u", mesh);
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               const auto x    = cell.center();
                               const double r2 = (x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5);
                               u[cell]         = std::exp(-200. * r2);
                           });
    samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);

    // A second field that follows the mesh without driving the adaptation
    auto v = samurai::make_scalar_field<double>("v", mesh);
    v.fill(1.);

    std::cout << "Before adaptation:\n";
    print_cells_per_level(mesh);

    // Adapt the mesh
    auto MRadaptation = samurai::make_MRAdapt(u);
    auto mra_config   = samurai::mra_config().epsilon(1e-3).regularity(1.).relative_detail(false);
    MRadaptation(mra_config, v);

    std::cout << "After adaptation:\n";
    print_cells_per_level(mesh);

    // Fill the ghosts before reading the neighbors of a cell
    samurai::update_ghost_mr(u);

    samurai::finalize();
    return 0;
}
