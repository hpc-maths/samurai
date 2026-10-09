#include <cmath>
#include <iomanip>
#include <iostream>

#include <samurai/algorithm.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>

int main(int argc, char** argv)
{
    samurai::initialize("Multiresolution adaptation in 1D", argc, argv);
    SAMURAI_PARSE(argc, argv);

    static constexpr std::size_t dim = 1;

    // Create the mesh and a bump centered in [0, 1)
    samurai::Box<double, dim> box({0.0}, {1.0});
    auto config = samurai::mesh_config<dim>();
    config.min_level(3).max_level(7);
    auto mesh = samurai::mra::make_mesh(box, config);

    auto u = samurai::make_scalar_field<double>("u", mesh);
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               const double dx = cell.center(0) - 0.5;
                               u[cell]         = std::exp(-200. * dx * dx);
                           });
    samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);

    // Adapt the mesh
    auto MRadaptation = samurai::make_MRAdapt(u);
    auto mra_config   = samurai::mra_config();
    mra_config.epsilon(1e-3);
    MRadaptation(mra_config);

    // The cells and the threshold on the details of each level
    using mesh_id_t      = typename decltype(mesh)::mesh_id_t;
    const auto max_level = mesh.max_level();
    const double epsilon = mra_config.epsilon();
    for (auto l = mesh.min_level(); l <= max_level; ++l)
    {
        // eps_l = epsilon / 2^(dim (max_level - l))
        const double eps_l = epsilon
                           / static_cast<double>(1 << (dim * (max_level - l)));
        std::cout << "level " << l << ": threshold " << std::scientific
                  << std::setprecision(2) << eps_l;
        std::cout << ", " << mesh.nb_cells(l, mesh_id_t::cells) << " cells";
        samurai::for_each_interval(mesh[mesh_id_t::cells][l],
                                   [](auto, const auto& i, auto)
                                   {
                                       std::cout << " [" << i.start << ", "
                                                 << i.end << ")";
                                   });
        std::cout << "\n";
    }
    std::cout << "total: " << mesh.nb_cells(mesh_id_t::cells) << " cells"
              << std::endl;

    samurai::finalize();
    return 0;
}
