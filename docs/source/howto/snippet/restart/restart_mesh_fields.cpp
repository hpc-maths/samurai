#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
namespace fs = std::filesystem;

#include <samurai/algorithm.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/io/restart.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>

static constexpr std::size_t dim = 2;

// Used by the run that writes the checkpoint and by the restart
auto make_config()
{
    return samurai::mesh_config<dim>().min_level(3).max_level(7);
}

// Print the number of cells of each level of the mesh
template <class Mesh>
void print_cells_per_level(const Mesh& mesh)
{
    using mesh_id_t = typename Mesh::mesh_id_t;
    for (auto l = mesh.min_level(); l <= mesh.max_level(); ++l)
    {
        const auto n = mesh.nb_cells(l, mesh_id_t::cells);
        std::cout << "  level " << l << ": " << n << " cells\n";
    }
    const auto n = mesh.nb_cells(mesh_id_t::cells);
    std::cout << "  total: " << n << " cells" << std::endl;
}

int main(int argc, char** argv)
{
    samurai::initialize("Restart from a checkpoint", argc, argv);
    SAMURAI_PARSE(argc, argv);

    const fs::path path = "checkpoints";
    fs::create_directories(path);

    // First run: compute a solution on an adapted mesh
    samurai::Box<double, dim> box({0.0, 0.0}, {1.0, 1.0});
    auto mesh = samurai::mra::make_mesh(box, make_config());

    auto u    = samurai::make_scalar_field<double>("u", mesh);
    auto v    = samurai::make_vector_field<double, dim>("v", mesh);
    auto init = [&](const auto& cell)
    {
        const auto x    = cell.center();
        const double dx = x[0] - 0.5;
        const double dy = x[1] - 0.5;
        u[cell]         = std::exp(-200. * (dx * dx + dy * dy));
        v[cell][0]      = x[0];
        v[cell][1]      = x[1];
    };
    samurai::for_each_cell(mesh, init);
    samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);

    auto MRadaptation = samurai::make_MRAdapt(u);
    auto mra_config   = samurai::mra_config().epsilon(1e-3);
    MRadaptation(mra_config, v);

    double t              = 0.25;
    std::size_t iteration = 10;

    // Write the checkpoint
    samurai::dump(
        path,
        "checkpoint",
        [&](samurai::MetadataWriter& writer)
        {
            writer.time(t).attribute("iteration", iteration);
        },
        mesh,
        u,
        v);
    // End of the checkpoint

    // Restart: load the checkpoint into an empty mesh and fields
    auto new_mesh = samurai::mra::make_empty_mesh(make_config());
    auto new_u    = samurai::make_scalar_field<double>("u", new_mesh);
    auto new_v    = samurai::make_vector_field<double, dim>("v", new_mesh);

    double new_t              = 0.;
    std::size_t new_iteration = 0;

    samurai::load(
        path,
        "checkpoint",
        [&](const samurai::MetadataReader& reader)
        {
            new_t         = reader.time();
            new_iteration = reader.attribute<std::size_t>("iteration");
        },
        new_mesh,
        new_u,
        new_v);

    // The checkpoint holds no boundary conditions: attach them again
    samurai::make_bc<samurai::Dirichlet<1>>(new_u, 0.);
    // End of the restart

    // Check that the mesh, the fields and the metadata are restored
    std::cout << "Dumped mesh:\n";
    print_cells_per_level(mesh);
    std::cout << "Loaded mesh:\n";
    print_cells_per_level(new_mesh);

    double max_diff  = 0.;
    auto update_diff = [&](const auto& cell)
    {
        max_diff = std::max(max_diff, std::abs(u[cell] - new_u[cell]));
        for (std::size_t d = 0; d < dim; ++d)
        {
            const double dv = v[cell][d] - new_v[cell][d];
            max_diff        = std::max(max_diff, std::abs(dv));
        }
    };
    samurai::for_each_cell(mesh, update_diff);

    std::cout << std::boolalpha;
    std::cout << "Same mesh: " << (mesh == new_mesh) << "\n";
    std::cout << "Max difference of u and v: " << max_diff << "\n";
    std::cout << "Time: " << new_t;
    std::cout << ", iteration: " << new_iteration << std::endl;

    samurai::finalize();
    return 0;
}
