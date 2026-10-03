#include <cmath>
#include <iostream>

#include <samurai/algorithm.hpp>
#include <samurai/algorithm/graduation.hpp>
#include <samurai/algorithm/update.hpp>
#include <samurai/amr/mesh.hpp>
#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/cell_flag.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/samurai.hpp>
#include <samurai/schemes/fv.hpp>

// Define the refinement criterion
template <class Field, class Tag>
void tag_cells(const Field& u, Tag& tag, double delta)
{
    using namespace samurai::math;
    using mesh_id_t = typename Field::mesh_t::mesh_id_t;

    const auto& mesh = u.mesh();

    auto refine = [](auto& e)
    {
        e = static_cast<int>(samurai::CellFlag::refine);
    };
    auto coarsen = [](auto& e)
    {
        e = static_cast<int>(samurai::CellFlag::coarsen);
    };

    auto tag_interval = [&](std::size_t level, const auto& i, auto)
    {
        const double dx = mesh.cell_length(level);
        auto jump       = abs(u(level, i + 1) - u(level, i - 1));
        auto du_dx      = samurai::eval(jump / (2. * dx));
        auto steep      = du_dx > delta;

        if (level < mesh.max_level())
        {
            samurai::apply_on_masked(tag(level, i), steep, refine);
        }
        if (level > mesh.min_level())
        {
            samurai::apply_on_masked(tag(level, i), !steep, coarsen);
        }
    };

    tag.fill(static_cast<int>(samurai::CellFlag::keep));
    samurai::for_each_interval(mesh[mesh_id_t::cells], tag_interval);
}

int main(int argc, char* argv[])
{
    // Read the options
    auto& app = samurai::initialize("1D Burgers with AMR", argc, argv);

    double Tf    = 1.5;  // final time, after the shock forms at T* = 1
    double cfl   = 0.99; // CFL number, stable for cfl <= 1
    double delta = 0.1;  // refinement threshold on |du/dx|
    app.option_defaults()->always_capture_default();
    app.add_option("--Tf", Tf, "Final time");
    app.add_option("--cfl", cfl, "CFL number");
    app.add_option("--delta", delta, "Refinement threshold");
    SAMURAI_PARSE(argc, argv);

    // Build the mesh
    constexpr std::size_t dim = 1;

    const samurai::Box<double, dim> box({-3}, {3});
    auto config = samurai::mesh_config<dim>();
    config.min_level(2).max_level(8);
    auto mesh = samurai::amr::make_mesh(box, config);

    std::cout << mesh << std::endl;

    // Set the initial condition
    auto u = samurai::make_scalar_field<double>("u", mesh);

    auto hat = [](double x)
    {
        return (x < -1. || x > 1.) ? 0. : 1. - std::abs(x);
    };
    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               u[cell] = hat(cell.center(0));
                           });

    // Set the boundary condition
    samurai::make_bc<samurai::Neumann<1>>(u, 0.);

    // Define the scheme
    auto conv = 0.5 * samurai::make_convection_upwind<decltype(u)>();

    // Write the time loop
    using mesh_id_t = decltype(mesh)::mesh_id_t;

    auto unp1 = samurai::make_scalar_field<double>("unp1", mesh);
    auto tag  = samurai::make_scalar_field<int>("tag", mesh);

    const xt::xtensor_fixed<int, xt::xshape<2, 1>> stencil{{1}, {-1}};

    double dt      = cfl * mesh.min_cell_length(); // sup |u_0| = 1
    double t       = 0.;
    std::size_t nt = 0;

    // A pass changes a cell by one level at most
    const std::size_t npasses = mesh.max_level() - mesh.min_level();

    while (t != Tf)
    {
        // Adapt the mesh
        for (std::size_t pass = 0; pass < npasses; ++pass)
        {
            samurai::update_ghost(u);
            tag.resize();
            tag_cells(u, tag, delta);
            samurai::graduation(tag, stencil);
            if (samurai::update_field(tag, u))
            {
                break;
            }
        }
        unp1.resize();

        // Advance the solution
        t += dt;
        if (t > Tf)
        {
            dt += Tf - t;
            t = Tf;
        }

        unp1 = u - dt * conv(u);

        samurai::swap(u, unp1);

        const auto ncells = mesh.nb_cells(mesh_id_t::cells);
        std::cout << "iteration " << ++nt << ": t = " << t;
        std::cout << ", " << ncells << " cells" << std::endl;
    }

    // Save the solution
    auto lvl = samurai::make_scalar_field<std::size_t>("level", mesh);
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               lvl[cell] = cell.level;
                           });
    samurai::save("burgers_1d", mesh, u, lvl);

    samurai::finalize();
    return 0;
}
