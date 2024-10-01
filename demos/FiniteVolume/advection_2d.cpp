// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
#include <CLI/CLI.hpp>

#include <array>

#include <xtensor/xfixed.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/stencil_field.hpp>
#include <samurai/subset/subset_op.hpp>

#include <samurai/load_balancing.hpp>
#include <samurai/load_balancing_sfc.hpp>
#include <samurai/load_balancing_sfc_w.hpp>
#include <samurai/load_balancing_diffusion.hpp>
#include <samurai/load_balancing_force.hpp>
#include <samurai/load_balancing_diffusion_interval.hpp>
#include <samurai/load_balancing_void.hpp>

#include <samurai/timers.hpp>

#include <filesystem>
namespace fs = std::filesystem;

template <class Mesh>
auto init(Mesh& mesh, const double radius, const double x_center, const double y_center)
{
    auto u = samurai::make_field<double, 1>("u", mesh);

    u.fill( 0. );

    samurai::for_each_cell(
        mesh,
        [&](auto& cell)
        {
            auto center = cell.center();
            // const double radius   = .2;
            // const double x_center = 0.3;
            // const double y_center = 0.3;
            if (((center[0] - x_center) * (center[0] - x_center) + (center[1] - y_center) * (center[1] - y_center)) <= radius * radius)
            {
                u[cell] += 1;
            }

        });

    return u;
}

template <std::size_t dim, class Mesh>
auto init(Mesh& mesh, const std::vector<double> & radius, const std::vector<double> & rcenters )
{
    auto u = samurai::make_field<double, 1>("u", mesh);

    u.fill( 0. );

    samurai::for_each_cell(
        mesh,
        [&](auto& cell)
        {
            auto center = cell.center();

            for(std::size_t nc=0; nc<radius.size(); ++nc ){

                double dist = 0;
                for(size_t idim=0; idim<dim; ++idim) {
                    size_t id = nc * dim + idim;
                    dist += (center[ idim ] - rcenters[ id ]) * (center[ idim ] - rcenters[ id ]);
                }

                if ( dist <= radius[nc] * radius[nc] )
                {
                    u[cell] = 1.;
                }

            }

        });

    return u;
}

template <class Field>
void flux_correction(double dt, const std::array<double, 2>& a, const Field& u, Field& unp1)
{
    using mesh_t              = typename Field::mesh_t;
    using mesh_id_t           = typename mesh_t::mesh_id_t;
    using interval_t          = typename mesh_t::interval_t;
    constexpr std::size_t dim = Field::dim;

    auto mesh = u.mesh();

    for (std::size_t level = mesh.min_level(); level < mesh.max_level(); ++level)
    {
        xt::xtensor_fixed<int, xt::xshape<dim>> stencil;

        stencil = {
            {-1, 0}
        };

        auto subset_right = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level + 1], stencil),
                                                  mesh[mesh_id_t::cells][level])
                                .on(level);

        subset_right(
            [&](const auto& i, const auto& index)
            {
                auto j          = index[0];
                const double dx = samurai::cell_length(level);

                unp1(level, i, j) = unp1(level, i, j)
                                  + dt / dx
                                        * (samurai::upwind_op<dim, interval_t>(level, i, j).right_flux(a, u)
                                           - .5 * samurai::upwind_op<dim, interval_t>(level + 1, 2 * i + 1, 2 * j).right_flux(a, u)
                                           - .5 * samurai::upwind_op<dim, interval_t>(level + 1, 2 * i + 1, 2 * j + 1).right_flux(a, u));
            });

        stencil = {
            {1, 0}
        };

        auto subset_left = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level + 1], stencil),
                                                 mesh[mesh_id_t::cells][level])
                               .on(level);

        subset_left(
            [&](const auto& i, const auto& index)
            {
                auto j          = index[0];
                const double dx = samurai::cell_length(level);

                unp1(level, i, j) = unp1(level, i, j)
                                  - dt / dx
                                        * (samurai::upwind_op<dim, interval_t>(level, i, j).left_flux(a, u)
                                           - .5 * samurai::upwind_op<dim, interval_t>(level + 1, 2 * i, 2 * j).left_flux(a, u)
                                           - .5 * samurai::upwind_op<dim, interval_t>(level + 1, 2 * i, 2 * j + 1).left_flux(a, u));
            });

        stencil = {
            {0, -1}
        };

        auto subset_up = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level + 1], stencil), mesh[mesh_id_t::cells][level])
                             .on(level);

        subset_up(
            [&](const auto& i, const auto& index)
            {
                auto j          = index[0];
                const double dx = samurai::cell_length(level);

                unp1(level, i, j) = unp1(level, i, j)
                                  + dt / dx
                                        * (samurai::upwind_op<dim, interval_t>(level, i, j).up_flux(a, u)
                                           - .5 * samurai::upwind_op<dim, interval_t>(level + 1, 2 * i, 2 * j + 1).up_flux(a, u)
                                           - .5 * samurai::upwind_op<dim, interval_t>(level + 1, 2 * i + 1, 2 * j + 1).up_flux(a, u));
            });

        stencil = {
            {0, 1}
        };

        auto subset_down = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level + 1], stencil),
                                                 mesh[mesh_id_t::cells][level])
                               .on(level);

        subset_down(
            [&](const auto& i, const auto& index)
            {
                auto j          = index[0];
                const double dx = samurai::cell_length(level);

                unp1(level, i, j) = unp1(level, i, j)
                                  - dt / dx
                                        * (samurai::upwind_op<dim, interval_t>(level, i, j).down_flux(a, u)
                                           - .5 * samurai::upwind_op<dim, interval_t>(level + 1, 2 * i, 2 * j).down_flux(a, u)
                                           - .5 * samurai::upwind_op<dim, interval_t>(level + 1, 2 * i + 1, 2 * j).down_flux(a, u));
            });
    }
}

template <class Field>
void save(const fs::path& path, const std::string& filename, const Field& u, const std::string& suffix = "")
{
    auto mesh    = u.mesh();
    auto level_  = samurai::make_field<std::size_t, 1>("level", mesh);
    auto domain_ = samurai::make_field<int, 1>("domain", mesh);

    if (!fs::exists(path))
    {
        fs::create_directory(path);
    }

    int mrank = 0;

    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               level_[cell]  = cell.level;
                               domain_[cell] = mrank;
                           });
#ifdef SAMURAI_WITH_MPI
    mpi::communicator world;
    mrank = world.rank();
    samurai::save(path, fmt::format("{}_size_{}{}", filename, world.size(), suffix), mesh, u, level_);
#else
    samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, u, level_);
#endif
}

int main(int argc, char* argv[])
{
    samurai::initialize(argc, argv);

    constexpr std::size_t dim = 2;
    using Config              = samurai::MRConfig<dim>;

    // Simulation parameters
    double radius = 0.2, x_center = 0.3, y_center = 0.3;
    xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0., 0.};
    xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {4., 4.};
    std::array<double, dim> a{
        {1, 1}
    };
    double Tf  = .9;
    double cfl = 0.5;

    // Multiresolution parameters
    std::size_t min_level = 4;
    std::size_t max_level = 6;
    double mr_epsilon     = 2.e-4; // Threshold used by multiresolution
    double mr_regularity  = 1.;    // Regularity guess for multiresolution
    bool correction       = false;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "FV_advection_2d";
    std::size_t nfiles   = 10;
    std::size_t nt_loadbalance = 10; // nombre d'iteratio entre les equilibrages

    CLI::App app{"Finite volume example for the advection equation in 2d "
                 "using multiresolution"};
    app.add_option("--min-corner", min_corner, "The min corner of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--max-corner", max_corner, "The max corner of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--velocity", a, "The velocity of the advection equation")->capture_default_str()->group("Simulation parameters");
    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--min-level", min_level, "Minimum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--max-level", max_level, "Maximum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--nt-loadbalance", nt_loadbalance, "Maximum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--mr-eps", mr_epsilon, "The epsilon used by the multiresolution to adapt the mesh")
        ->capture_default_str()
        ->group("Multiresolution");
    app.add_option("--mr-reg",
                   mr_regularity,
                   "The regularity criteria used by the multiresolution to "
                   "adapt the mesh")
        ->capture_default_str()
        ->group("Multiresolution");
    app.add_option("--with-correction", correction, "Apply flux correction at the interface of two refinement levels")
        ->capture_default_str()
        ->group("Multiresolution");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Ouput");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Ouput");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Ouput");
    app.add_option("--radius", radius, "Bubble radius")->capture_default_str()->group("Simulation parameters");
    app.add_option("--xcenter", x_center, "Bubble x-axis center")->capture_default_str()->group("Simulation parameters");
    app.add_option("--ycenter", y_center, "Bubble y-axis center")->capture_default_str()->group("Simulation parameters");
    CLI11_PARSE(app, argc, argv);

    const samurai::Box<double, dim> box(min_corner, max_corner);
    samurai::MRMesh<Config> mesh{box, min_level, max_level};

    double dt            = cfl / static_cast<double>( 1 << max_level );
    const double dt_save = Tf / static_cast<double>(nfiles);
    double t             = 0.;

    std::vector<double> rcenters, radii;
    rcenters.emplace_back( x_center );
    rcenters.emplace_back( y_center );
    radii.emplace_back( radius );

    rcenters = { 0.8, 0.8, 
                 2., 2.,
                 3., 1. };
    radii    = { 0.8, 0.4, 0.2 };

    auto u = init<dim>(mesh, radii, rcenters);

    samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);
    auto unp1 = samurai::make_field<double, 1>("unp1", mesh);

    auto MRadaptation = samurai::make_MRAdapt(u);

    samurai::times::timers.start("MRadaptation");
    MRadaptation(mr_epsilon, mr_regularity);
    samurai::times::timers.stop("MRadaptation");

    save(path, filename, u, "_init");

    std::size_t nsave = 1;
    std::size_t nt    = 0;
    
    const int iof = 1;

    SFC_LoadBalancer_interval<dim, Morton> balancer;

    // Void_LoadBalancer<dim> balancer;
    // Diffusion_LoadBalancer_cell<dim> balancer;
    // Diffusion_LoadBalancer_interval<dim> balancer;
    // Load_balancing::Diffusion balancer;
    // Load_balancing::SFCw<dim, Morton> balancer;

    std::ofstream logs;
    boost::mpi::communicator world;
    logs.open( fmt::format("log_{}.dat", world.rank()), std::ofstream::app );

    while (t != Tf)
    {
        bool reqBalance = balancer.require_balance( mesh );

        if( reqBalance ) std::cerr << "\t> Load Balancing required !!! " << std::endl;

        // if ( ( nt % nt_loadbalance == 0 || reqBalance ) && nt > 1 )
        if ( ( nt % nt_loadbalance == 0 ) && nt > 1 )
        // if ( reqBalance && nt > 1 )
        {
            samurai::times::timers.start("load-balancing");
            balancer.load_balance(mesh, u);
            samurai::times::timers.stop("load-balancing");
        }

        samurai::times::timers.start("MRadaptation");
        MRadaptation(mr_epsilon, mr_regularity);
        samurai::times::timers.stop("MRadaptation");

        t += dt;
        if (t > Tf)
        {
            dt += Tf - t;
            t = Tf;
        }

        std::cout << fmt::format("iteration {}: t = {}, dt = {}", nt++, t, dt) << std::endl;

        samurai::update_ghost_mr(u);

        unp1.resize();

        samurai::times::timers.start("upwind");
        unp1 = u - dt * samurai::upwind(a, u);
        samurai::times::timers.stop("upwind");

        if (correction)
        {
            // flux_correction(dt, a, u, unp1);
        }

        std::swap(u.array(), unp1.array());

        samurai::times::timers.start("I/O");
        if (t >= static_cast<double>(nsave + 1) * dt_save || t == Tf || nt % iof == 0  )
        {
            const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            save(path, filename, u, suffix);
        }
        samurai::times::timers.stop("I/O");

    }

    samurai::finalize();
    return 0;
}
