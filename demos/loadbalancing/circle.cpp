#include <iostream>

// CLI
#include <CLI/CLI.hpp>

// samurai 
#include <samurai/samurai.hpp>
#include <samurai/timers.hpp>
#include <samurai/field.hpp>

#include <samurai/load_balancing.hpp>
#include <samurai/load_balancing_sfc.hpp>
#include <samurai/load_balancing_diffusion_interval.hpp>
#include <samurai/load_balancing_diffusion_cell.hpp>

#include <samurai/mr/mesh.hpp>
#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/assertLogTrace.hpp>
#include <samurai/schemes/fv.hpp>

#include <cassert>
#include <algorithm>

template <int dim, class Mesh>
auto init(Mesh& mesh, const double radius, const double x_center, const double y_center )
{
    auto u = samurai::make_field<double, 1>("u", mesh);

    samurai::for_each_cell( mesh, [&](auto& cell) {
        auto center = cell.center();
        double rad  = 0.;
        double z_center = 0.5;
        
        if constexpr ( dim == 3 ){
            rad = ((center[0] - x_center) * (center[0] - x_center) + 
                   (center[1] - y_center) * (center[1] - y_center) +
                   (center[2] - z_center) * (center[2] - z_center) );
        }else{
            rad = ((center[0] - x_center) * (center[0] - x_center) + 
                   (center[1] - y_center) * (center[1] - y_center) );
        }
        if ( rad <= radius * radius) {
            u[cell] = 1;
        } else {
            u[cell] = 0;
        }
    });

    return u;
}

template <int dim, class Mesh, class Coord_t>
auto initMultiCircles(Mesh& mesh, const std::vector<double> &radius, const std::vector<Coord_t> & centers )
{
    auto u = samurai::make_field<double, 1>("u", mesh);

    samurai::for_each_cell( mesh, [&](auto& cell) {
        auto cc = cell.center();

        for(size_t icerc=0; icerc<centers.size(); ++icerc ){
            double rad  = 0.;
            
            for( size_t idim=0; idim<dim; ++idim ){
                rad += (cc[ idim ] - centers[ icerc ][ idim ]) * (cc[ idim ] - centers[ icerc ][ idim ]);
            }

            if ( rad <= radius[ icerc ] * radius[ icerc ]) {
                u[ cell ] += 1;
            } else {
                u[ cell ] += 0;
            }
        }

    });

    return u;
}

int main( int argc, char * argv[] ){

    samurai::initialize(argc, argv);

    constexpr int dim = 2;
    int ndomains = 1, nbIterLoadBalancing = 2;
    std::size_t minLevel = 2, maxLevel = 8;
    double mr_regularity = 1.0, mr_epsilon = 2.e-4;
    xt::xtensor_fixed<double, xt::xshape<dim>> minCorner = {0., 0., 0.};
    xt::xtensor_fixed<double, xt::xshape<dim>> maxCorner = {1., 1., 1.};
    std::string filename = "ld_mesh_init";

    double radius = 0.2, x_center = 0.5, y_center = 0.5;
    int niterBench = 1;

    CLI::App app{"Load balancing test"};
    app.add_option("--ndomains", ndomains, "Number of desired domains")->capture_default_str()->group("Simulation parameters");
    app.add_option("--nbIterLB", nbIterLoadBalancing, "Number of desired lb iteration")
        ->capture_default_str()->group("Simulation parameters");
    app.add_option("--niterBench", niterBench, "Number of iteration for bench")->capture_default_str()->group("Simulation parameters");
    app.add_option("--min-corner", minCorner, "The min corner of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--max-corner", maxCorner, "The max corner of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--min-level", minLevel, "Minimum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--max-level", maxLevel, "Maximum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--mr-eps", mr_epsilon, "The epsilon used by the multiresolution to adapt the mesh")
        ->capture_default_str()->group("Multiresolution");
    app.add_option("--mr-reg", mr_regularity,"The regularity criteria used by the multiresolution to "
                   "adapt the mesh")->capture_default_str()->group("Multiresolution");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Ouput");
    app.add_option("--radius", radius, "Bubble radius")->capture_default_str()->group("Simulation parameters");
    app.add_option("--xcenter", x_center, "Bubble x-axis center")->capture_default_str()->group("Simulation parameters");
    app.add_option("--ycenter", y_center, "Bubble y-axis center")->capture_default_str()->group("Simulation parameters");
    
    CLI11_PARSE(app, argc, argv);

    double cfl = 0.5;
    double dt  = cfl / (1 << maxLevel);

    // Convection operator
    samurai::VelocityVector<dim> velocity;
    velocity.fill(1);
    velocity(1) = -1;

    using Config    = samurai::MRConfig<dim>;
    using Mesh_t    = samurai::MRMesh<Config>;
    using mesh_id_t = typename Mesh_t::mesh_id_t;

    boost::mpi::communicator world;

    if( world.rank() == 0 ) std::cerr << "\t> Testing Diffusion_LoadBalancer_interval " << std::endl;

    { // load balancing cells by cells using gravity loadbalancer

        Timers myTimers;

        /**
         * Initialize AMR mesh with spherical buble 
        */
        myTimers.start("InitMesh");

        samurai::Box<double, dim> box( minCorner, maxCorner );

        Mesh_t mesh(box, minLevel, maxLevel);

        std::vector<std::array<double, dim>> bubles_c = {{0.2, 0.2}, {0.5, 0.5}, {0.8, 0.8}};

        auto u = init<dim>( mesh, radius, x_center, y_center );

        // auto u = initMultiCircles<dim>( mesh, {0.2, 0.1, 0.05}, bubles_c );

        samurai::make_bc<samurai::Dirichlet>(u, 0.);
        auto mradapt = samurai::make_MRAdapt( u );
        mradapt( mr_epsilon, mr_regularity );

        myTimers.stop("InitMesh");

        samurai::save( "init_circle_"+std::to_string( ndomains)+"_domains", mesh );

        myTimers.start( "balance_DIF_inter" );
        Diffusion_LoadBalancer_interval<dim> _diff_lb;
        for(size_t lb_iter=0; lb_iter<nbIterLoadBalancing; ++lb_iter ){
            _diff_lb.load_balance( mesh );
            auto s_ = fmt::format("balance_diffusion_interval_{}_iter_{}", ndomains, lb_iter);
            samurai::save( s_ , mesh );
        }
        myTimers.stop( "balance_DIF_inter" );

        _diff_lb.evaluate_balancing( mesh );

        {
            Mesh_t newmesh( mesh[mesh_id_t::cells], mesh );
            auto u2 = init<dim>( newmesh, radius, x_center, y_center );
            
            samurai::save( "init2_circle_"+std::to_string( ndomains)+"_domains", newmesh, u2 );

            auto conv = samurai::make_convection<decltype(u2)>(velocity);

            // myTimers.start( "update_ghost_mr" );
            // for(int i=0; i<niterBench; ++i){
            //     samurai::update_ghost_mr(u2);
            // }           
            // myTimers.stop( "update_ghost_mr" );

            auto unp1 = samurai::make_field<double, 1>("unp1", newmesh);

            myTimers.start( "upwind" );
            for(int i=0; i<niterBench; ++i){
                unp1 = u2 - dt * conv( u2 );
            }
            myTimers.stop( "upwind" );

            
            // auto mradapt = samurai::make_MRAdapt( u2 );

            // myTimers.start( "mradapt" );
            // mradapt( mr_epsilon, mr_regularity );
            // myTimers.stop( "mradapt" );

        }

        

        myTimers.print();
    
    }

    
    world.barrier();

    if( world.rank() == 0 ) std::cerr << "\t> Testing Diffusion_LoadBalancer_cell " << std::endl;

    { // load balancing cells by cells using gravity loadbalancer

        Timers myTimers;

        /**
         * Initialize AMR mesh with spherical buble 
        */
        myTimers.start("InitMesh");

        samurai::Box<double, dim> box( minCorner, maxCorner );

        Mesh_t mesh(box, minLevel, maxLevel);

        std::vector<std::array<double, dim>> bubles_c = {{0.2, 0.2}, {0.5, 0.5}, {0.8, 0.8}};

        auto u = init<dim>( mesh, radius, x_center, y_center );
        // auto u = initMultiCircles<dim>( mesh, {0.2, 0.1, 0.05}, bubles_c );

        samurai::make_bc<samurai::Dirichlet>(u, 0.);
        auto mradapt = samurai::make_MRAdapt( u );
        mradapt( mr_epsilon, mr_regularity );

        myTimers.stop("InitMesh");

        samurai::save( "init_circle_"+std::to_string( ndomains)+"_domains", mesh );

        myTimers.start( "balance_DIF_cell" );
        Diffusion_LoadBalancer_cell<dim> _diff_lb_c;
        for(int lb_iter=0; lb_iter<nbIterLoadBalancing; ++lb_iter ){
            _diff_lb_c.load_balance( mesh );
            auto s_ = fmt::format("balance_diffusion_cell_{}_iter_{}", ndomains, lb_iter);
            samurai::save( s_ , mesh );
        }
        myTimers.stop( "balance_DIF_cell" );

        _diff_lb_c.evaluate_balancing( mesh );

        {
            Mesh_t newmesh( mesh[mesh_id_t::cells], mesh );
            auto u2 = init<dim>( newmesh, radius, x_center, y_center );
            
            samurai::save( "init2_circle_"+std::to_string( ndomains)+"_domains", newmesh, u2 );

            auto conv = samurai::make_convection<decltype(u2)>(velocity);

            // myTimers.start( "update_ghost_mr" );
            // for(int i=0; i<niterBench; ++i){
            //     samurai::update_ghost_mr(u2);
            // }           
            // myTimers.stop( "update_ghost_mr" );

            auto unp1 = samurai::make_field<double, 1>("unp1", newmesh);

            myTimers.start( "upwind" );
            for(int i=0; i<niterBench; ++i){
                unp1 = u2 - dt * conv( u2 );
            }
            myTimers.stop( "upwind" );

            
            // auto mradapt = samurai::make_MRAdapt( u2 );

            // myTimers.start( "mradapt" );
            // mradapt( mr_epsilon, mr_regularity );
            // myTimers.stop( "mradapt" );

        }

        myTimers.print();
    
    }

    world.barrier();

    if( world.rank() == 0 ) std::cerr << "\t> Testing Morton_LoadBalancer_interval " << std::endl;

    { // load balancing cells by cells using gravity loadbalancer

        Timers myTimers;

        /**
         * Initialize AMR mesh with spherical buble 
        */
        myTimers.start("InitMesh");

        samurai::Box<double, dim> box( minCorner, maxCorner );

        Mesh_t mesh(box, minLevel, maxLevel);

        std::vector<std::array<double, dim>> bubles_c = {{0.2, 0.2}, {0.5, 0.5}, {0.8, 0.8}};

        auto u = init<dim>( mesh, radius, x_center, y_center );
        // auto u = initMultiCircles<dim>( mesh, {0.2, 0.1, 0.05}, bubles_c );

        samurai::make_bc<samurai::Dirichlet>(u, 0.);
        auto mradapt = samurai::make_MRAdapt( u );
        mradapt( mr_epsilon, mr_regularity );

        myTimers.stop("InitMesh");

        samurai::save( "init_circle_"+std::to_string( ndomains)+"_domains", mesh );

        myTimers.start( "balance_Morton_cell" );
        SFC_LoadBalancer_interval<dim, Morton> loadb_morton;
        for(int lb_iter=0; lb_iter<nbIterLoadBalancing; ++lb_iter ){
            loadb_morton.load_balance( mesh );
            auto s_ = fmt::format("balance_morton_cell_{}_iter_{}", ndomains, lb_iter);
            samurai::save( s_ , mesh );
        }
        myTimers.stop( "balance_Morton_cell" );

        loadb_morton.evaluate_balancing( mesh );

        {
            Mesh_t newmesh( mesh[mesh_id_t::cells], mesh );
            auto u2 = init<dim>( newmesh, radius, x_center, y_center );
            
            samurai::save( "init2_circle_"+std::to_string( ndomains)+"_domains", newmesh, u2 );

            auto conv = samurai::make_convection<decltype(u2)>(velocity);

            // myTimers.start( "update_ghost_mr" );
            // for(int i=0; i<niterBench; ++i){
            //     samurai::update_ghost_mr(u2);
            // }           
            // myTimers.stop( "update_ghost_mr" );

            auto unp1 = samurai::make_field<double, 1>("unp1", newmesh);

            myTimers.start( "upwind" );
            for(int i=0; i<niterBench; ++i){
                unp1 = u2 - dt * conv( u2 );
            }
            myTimers.stop( "upwind" );

            
            // auto mradapt = samurai::make_MRAdapt( u2 );

            // myTimers.start( "mradapt" );
            // mradapt( mr_epsilon, mr_regularity );
            // myTimers.stop( "mradapt" );

        }

        myTimers.print();
    
    }

     if( world.rank() == 0 ) std::cerr << "\t> Testing Morton_LoadBalancer_interval " << std::endl;

    { // load balancing cells by cells using gravity loadbalancer

        Timers myTimers;

        /**
         * Initialize AMR mesh with spherical buble 
        */
        myTimers.start("InitMesh");

        samurai::Box<double, dim> box( minCorner, maxCorner );

        Mesh_t mesh(box, minLevel, maxLevel);

        std::vector<std::array<double, dim>> bubles_c = {{0.2, 0.2}, {0.5, 0.5}, {0.8, 0.8}};

        auto u = init<dim>( mesh, radius, x_center, y_center );
        // auto u = initMultiCircles<dim>( mesh, {0.2, 0.1, 0.05}, bubles_c );

        samurai::make_bc<samurai::Dirichlet>(u, 0.);
        auto mradapt = samurai::make_MRAdapt( u );
        mradapt( mr_epsilon, mr_regularity );

        myTimers.stop("InitMesh");

        samurai::save( "init_circle_"+std::to_string( ndomains)+"_domains", mesh );

        myTimers.start( "balance_Hilbert_cell" );
        SFC_LoadBalancer_interval<dim, Hilbert> loadb_hilbert;
        for(int lb_iter=0; lb_iter<nbIterLoadBalancing; ++lb_iter ){
            loadb_hilbert.load_balance( mesh );
            auto s_ = fmt::format("balance_hilbert_interval_{}_iter_{}", ndomains, lb_iter);
            samurai::save( s_ , mesh );
        }
        myTimers.stop( "balance_Hilbert_cell" );

        loadb_hilbert.evaluate_balancing( mesh );

        {
            Mesh_t newmesh( mesh[mesh_id_t::cells], mesh );
            auto u2 = init<dim>( newmesh, radius, x_center, y_center );
            
            samurai::save( "init2_circle_"+std::to_string( ndomains)+"_domains", newmesh, u2 );

            auto conv = samurai::make_convection<decltype(u2)>(velocity);

            // myTimers.start( "update_ghost_mr" );
            // for(int i=0; i<niterBench; ++i){
            //     samurai::update_ghost_mr(u2);
            // }           
            // myTimers.stop( "update_ghost_mr" );

            auto unp1 = samurai::make_field<double, 1>("unp1", newmesh);

            myTimers.start( "upwind" );
            for(int i=0; i<niterBench; ++i){
                unp1 = u2 - dt * conv( u2 );
            }
            myTimers.stop( "upwind" );

            
            // auto mradapt = samurai::make_MRAdapt( u2 );

            // myTimers.start( "mradapt" );
            // mradapt( mr_epsilon, mr_regularity );
            // myTimers.stop( "mradapt" );

        }

        myTimers.print();
    
    }


    samurai::finalize();

    return EXIT_SUCCESS;
}