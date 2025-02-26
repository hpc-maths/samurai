#include <iostream>

// CLI
#include <CLI/CLI.hpp>

// samurai 
#include <samurai/samurai.hpp>
#include <samurai/timers.hpp>
#include <samurai/field.hpp>
#include <samurai/algorithm/update.hpp>

#include <samurai/load_balancing.hpp>
#include <samurai/load_balancing_sfc.hpp>
#include <samurai/load_balancing_diffusion_interval.hpp>
#include <samurai/load_balancing_diffusion_cell.hpp>
#include <samurai/load_balancing_void.hpp>

#include <samurai/mr/mesh.hpp>
#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/assertLogTrace.hpp>
#include <samurai/schemes/fv.hpp>

#include <cassert>
#include <algorithm>

#include <vector>

template <class Mesh>
auto getLevel(Mesh& mesh)
{
    auto level = samurai::make_field<int, 1>("level", mesh);

    samurai::for_each_cell( mesh, [&]( auto & cell ) {
        level[ cell ] = static_cast<int>( cell.level );
    });

    return level;
}

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
        u[ cell ] = 0.;
    });

    samurai::for_each_cell( mesh, [&](auto& cell) {
        auto cc = cell.center();

        for(size_t icerc=0; icerc<centers.size(); ++icerc ){
            double rad  = 0.;
            
            for( size_t idim=0; idim<dim; ++idim ){
                rad += (cc[ idim ] - centers[ icerc ][ idim ]) * (cc[ idim ] - centers[ icerc ][ idim ]);
            }

            if ( rad <= radius[ icerc ] * radius[ icerc ]) {
                u[ cell ] += 1.;
            } else {
                u[ cell ] += 0.;
            }
        }

    });

    return u;
}

template<class Mesh_t, class Field_t, class Conv_t>
void upWind(int niter_benchmark, const Mesh_t & mesh, Field_t & u2, Field_t & unp1, double dt, Conv_t & conv) {

    for(int i=0; i<niter_benchmark; ++i){
        auto c = conv(u2);
        samurai::for_each_interval( mesh, [&]( [[maybe_unused]] std::size_t level, const auto& i, [[maybe_unused]] const auto & index ){
            for(int64_t ii=i.index+i.start; ii<i.index+i.end; ++ii){
                unp1[ ii ] = u2[ ii ] - dt * c[ii];
            }
        }
        );
        // unp1 = u2 - dt * 2; // conv( u2 );
    }
}

template<int dim>
struct Config_test_load_balancing {
    int niter_benchmark   = 1;
    int niter_loadbalance = 1; // number of iteration of load balancing
    int ndomains          = 1; // number of partition (mpi processes)
    int rank              = 0; // current mpi rank
    std::size_t minlevel  = 6;
    std::size_t maxlevel  = 8;
    double dt             = 0.1;
    double mr_regularity  = 1.0;
    double mr_epsilon     = 2.e-4;
    xt::xtensor_fixed<double, xt::xshape<dim>> minCorner = {0., 0., 0.};
    xt::xtensor_fixed<double, xt::xshape<dim>> maxCorner = {1., 1., 1.};
    std::vector<double> bubles_r;                 // bubles radius
    std::vector<std::array<double, 3>> bubles_c;  // bubles centers
};

template<int dim, class Load_Balancer_t, typename Vel_t>
Timers benchmark_loadbalancing( struct Config_test_load_balancing<dim> & conf, Load_Balancer_t && lb, Vel_t & velocity ){
    
    using Config    = samurai::MRConfig<dim>;
    using Mesh_t    = samurai::MRMesh<Config>;
    using mesh_id_t = typename Mesh_t::mesh_id_t;

    Timers myTimers;

    /**
     * Initialize AMR mesh with spherical buble 
    */

    samurai::Box<double, dim> box( conf.minCorner, conf.maxCorner );

    Mesh_t mesh( box, conf.minlevel, conf.maxlevel );

    // auto u = init<dim>( mesh, radius, x_center, y_center );
    myTimers.start("InitMesh");
    auto u = initMultiCircles<dim>( mesh, conf.bubles_r, conf.bubles_c );
    myTimers.stop("InitMesh");

    auto lvl = getLevel( mesh );

    samurai::make_bc<samurai::Dirichlet<1>>( u, 0. );

    myTimers.start("mradapt");
    auto mradapt = samurai::make_MRAdapt( u );
    mradapt( conf.mr_epsilon, conf.mr_regularity );
    myTimers.stop("mradapt");

    // samurai::save( "init_circle_"+std::to_string( ndomains)+"_domains", mesh, lvl );

    myTimers.start( lb.getName() );
    for(int lb_iter=0; lb_iter<conf.niter_loadbalance; ++lb_iter ){
        lb.load_balance( mesh );
        auto s_ = fmt::format( "{}_{}_iter_{}", lb.getName(), conf.ndomains, lb_iter);
        samurai::save( s_ , mesh );
    }
    myTimers.stop( lb.getName() );

    std::string _stats = fmt::format( "stats_{}_process_{}", lb.getName(), conf.rank );
    samurai::Statistics s ( _stats );
    s( "stats", mesh );

    {
        Mesh_t newmesh( mesh[mesh_id_t::cells], mesh );
        // auto u2 = init<dim>( newmesh, radius, x_center, y_center );
        auto u2 = initMultiCircles<dim>( newmesh, conf.bubles_r, conf.bubles_c );
        samurai::save( "init2_circle_"+std::to_string( conf.ndomains )+"_domains", newmesh, u2 );

        auto conv = samurai::make_convection_upwind<decltype(u2)>( velocity );
        auto unp1 = samurai::make_field<double, 1>("unp1", newmesh);

        // myTimers.start( "update_ghost_mr_bf" );
        // for(int i=0; i<conf.niter_benchmark; ++i){
        //     samurai::update_ghost_mr( u2 );
        // }           
        // myTimers.stop( "update_ghost_mr_bf" );

        myTimers.start( "upwind" );
        for(int i=0; i<conf.niter_benchmark; ++i){
            unp1 = u2 - conf.dt *  conv( u2 );
        }
        // upWind( conf.niter_benchmark, mesh, u2, unp1, conf.dt, conv );
        myTimers.stop( "upwind" );

        myTimers.start( "update_ghost_mr" );
        for(int i=0; i<conf.niter_benchmark; ++i){
            samurai::update_ghost_mr( u2 );
        }           
        myTimers.stop( "update_ghost_mr" );
        
        // auto mradapt = samurai::make_MRAdapt( u2 );

        // myTimers.start( "mradapt" );
        // mradapt( conf.mr_epsilon, conf.mr_regularity );
        // myTimers.stop( "mradapt" );

    }

    return myTimers;
}

int main( int argc, char * argv[] ){

    samurai::initialize(argc, argv);

    constexpr int    dim = 2;
    constexpr double cfl = 0.5;

    struct Config_test_load_balancing<dim> conf;

    std::string filename = "ld_mesh_init";

    double radius = 0.2, x_center = 0.5, y_center = 0.5;
    bool multi = false;

    CLI::App app{"Load balancing test"};
    app.add_option("--iter-loadbalance", conf.niter_loadbalance, "Number of desired lb iteration")
                   ->capture_default_str()->group("Simulation parameters");
    app.add_option("--iter-bench", conf.niter_benchmark, "Number of iteration for bench")->capture_default_str()->group("Simulation parameters");
    app.add_option("--min-level", conf.minlevel, "Minimum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--max-level", conf.maxlevel, "Maximum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--mr-eps", conf.mr_epsilon, "The epsilon used by the multiresolution to adapt the mesh")
        ->capture_default_str()->group("Multiresolution");
    app.add_option("--mr-reg", conf.mr_regularity,"The regularity criteria used by the multiresolution to "
                   "adapt the mesh")->capture_default_str()->group("Multiresolution");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Ouput");
    app.add_option("--radius", radius, "Bubble radius")->capture_default_str()->group("Simulation parameters");
    app.add_option("--xcenter", x_center, "Bubble x-axis center")->capture_default_str()->group("Simulation parameters");
    app.add_option("--ycenter", y_center, "Bubble y-axis center")->capture_default_str()->group("Simulation parameters");
    app.add_flag("--multi", multi, "Multiple bubles")->group("Simulation parameters");
    
    CLI11_PARSE(app, argc, argv);

    conf.dt = cfl / ( 1 << conf.maxlevel );

    // Convection operator
    samurai::VelocityVector<dim> velocity;
    velocity.fill( 1 );
    velocity( 1 ) = -1;

    boost::mpi::communicator world;
    conf.rank     = world.rank();
    conf.ndomains = static_cast<int>( world.size() );

    if( multi ) {
        conf.bubles_r = { 0.2, 0.1, 0.05};
        conf.bubles_c = {{ 0.2, 0.2, 0.2 }, { 0.5, 0.5, 0.5 }, { 0.8, 0.8, 0.8 }};
    }else{
        conf.bubles_r = { radius };
        conf.bubles_c = {{ x_center, y_center, 0.5 }};
    }

    if( conf.rank == 0 ) std::cerr << "\n\t> Testing 'Interval propagation load balancer' " << std::endl;

    if( conf.ndomains > 1 ) { // load balancing cells by cells using interface propagation
        auto times = benchmark_loadbalancing( conf, Diffusion_LoadBalancer_interval<dim> (), velocity );
        times.print();
        world.barrier();
    }

    if( conf.rank == 0 ) std::cerr << "\n\t> Testing 'Gravity-based cell exchange' load balancer' " << std::endl;

    if( conf.ndomains > 1 ) { // load balancing cells by cells using gravity
        auto times = benchmark_loadbalancing( conf, Diffusion_LoadBalancer_cell<dim> (), velocity );
        times.print();
        world.barrier();
    }

    if( conf.rank == 0 ) std::cerr << "\n\t> Testing Morton_LoadBalancer_interval " << std::endl;

    if( conf.ndomains > 1 ) { // load balancing using SFC morton
        auto times = benchmark_loadbalancing( conf, SFC_LoadBalancer_interval<dim, Morton> (), velocity );
        times.print();
        world.barrier();
    }

    if( conf.rank == 0 ) std::cerr << "\n\t> Testing Hilbert_LoadBalancer_interval " << std::endl;

    if( conf.ndomains > 1 ) { // load balancing using SFC hilbert
        auto times = benchmark_loadbalancing( conf, SFC_LoadBalancer_interval<dim, Hilbert> (), velocity );
        times.print();
        world.barrier();
    }

     if( conf.rank == 0 ) std::cerr << "\n\t> Testing no loadbalancing (initial partitionning) " << std::endl;

    { 
        auto times = benchmark_loadbalancing( conf, Void_LoadBalancer<dim> (), velocity );
        times.print();
        world.barrier();
    }

    samurai::finalize();

    return 0;
}