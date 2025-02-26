#include <iostream>

// CLI
#include <CLI/CLI.hpp>

// samurai 
#include <samurai/samurai.hpp>
#include <samurai/timers.hpp>
#include <samurai/field.hpp>

#include <samurai/load_balancing.hpp>

#include <samurai/mr/mesh.hpp>
#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/assertLogTrace.hpp>

#include <cassert>
#include <algorithm>

template <class Mesh>
auto init(Mesh& mesh, const double radius, const double x_center, const double y_center )
{
    auto u = samurai::make_field<double, 1>("u", mesh);

    samurai::for_each_cell( mesh, [&](auto& cell) {
        auto center = cell.center();
        double rad  = ((center[0] - x_center) * (center[0] - x_center) + 
                       (center[1] - y_center) * (center[1] - y_center) );
        if ( rad <= radius * radius) {
            u[cell] = 1;
        } else {
            u[cell] = 0;
        }
    });

    return u;
}

template<int dim, class CellArray_t>
void detectInterface( std::vector<CellArray_t> & meshes, std::vector<samurai::MPI_Load_Balance> & graph ){

    std::cerr << "\t> [detectInterface] Number of mesh : " << meshes.size() << std::endl;
    
    for(int irank=0; irank<meshes.size(); ++irank ){

        std::cerr << "\t> Processing domain # " << irank << std::endl;

        for( int nbi=0; nbi<graph[ irank ].neighbour.size(); ++nbi ){

            auto neighbour_rank = graph[ irank ].neighbour[ nbi ];
            std::cerr << "\t\t> Neighbour process # " << neighbour_rank << std::endl;

            for( int idim=0; idim<2*dim; ++idim ){
                
                xt::xtensor_fixed<int, xt::xshape<dim>> stencil;
                stencil.fill( 0 );

                stencil[ idim / 2 ] = 1 - ( idim % 2 ) * 2;

                std::cerr << "\t\t\t> Stencil {" << stencil[ 0 ] << ", " << stencil[ 1 ] << "}" << std::endl;

                size_t nbCellsIntersection = 0, nbIntervalIntersection = 0;
                int minlevel = meshes[ neighbour_rank ].min_level();
                int maxlevel = meshes[ neighbour_rank ].max_level();

                for( int level=minlevel; level<=maxlevel; ++level ) {

                    // for each level look at level -1 / 0 / +1
                    int minlevel_proj = meshes[ irank ].min_level();
                    int maxlevel_proj = meshes[ irank ].max_level();
                    for(size_t projlevel=std::max(minlevel_proj, level-1); projlevel<=std::min(maxlevel_proj, level+1); ++projlevel ){
                        std::cerr << "\t\t\t> Looking intersection " << level << " onto level " << projlevel << std::endl;

                        // intersection avec myself
                        // translate neighbour from dir (hopefully to current)
                        auto set = translate( meshes[ neighbour_rank ][ level ], stencil );
                        auto intersect = intersection( set, meshes[ irank ][ projlevel ] ).on( projlevel );

                        size_t nbInter_ = 0, nbCells_ = 0;
                        intersect([&]( auto & i, auto & index ) {
                            nbInter_ += 1;
                            nbCells_ += i.size();
                        });

                        // we get more interval / cells, than wanted because neighbour has bigger cells
                        if( nbInter_ > 0 && projlevel > level ){
                            std::cerr << "\t\t\t> Too much cell selected  ... " << std::endl;
                            auto set_  = translate( meshes[ irank ][ projlevel ], stencil );
                            auto diff_ = difference( intersect, set_ );
                            
                            nbInter_ = 0;
                            nbCells_ = 0;
                            diff_([&]( auto & i, auto & index ) {
                                nbInter_ += 1;
                                nbCells_ += i.size();
                            });

                        }

                        std::cerr << "\t\t\t> nbInter_ : " << nbInter_ << ", nbCells_ " << nbCells_ << std::endl;

                        nbCellsIntersection    = nbCells_ ;
                        nbIntervalIntersection = nbInter_ ;

                    }

                }
                std::cerr << "\t\t> Total Ncells intersected " << nbCellsIntersection << ", nb interval : " << nbIntervalIntersection << std::endl;

            }

        }
    
    }

}

int main( int argc, char * argv[] ){

    samurai::initialize(argc, argv);
    Timers myTimers;

    constexpr int dim = 2;
    int ndomains = 1;
    std::size_t minLevel = 3, maxLevel = 8;
    double mr_regularity = 1.0, mr_epsilon = 2.e-4;
    xt::xtensor_fixed<double, xt::xshape<dim>> minCorner = {0., 0.};
    xt::xtensor_fixed<double, xt::xshape<dim>> maxCorner = {1., 1.};
    std::string filename = "ld_mesh_init";

    double radius = 0.2, x_center = 0.5, y_center = 0.5;

    CLI::App app{"Load balancing test"};
    app.add_option("--ndomains", ndomains, "Number of desired domains")->capture_default_str()->group("Simulation parameters");
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

    using Config = samurai::MRConfig<dim>;
    using Mesh_t = samurai::MRMesh<Config>;
    using CellList_t = typename Mesh_t::cl_type;
    using CellArray_t = typename Mesh_t::ca_type;

    /**
     * Initialize AMR mesh with spherical buble 
    */
    myTimers.start("InitMesh");
    // samurai::Box<double, dim> box( minCorner, maxCorner );

    // Mesh_t mesh(box, minLevel, maxLevel);

    // auto u = init( mesh, radius, x_center, y_center );
    // samurai::make_bc<samurai::Dirichlet>(u, 0.);
    // auto mradapt = samurai::make_MRAdapt( u );
    // mradapt( mr_epsilon, mr_regularity );

    std::vector<CellList_t> clists( 3 );
    std::vector<CellArray_t> carrays( 3 );

    clists[0][1][{0}].add_interval({1, 4});
    clists[0][1][{1}].add_interval({1, 4});
    clists[0][1][{2}].add_interval({0, 4});
    clists[0][1][{3}].add_interval({0, 4});
    clists[0][2][{0}].add_interval({0, 2});
    clists[0][2][{1}].add_interval({0, 2});
    clists[0][2][{2}].add_interval({0, 2});
    clists[0][2][{3}].add_interval({0, 2});

    // subdomain 2
    clists[1][0][{2}].add_interval({0, 4});
    clists[1][0][{3}].add_interval({0, 4});

    // subdomain 3
    clists[2][0][{0}].add_interval({2, 4});
    clists[2][0][{1}].add_interval({2, 4});

    carrays[0] = { clists[0], true };
    carrays[1] = { clists[1], true };
    carrays[2] = { clists[2], true };

    myTimers.stop("InitMesh");

    std::vector<samurai::MPI_Load_Balance> graph( 3 );
    { // fill graph info
        
        // find a way to compute this 
        graph[ 0 ].neighbour = {1, 2};
        graph[ 1 ].neighbour = {0, 2};
        graph[ 2 ].neighbour = {0, 1};

        // fill "current" load  ----> no MPI comm, local info
        for(size_t i=0; i<3; ++i ){
            graph[ i ]._load = carrays[ i ].nb_cells();
        }

        // fill neighbour load -----> this will imply MPI comm
        for(size_t i=0; i<3; ++i ){      
            graph[ i ].load.resize( graph[ i ].neighbour.size() );
            for(size_t ni=0; ni<graph[ i ].neighbour.size(); ++ni ){
                auto neighbour_rank = graph[ i ].neighbour[ ni ];
                graph[ i ].load[ ni ] = graph[ neighbour_rank ]._load ;
            }
        }
    }

    CellList_t global;
    for( int level=0; level<3; ++level){
        auto un = samurai::union_( carrays[0][level], carrays[1][level], carrays[2][level] );

        un([&]( auto & i, auto & index ) {
            global[ level ][ index ].add_interval( i );
        });
    }


    detectInterface<dim>( carrays, graph );


    CellArray_t mesh = { global, true };

    auto init_rank = samurai::make_field<int, 1>( "init_rank", mesh);

    { // initialize fake_rank
        
        for(size_t m_=0; m_<carrays.size(); ++m_ ){

            for( int level=std::min( mesh.min_level(), carrays[ m_ ].min_level()); level<4; ++level ) {
                auto intersect = intersection( mesh[ level ], carrays[ m_ ][ level ]);

                intersect([&]( auto & i, auto & index ) {
                    init_rank( level, i, index ) = static_cast<int>( m_ );
                });
            }
        }

    }

    // output mesh 
    SAMURAI_LOG( "\t> Initial configurtion : " + filename );
    samurai::save( filename, mesh, init_rank );

    myTimers.start( "load_balance_fluxes" );
    compute_load_balancing_fluxes( graph );
    myTimers.start( "load_balance_fluxes" );

    // explicitly call load_balancing();

    // std::cerr << "\n\n" << std::endl;
    // SAMURAI_LOG( "[partition] Explicitly calling load_balancing() ... ");

    // for(int il=0; il<3; ++il ){
    //     myTimers.start("load-balancing");
    //     mesh.force_load_balancing();
    //     myTimers.stop("load-balancing");

    //     // output mesh 
    //     std::cerr << "\t> Dumping mesh to file : " << "load-balanced" << std::endl;
    //     samurai::save( "load-balanced_"+std::to_string(il), mesh, u );
    // }

    auto sfc_morton_rank = samurai::make_field<int, 1>( "sfc_morton_rank", mesh);

    myTimers.start( "balance_morton" );
    // perform_load_balancing_SFC<dim>( mesh, ndomains, sfc_morton_rank );
    myTimers.stop( "balance_morton" );


    auto diffusion_rank = samurai::make_field<int, 1>( "diffusion_rank", mesh);

    { // initialize fake_rank
        
        for(size_t m_=0; m_<carrays.size(); ++m_ ){

            for( int level=std::min( mesh.min_level(), carrays[ m_ ].min_level()); level<4; ++level ) {
                auto intersect = intersection( mesh[ level ], carrays[ m_ ][ level ]);

                intersect([&]( auto & i, auto & index ) {
                    diffusion_rank( level, i, index ) = static_cast<int>( m_ );
                });
            }
        }

    }

    samurai::save( "load_balanced_diff_init_"+std::to_string( ndomains), mesh, sfc_morton_rank, diffusion_rank );

    for_each_cell( mesh, [&](const auto & cell ){
        diffusion_rank[ cell ] = -1;
    });

    myTimers.start( "balance_diffusion" );
    perform_load_balancing_diffusion<dim>( mesh, carrays, ndomains, graph, diffusion_rank );
    myTimers.stop( "balance_diffusion" );

    filename = "load_balanced_diff_" + std::to_string( ndomains);
    SAMURAI_LOG( "\t> Generated file : " + filename );
    samurai::save( filename, mesh, sfc_morton_rank, diffusion_rank );
    

    myTimers.print();
    samurai::finalize();

    return EXIT_SUCCESS;
}