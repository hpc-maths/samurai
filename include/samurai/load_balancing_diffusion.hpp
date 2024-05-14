#pragma once

#include <map>
#include "load_balancing.hpp"
#include <samurai/field.hpp>

// for std::sort
#include <algorithm>

namespace Load_balancing{

    class Diffusion : public samurai::LoadBalancer<Diffusion> {

        private:
            int _ndomains;
            int _rank;

            template<class Mesh_t, class Stencil, class Field_t>  
            void propagate( const Mesh_t & mesh, const Stencil & dir, Field_t & field, int value, int &given ) const {

            }

        public:

            Diffusion() {

    #ifdef SAMURAI_WITH_MPI
                boost::mpi::communicator world;
                _ndomains = world.size();
                _rank     = world.rank();
    #else
                _ndomains = 1;
                _rank     = 0;
    #endif
            }

            inline std::string getName() const { return "diffusion"; }

            template<class Mesh_t>
            Mesh_t load_balance_impl( Mesh_t & mesh ){

                using mpi_subdomain_t = typename Mesh_t::mpi_subdomain_t;
                using CellList_t      = typename Mesh_t::cl_type;
                using CellArray_t     = typename Mesh_t::ca_type;
                using mesh_id_t       = typename Mesh_t::mesh_id_t;

                using Coord_t = xt::xtensor_fixed<double, xt::xshape<Mesh_t::dim>>;
                using Stencil = xt::xtensor_fixed<int, xt::xshape<Mesh_t::dim>>;

                boost::mpi::communicator world;

                // For debug
                std::ofstream logs; 
                logs.open( fmt::format("log_{}.dat", world.rank()), std::ofstream::app );
                logs << fmt::format("> New load-balancing using {} ", getName() ) << std::endl;

                // compute fluxes in terms of number of intervals to transfer/receive
                std::vector<int> fluxes = samurai::cmptFluxes<samurai::BalanceElement_t::CELL>( mesh );
                std::vector<int> new_fluxes( fluxes );

                // get loads from everyone
                std::vector<int> loads;
                int my_load = static_cast<int>( samurai::cmptLoad<samurai::BalanceElement_t::CELL>( mesh ) );
                boost::mpi::all_gather( world, my_load, loads );

                std::vector<mpi_subdomain_t> & neighbourhood = mesh.mpi_neighbourhood();
                size_t n_neighbours = neighbourhood.size();

                { // some debug info
                    logs << "load : " << my_load << std::endl;
                    logs << "nneighbours : " << n_neighbours << std::endl;
                    logs << "neighbours : ";
                    for( size_t in=0; in<neighbourhood.size(); ++in )
                        logs << neighbourhood[ in ].rank << ", ";
                    logs << std::endl << "fluxes : ";
                    for( size_t in=0; in<neighbourhood.size(); ++in )
                        logs << fluxes[ in ] << ", ";
                    logs << std::endl;
                }

                std::vector<CellList_t> cl_to_send( n_neighbours );

                // set field "flags" for each rank. Initialized to current for all cells (leaves only)
                auto flags = samurai::make_field<int, 1>("diffusion_flag", mesh);
                flags.fill( world.rank() );

                // load balancing order
                std::vector<size_t> order( n_neighbours );
                {
                    for( size_t i=0; i<order.size(); ++i ){ order[ i ] = i; }

                    // order neighbour to echange data with, based on load
                    std::sort( order.begin(), order.end(), [&fluxes]( size_t i, size_t j){
                        return fluxes[ i ] < fluxes[ j ] ;
                    });

                }

                for( size_t neigh_i=0; neigh_i<n_neighbours; ++neigh_i ){

                    // neighbour [0, n_neighbours[
                    auto neighbour_local_id = order[ neigh_i ];

                    // all cells have been given, neighbours that might left are "givers" (remember the fluxes were sorted)
                    if( fluxes[ neighbour_local_id ] >= 0 ) break;

                    logs << fmt::format("\t> Working on neighbour # {}", neighbourhood[ neighbour_local_id ].rank ) << std::endl;

                    // compute initial interface with this neighbour
                    auto interface = samurai::cmptInterface<Mesh_t::dim, samurai::Direction_t::FACE>( mesh, neighbourhood[ neighbour_local_id ].mesh );

                    {
                        size_t nCellsAtInterfaceGiven = 0, nCellsAtInterface = 0;
                        samurai::for_each_interval( interface, [&]( std::size_t level, const auto & interval, const auto & index ){

                            for(size_t ii=0; ii<interval.size(); ++ii){
                                if( flags(level, interval, index)[ ii ] == world.rank() ){
                                    flags( level, interval, index )[ ii ] = neighbourhood[ neighbour_local_id ].rank;
                                    nCellsAtInterfaceGiven += 1;
                                }
                                nCellsAtInterface += 1;
                            }
                        });
                        logs << fmt::format("\t\t> NCellsAtInterface : {}, NCellsAtInterfaceGiven : {}", nCellsAtInterface, nCellsAtInterfaceGiven ) << std::endl;
                    }

                    // move the interface in the direction of "the center of mass" of the domain
                    // we basically want to move based on the normalized cartesian axis
                    //
                    // Q?: take into account already given intervals to compute BC ?
                    // (no weight here) 
                    Coord_t bc_current   = samurai::_cmpCellBarycenter<Mesh_t::dim>( mesh[ mesh_id_t::cells ] );
                    Coord_t bc_neighbour = samurai::_cmpCellBarycenter<Mesh_t::dim>( neighbourhood[ neighbour_local_id ].mesh[ mesh_id_t::cells ] );

                    // Compute normalized direction to neighbour, i.e. stencil
                    Stencil dir_from_neighbour;
                    {
                        Coord_t tmp;
                        double n2 = 0.;
                        for( size_t idim = 0; idim<Mesh_t::dim; ++idim ){
                            tmp( idim ) = bc_current( idim ) - bc_neighbour( idim );
                            n2 += tmp( idim ) * tmp( idim );
                        }

                        n2 = std::sqrt( n2 );

                        for( size_t idim = 0; idim<Mesh_t::dim; ++idim ){
                            tmp( idim ) /= n2;
                            dir_from_neighbour( idim ) = static_cast<int>( tmp( idim ) / 0.5 );
                        }

                        logs << fmt::format("\t\t> stencil for this neighbour # {} :", neighbourhood[ neighbour_local_id ].rank);
                        for(size_t idim=0; idim<Mesh_t::dim; ++idim ){
                            logs << dir_from_neighbour( idim ) << ",";
                        }
                        logs << std::endl;
                    }

                    // Avoid diagonals exchange, and emphaze x-axis

                    if constexpr ( Mesh_t::dim == 2 ) { 
                        if( std::abs(dir_from_neighbour[0]) == 1 && std::abs( dir_from_neighbour[1]) == 1 ){
                            dir_from_neighbour[1] = 0;
                        }
                    }

                    if constexpr ( Mesh_t::dim == 3 ) { 
                        if( std::abs(dir_from_neighbour[0]) == 1 && std::abs( dir_from_neighbour[1]) == 1 && std::abs(dir_from_neighbour[2]) == 1){
                            dir_from_neighbour[1] = 0;
                            dir_from_neighbour[2] = 0;
                        }
                    }

                    // propagate in direction
                    {
                        int nbInterStep = 1; // validate the while condition on starter

                        // let's suppose the interface is up-to-date based on "flags" already given
                        int offset = 1;
                        
                        logs << fmt::format("\t\t\t> Propagate for neighbour rank # {}", neighbourhood[ neighbour_local_id ].rank) << std::endl;

                        while( new_fluxes[ neighbour_local_id ] < 0 && nbInterStep != 0 ){
                            
                            int nbGiven = 0; 
                            CellList_t cl_given;

                            nbInterStep = 0;
                            for (size_t level = mesh.min_level(); level <= mesh.max_level(); ++level) {

                                std::size_t minlevel_check = static_cast<std::size_t>( std::max(static_cast<int>(mesh.min_level()), static_cast<int>(level) - 1 ) );
                                std::size_t maxlevel_check = std::min( mesh.max_level(), level + 1 );

                                size_t nCellsAtInterface = 0, nCellsAtInterfaceGiven = 0;                                
                                for (size_t proj_level = minlevel_check; proj_level <= maxlevel_check; ++proj_level){

                                    // translate interface in direction of center of current mesh
                                    auto set       = samurai::translate( interface[ level ], dir_from_neighbour );
                                    auto intersect = samurai::intersection( set, mesh[ mesh_id_t::cells ][ proj_level ]).on( proj_level ); // need handle level difference here !

                                    if ( proj_level > level) {
                                        auto set_  = samurai::translate( interface[ proj_level ], dir_from_neighbour );
                                        auto diff_ = samurai::difference( intersect, set_ );

                                        diff_( [&]( const auto & interval, const auto & index ) {
                                            for(size_t ii=0; ii<interval.size(); ++ii){
                                                if( flags( proj_level, interval, index)[ ii ] == world.rank() ){
                                                    flags( proj_level, interval, index )[ ii ] = neighbourhood[ neighbour_local_id ].rank;
                                                    nCellsAtInterfaceGiven += 1;

                                                    cl_given[ proj_level ][ index ].add_point( interval.start + ii );
                                                }
                                                nCellsAtInterface += 1;
                                            }
                                        });

                                    }else{
                                        intersect( [&]( const auto & interval, [[maybe_unused]] const auto & index ){
                                            
                                            nbInterStep += 1;

                                            for(size_t ii=0; ii<interval.size(); ++ii){
                                                if( flags( proj_level, interval, index )[ ii ] == world.rank() ){
                                                    flags( proj_level, interval, index )[ ii ] = neighbourhood[ neighbour_local_id ].rank;
                                                    nCellsAtInterfaceGiven += 1;

                                                    cl_given[ proj_level ][ index ].add_point( interval.start + ii );
                                                }
                                                nCellsAtInterface += 1;
                                            }

                                        });
                                    }

                                }

                                nbGiven += nCellsAtInterfaceGiven;

                                logs << fmt::format("\t\t\t\t> At level {}, NCellsAtInterface : {}, NCellsAtInterfaceGiven : {}, (offset:{})", level,
                                                    nCellsAtInterface, nCellsAtInterfaceGiven, offset ) << std::endl;

                            }

                            interface = { cl_given, false };

                            new_fluxes[ neighbour_local_id ] += nbGiven;
                        }
                        
                    }
                    
                    if( new_fluxes[ neighbour_local_id ] < 0 ){
                        std::cerr << fmt::format("\t> Error cannot fullfill the neighbour # {} ", neighbourhood[ neighbour_local_id ].rank ) << std::endl;
                    }

                }

                CellList_t new_cl;
                std::vector<CellList_t> payload( world.size() );

                samurai::for_each_cell( mesh[mesh_id_t::cells], [&]( const auto & cell ){
                
                    if( flags[ cell ] == world.rank() ){
                        if constexpr ( Mesh_t::dim == 1 ){ new_cl[ cell.level ][ {} ].add_point( cell.indices[ 0 ] ); }
                        if constexpr ( Mesh_t::dim == 2 ){ new_cl[ cell.level ][ { cell.indices[ 1 ] } ].add_point( cell.indices[ 0 ] ); }
                        if constexpr ( Mesh_t::dim == 3 ){ new_cl[ cell.level ][ { cell.indices[ 1 ], cell.indices[ 2 ] } ].add_point( cell.indices[ 0 ] ); }                        
                    }else{
                        if constexpr ( Mesh_t::dim == 1 ){ payload[ flags[ cell ] ][ cell.level ][ {} ].add_point( cell.indices[ 0 ] ); }
                        if constexpr ( Mesh_t::dim == 2 ){ payload[ flags[ cell ] ][ cell.level ][ { cell.indices[ 1 ] } ].add_point( cell.indices[ 0 ] ); }
                        if constexpr ( Mesh_t::dim == 3 ){ payload[ flags[ cell ] ][ cell.level ][ { cell.indices[ 1 ], cell.indices[ 2 ] } ].add_point( cell.indices[ 0 ] ); }
                    } 

                });

                /* ---------------------------------------------------------------------------------------------------------- */
                /* ------- Data transfer between processes ------------------------------------------------------------------ */ 
                /* ---------------------------------------------------------------------------------------------------------- */

                for( size_t neigh_i=0; neigh_i<n_neighbours; ++neigh_i ){

                    if( fluxes[ neigh_i ] == 0 ) continue;

                    if( fluxes[ neigh_i ] > 0 ){ // receiver
                        CellArray_t to_rcv;
                        world.recv( neighbourhood[ neigh_i ].rank, 42, to_rcv );

                        samurai::for_each_interval(to_rcv, [&](std::size_t level, const auto & interval, const auto & index ){
                            new_cl[ level ][ index ].add_interval( interval );
                        });

                    }else{ // sender
                        CellArray_t to_send = { payload[ neighbourhood[ neigh_i ].rank ], false };
                        world.send( neighbourhood[ neigh_i ].rank, 42, to_send );

                    }

                }

                /* ---------------------------------------------------------------------------------------------------------- */
                /* ------- Construct new mesh for current process ----------------------------------------------------------- */ 
                /* ---------------------------------------------------------------------------------------------------------- */

                Mesh_t new_mesh( new_cl, mesh );

                return new_mesh;
            }

    };
}