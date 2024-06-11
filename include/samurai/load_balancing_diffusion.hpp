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
            auto reordering_impl( Mesh_t & mesh ) {
                auto flags = samurai::make_field<int, 1>("diffusion_flag", mesh);
                flags.fill( _rank );

                return flags;
            }

            template<class Mesh_t>
            auto load_balance_impl( Mesh_t & mesh ){

                using mpi_subdomain_t = typename Mesh_t::mpi_subdomain_t;
                using CellList_t      = typename Mesh_t::cl_type;
                using mesh_id_t       = typename Mesh_t::mesh_id_t;

                using Coord_t = xt::xtensor_fixed<double, xt::xshape<Mesh_t::dim>>;
                using Stencil = xt::xtensor_fixed<int, xt::xshape<Mesh_t::dim>>;

                boost::mpi::communicator world;

                // For debug
                std::ofstream logs; 
                logs.open( fmt::format("log_{}.dat", world.rank()), std::ofstream::app );
                logs << fmt::format("> New load-balancing using {} ", getName() ) << std::endl;
                
                std::vector<mpi_subdomain_t> neighbourhood;

                std::vector<int> forceNeighbour;
                {
                    std::vector<mpi_subdomain_t> & neighbourhood_tmp = mesh.mpi_neighbourhood();

                    for( auto & neighbour : neighbourhood_tmp ){
                        auto interface = samurai::cmptInterface<Mesh_t::dim, samurai::Direction_t::FACE>( mesh, neighbour.mesh );
                        size_t nintervals = 0;
                        for_each_interval(interface, [&]( [[maybe_unused]] size_t level, [[maybe_unused]] const auto & i, [[maybe_unused]] const auto & ii ){
                            nintervals ++;
                        });
                        if( nintervals > 0 ){
                            forceNeighbour.emplace_back( neighbour.rank );
                            neighbourhood.emplace_back( neighbour );
                        }
                    }

                    logs << "Corrected neighbours : ";
                    for(const auto & fn : forceNeighbour ) 
                        logs << fn << ", ";
                    logs << std::endl;
                }

                size_t n_neighbours = neighbourhood.size();

                // compute fluxes in terms of number of intervals to transfer/receive
                // by default, perform 5 iterations
                std::vector<int> fluxes = samurai::cmptFluxes<samurai::BalanceElement_t::CELL>( mesh, forceNeighbour, 5 );

                std::vector<int> new_fluxes( fluxes );

                // get loads from everyone
                std::vector<int> loads;
                int my_load = static_cast<int>( samurai::cmptLoad<samurai::BalanceElement_t::CELL>( mesh ) );
                boost::mpi::all_gather( world, my_load, loads );

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

                    // move the interface in the direction of "the center of mass" of the domain
                    // we basically want to move based on the normalized cartesian axis
                    //
                    // Q?: take into account already given intervals to compute BC ?
                    // (no weight here) 
                    Coord_t bc_current   = samurai::_cmpCellBarycenter<Mesh_t::dim>( mesh[ mesh_id_t::cells ] );
                    Coord_t bc_neighbour = samurai::_cmpCellBarycenter<Mesh_t::dim>( neighbourhood[ neighbour_local_id ].mesh[ mesh_id_t::cells ] );

                    // Compute normalized direction to neighbour
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

                            // FIXME why needed ? 
                            if( std::abs( dir_from_neighbour( idim ) ) > 1 ) {
                                dir_from_neighbour( idim ) < 0 ? dir_from_neighbour( idim ) = -1 : dir_from_neighbour( idim ) = 1;
                            }
                        }
                        
                        // Avoid diagonals exchange, and emphaze x-axis. Maybe two phases propagation in case of diagonal ?
                        // i.e.: if (1, 1) -> (1, 0) then (0, 1) ?

                        if constexpr ( Mesh_t::dim == 2 ) { 
                            if( std::abs(dir_from_neighbour[0]) == 1 && std::abs( dir_from_neighbour[1]) == 1 ){
                                // dir_from_neighbour[0] = 1;
                                dir_from_neighbour[1] = 0;
                            }
                        }

                        if constexpr ( Mesh_t::dim == 3 ) { 
                            if( std::abs(dir_from_neighbour[0]) == 1 && std::abs( dir_from_neighbour[1]) == 1 && std::abs(dir_from_neighbour[2]) == 1){
                                dir_from_neighbour[1] = 0;
                                dir_from_neighbour[2] = 0;
                            }
                        }

                        logs << fmt::format("\t\t> (corrected) stencil for this neighbour # {} :", neighbourhood[ neighbour_local_id ].rank);
                        for(size_t idim=0; idim<Mesh_t::dim; ++idim ){
                            logs << dir_from_neighbour( idim ) << ",";
                        }
                        logs << std::endl;

                    }

                    // direction from neighbour domain to current domain
                    auto interface = samurai::cmptInterfaceUniform<Mesh_t::dim>( mesh, neighbourhood[ neighbour_local_id ].mesh, dir_from_neighbour );

                    bool empty = false;
                    {
                        size_t iii = 0;
                        samurai::for_each_interval( interface, [&]( [[maybe_unused]] size_t level, [[maybe_unused]] const auto & i, [[maybe_unused]] const auto & ii ){
                            iii ++;
                        });
                        if( iii == 0 ) empty = true;
                    }

                    if( empty ) {
                        logs << "\t> Skipping neighbour, empty interface ! " << std::endl;
                        continue;
                    }

                    {
                        size_t nCellsAtInterfaceGiven = 0, nCellsAtInterface = 0;
                        for (size_t level = mesh.min_level(); level <= mesh.max_level(); ++level) {
                            
                            auto intersect = samurai::intersection( interface[ interface.min_level() ], mesh[ mesh_id_t::cells ][ level ] ).on( level ); // need handle level difference here !
                            intersect( [&]( [[maybe_unused]] const auto & interval, [[maybe_unused]] const auto & index ){
                                for(size_t ii=0; ii<interval.size(); ++ii){
                                    if( flags( level, interval, index )[ ii ] == world.rank() )
                                    {
                                        flags( level, interval, index )[ ii ] = neighbourhood[ neighbour_local_id ].rank;
                                        nCellsAtInterfaceGiven += 1;
                                    }
                                    nCellsAtInterface += 1;
                                }

                            });

                        }

                        new_fluxes[ neighbour_local_id ] += nCellsAtInterfaceGiven;

                        logs << fmt::format("\t\t> NCellsAtInterface : {}, NCellsAtInterfaceGiven : {}", nCellsAtInterface, nCellsAtInterfaceGiven ) << std::endl;
                    }

                    // propagate until full-fill neighbour 
                    {
                        int nbElementGiven = 1; // validate the while condition on starter
                       
                        logs << fmt::format("\t\t\t> Propagate for neighbour rank # {}", neighbourhood[ neighbour_local_id ].rank) << std::endl;

                        int offset = 1;
                        while( new_fluxes[ neighbour_local_id ] < 0 && nbElementGiven > 0 ){

                            // intersection of interface with current mesh
                            size_t minLevelInInterface = mesh.max_level();

                            {
                                auto interface_on_mesh = samurai::translate( interface[ interface.min_level() ], dir_from_neighbour * offset ); // interface is monolevel !
                                for (size_t level = mesh.min_level(); level <= mesh.max_level(); ++level) {
                                    size_t nIntervalAtInterface = 0;
                                    auto intersect = samurai::intersection( interface_on_mesh, mesh[ mesh_id_t::cells ][ level ] ).on( level ); // need handle level difference here !
                                    intersect( [&]( [[maybe_unused]] const auto & interval, [[maybe_unused]] const auto & index ){
                                        nIntervalAtInterface += 1;
                                    });

                                    if( nIntervalAtInterface > 0 ) minLevelInInterface = std::min( minLevelInInterface, level );
                                }
                            }

                            if( minLevelInInterface != interface.min_level() ) { 
                                logs << "\t\t\t\t> [WARNING] Interface need to be update !" << std::endl;
                            }
                            
                            logs << fmt::format("\t\t\t\t> Min level in interface : {}", minLevelInInterface ) << std::endl;

                            // CellList_t cl_given;

                            nbElementGiven = 0;

                            auto interface_on_mesh = translate( interface[ interface.min_level() ], dir_from_neighbour * offset ); // interface is monolevel !
                            for (size_t level = mesh.min_level(); level <= mesh.max_level(); ++level) {

                                size_t nCellsAtInterface = 0, nCellsAtInterfaceGiven = 0;                                                                
                                auto intersect = intersection( interface_on_mesh, mesh[ Mesh_t::mesh_id_t::cells ][ level ] ).on( level ); // need handle level difference here !

                                intersect( [&]( const auto & interval, const auto & index ){

                                    for(size_t ii=0; ii<interval.size(); ++ii){
                                        if( flags( level, interval, index )[ ii ] == world.rank() )
                                        {
                                            flags( level, interval, index )[ ii ] = neighbourhood[ neighbour_local_id ].rank;
                                            nCellsAtInterfaceGiven += 1;

                                            // cl_given[ level ][ index ].add_point( interval.start + ii );
                                        }
                                        nCellsAtInterface += 1;
                                    }

                                });

                                logs << fmt::format("\t\t\t\t> At level {}, NCellsAtInterface : {}, NCellsAtInterfaceGiven : {}, nbElementGiven : {}", level,
                                                    nCellsAtInterface, nCellsAtInterfaceGiven, nbElementGiven ) << std::endl;

                                nbElementGiven += nCellsAtInterfaceGiven;

                            }

                            // interface = { cl_given, false };

                            new_fluxes[ neighbour_local_id ] += nbElementGiven;
                            offset ++;
                        }
                        
                    }
                    
                    if( new_fluxes[ neighbour_local_id ] < 0 ){
                        logs << fmt::format("\t> Error cannot fullfill the neighbour # {}, fluxes: {} ", neighbourhood[ neighbour_local_id ].rank, new_fluxes[ neighbour_local_id ] ) << std::endl;
                    }

                }

                return flags;
            }

    };
}