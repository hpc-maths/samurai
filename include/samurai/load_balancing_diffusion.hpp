#pragma once

#include <map>
#include "load_balancing.hpp"
#include <samurai/field.hpp>

namespace load_balancing{

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
            Mesh_t load_balance_impl( Mesh_t & mesh ){

                using mpi_subdomain_t = typename Mesh_t::mpi_subdomain_t;
                using CellList_t      = typename Mesh_t::cl_type;
                using CellArray_t     = typename Mesh_t::ca_type;
                using mesh_id_t       = typename Mesh_t::mesh_id_t;

                using Coord_t    = xt::xtensor_fixed<double, xt::xshape<Mesh_t::dim>>;

                constexpr int dim = static_cast<int>( Mesh_t::dim );

                boost::mpi::communicator world;

                // For debug
                std::ofstream logs; 
                logs.open( fmt::format("log_{}.dat", world.rank()), std::ofstream::app );
                logs << fmt::format("> New load-balancing using {} ", getName() ) << std::endl;

                // compute fluxes in terms of number of intervals to transfer/receive
                std::vector<int> fluxes = samurai::cmptFluxes<samurai::BalanceElement_t::INTERVAL>( mesh );
                std::vector<int> new_fluxes( fluxes );

                // get loads from everyone
                std::vector<int> loads;
                int my_load = static_cast<int>( samurai::cmptLoad<samurai::BalanceElement_t::INTERVAL>( mesh ) );
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

                auto flags = samurai::make_field<int, 1>("diffusion_flag", mesh);
                for_each_interval( mesh[ mesh_id_t::cells ], [&]( std::size_t level, const auto & interval, const auto & index ){
                    flags( level, interval, index ) = _rank;
                });

                

                bool balancing_done = false;
                while( ! balancing_done ){
                
                    // select neighbour with the highest needs of load
                    bool neighbour_found = false;
                    std::size_t requester   = 0;
                    int requested_load = 0;
                    for(std::size_t nbi=0; nbi<n_neighbours; ++nbi ){

                        // skip neighbour that need to send load
                        if( new_fluxes[ nbi ] >= 0 ) continue;

                        // FIX: Add condition (&& interface not empty ) ?
                        // Neighbourhood should have been updated in a way that an interface exists !
                        if( - new_fluxes[ nbi ] > requested_load ){
                            requested_load  = - new_fluxes[ nbi ];
                            requester       = nbi;
                            neighbour_found = true;
                        }
                    }

                    if( ! neighbour_found ){
                        { // debug 
                            logs << "No more neighbour found " << std::endl;
                        }

                        balancing_done = true;
                        break;
                    }

                    { // debug 
                        logs << "Requester neighbour : " << neighbourhood[ requester ].rank << ", fluxes : " << requested_load << std::endl;
                    }

                    // select interval for this neighbour by moving in cartesian direction by one 
                    auto interface = samurai::cmptInterface<Mesh_t::dim, samurai::Direction_t::FACE>( mesh, neighbourhood[ requester ].mesh );

                    { // check emptyness of interface, if it is empty, then set fluxes for this neighbour to 0
                        size_t nelement = 0;
                        samurai::for_each_interval( interface, [&]( [[maybe_unused]] std::size_t level, [[maybe_unused]] const auto & interval, 
                                                                    [[maybe_unused]] const auto & index ){
                            nelement += 1;
                        });

                        if( nelement == 0 ){
                            new_fluxes[ requester ] = 0;

                            { // debug
                                std::cerr << fmt::format("\t> Process {}, Warning no interface with a requester neighbour # {}", world.rank(),
                                                        neighbourhood[ requester ].rank ) << std::endl;
                                logs << "Requester neighbour, no interface found, set fluxes to " << new_fluxes[ requester ] << std::endl;
                            }
                        }else{
                            { // debug 
                                logs << "Requester neighbour, interface  " << nelement << " intervals " << std::endl;
                            }
                        }
                    }

                    // go through interval on the interface and add as much as possible
                    // skip this to add the whole interface
                    CellList_t cl_for_neighbour;

                    size_t nbIntervalAdded = 0;
                    samurai::for_each_interval( interface, [&]( std::size_t level, const auto & interval, const auto & index ){

                        if( new_fluxes[ requester ] < 0 ){
                            cl_for_neighbour[ level ][ index ].add_interval( interval );
                            // new_fluxes[ requester ] += 1; // here interval load == 1, no weight;
                            nbIntervalAdded += 1;
                        }

                    });

                    new_fluxes[ requester ] += nbIntervalAdded;

                    { // remove interval from current process mesh and add it to neighbour mesh local copy !
                        CellArray_t ca_for_neighbour = { cl_for_neighbour, false };
                        
                        // mesh.remove( ca_for_neighbour );
                        // neighbourhood[ requester ].mesh.merge( ca_for_neighbour );

                        // update gobal list that will be sent to neighbour process
                        samurai::for_each_interval( ca_for_neighbour, [&]( std::size_t level, const auto & interval, const auto & index ){
                            cl_to_send[ requester ][ level ][ index ].add_interval( interval );
                        });
                    }

                    { // debug 
                        logs << "New flux for this neighbour : " << new_fluxes[ requester ] << std::endl;
                    }

                }

                /* ---------------------------------------------------------------------------------------------------------- */
                /* ------- Data transfer between processes ------------------------------------------------------------------ */ 
                /* ---------------------------------------------------------------------------------------------------------- */
                
                CellList_t new_cl, need_remove;

                // at this point local mesh of neighbour are modified ( technically it should match what would results from send)
                // and local mesh has been modified
                for(std::size_t ni=0; ni<n_neighbours; ++ni ){
                    
                    if( fluxes [ ni ] == 0 ) continue; 

                    if( fluxes[ ni ] > 0 ) { // receive data
                        CellArray_t to_rcv;
                        world.recv( neighbourhood[ ni ].rank, 42, to_rcv );

                        logs << fmt::format("\t>[load_balance_impl]::interval Rank # {} receiving {} cells from rank # {}", world.rank(), to_rcv.nb_cells(), neighbourhood[ ni ].rank) << std::endl;
                        
                        // old strategy
                        // mesh.merge( to_rcv );

                        // add to future mesh of current process
                        samurai::for_each_interval(to_rcv,
                                [&](std::size_t level, const auto& interval, const auto& index)
                                {
                                    new_cl[ level ][ index ].add_interval( interval );
                                });

                    }else{ // send data to
                        CellArray_t to_send = { cl_to_send[ ni ], false };
                        world.send( neighbourhood[ ni ].rank, 42, to_send );

                        logs << fmt::format("\t>[load_balance_impl]::interval Rank # {} sending {} cells to rank # {}", world.rank(), to_send.nb_cells(), neighbourhood[ ni ].rank) << std::endl;

                        samurai::for_each_interval( to_send,
                                [&](std::size_t level, const auto& interval, const auto& index)
                                {
                                    need_remove[ level ][ index ].add_interval( interval );
                                });
                    }

                }

                /* ---------------------------------------------------------------------------------------------------------- */
                /* ------- Construct new mesh for current process ----------------------------------------------------------- */ 
                /* ---------------------------------------------------------------------------------------------------------- */

                // add to new_cl interval that were not sent
                CellArray_t need_remove_ca = { need_remove }; // to optimize
                for( size_t level=mesh.min_level(); level<=mesh.max_level(); ++level ){
                    auto diff = samurai::difference( mesh[ mesh_id_t::cells ][ level ], need_remove_ca[ level ] );

                    diff([&]( auto & interval, auto & index ){
                        new_cl[ level ][ index ].add_interval( interval );
                    });
                }

                Mesh_t new_mesh( new_cl, mesh );

                return new_mesh;
            }

    };
}