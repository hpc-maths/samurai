#pragma once

#include <map>
#include "load_balancing.hpp"

template<int dim>
class Diffusion_LoadBalancer_interval : public samurai::LoadBalancer<Diffusion_LoadBalancer_interval<dim>> {

    using Coord_t    = xt::xtensor_fixed<double, xt::xshape<dim>>;

    private:
        int _ndomains;
        int _rank;

        /**
         * Compute the barycenter of the current domain based on mid-center of intervals.
         * Weighted by level.
         * 
        */
        template<class Mesh_t>
        Coord_t _cmpIntervalBarycenter( Mesh_t & mesh ) { // compute barycenter of current domains ( here, all domains since with simulate multiple MPI domains)
            
            Coord_t bary;
            bary.fill( 0. );

            double wght_tot = 0.;
            samurai::for_each_interval( mesh, 
                                        [&]( std::size_t level, const auto& interval, const auto& index ) {
                
                // [OPTIMIZATION] precompute weight as array
                // double wght = 1. / ( 1 << ( mesh.max_level() - level ) );
                constexpr double wght = 1.;

                Coord_t mid = _getIntervalMidPoint( level, interval, index );

                for(int idim=0; idim<dim; ++idim ){
                    bary( idim ) += mid( idim ) * wght;
                }

                wght_tot += wght;

            });

            wght_tot = std::max( wght_tot, 1e-12 );
            for( int idim=0; idim<dim; ++idim ){
                bary( idim ) /= wght_tot;
            }

            return bary;

        }

        template<class Interval_t, class Index_t>
        inline Coord_t _getIntervalMidPoint( size_t level, const Interval_t & interval, 
                                             const Index_t & index ) const {
            Coord_t mid;
            double csize = samurai::cell_length( level );

            mid( 0 ) = ( (interval.end - interval.start) * 0.5 + interval.start ) * csize ;
            for( int idim=0; idim<dim-1; ++idim ){
                mid( idim + 1 ) = ( index( idim ) * csize ) + csize * 0.5;
            }

            return mid;

        }

    public:

        Diffusion_LoadBalancer_interval() {

#ifdef SAMURAI_WITH_MPI
            boost::mpi::communicator world;
            _ndomains = world.size();
            _rank     = world.rank();
#else
            _ndomains = 1;
            _rank     = 0;
#endif
        }

        inline std::string getName() const { return "Interface_Prop_LB"; } 

        template<class Mesh_t>
        Mesh_t load_balance_impl( Mesh_t & mesh ){

            using mpi_subdomain_t = typename Mesh_t::mpi_subdomain_t;
            using CellList_t      = typename Mesh_t::cl_type;
            using CellArray_t     = samurai::CellArray<dim>;
            using mesh_id_t       = typename Mesh_t::mesh_id_t;

            boost::mpi::communicator world;

            // For debug
            std::ofstream logs; 
            logs.open( "log_" + std::to_string( world.rank() ) + ".dat", std::ofstream::app );
            logs << "# New load balancing" << std::endl;

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
                auto interface = samurai::cmptInterface<dim, samurai::Direction_t::FACE>( mesh, neighbourhood[ requester ].mesh );

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

        template<class Mesh_t>
        [[deprecated]]
        void load_balance_impl2( Mesh_t & mesh ){

            using interval_t      = typename Mesh_t::interval_t;
            using mpi_subdomain_t = typename Mesh_t::mpi_subdomain_t;
            using CellList_t      = typename Mesh_t::cl_type;
            using CellArray_t     = samurai::CellArray<dim>;
            using mesh_id_t       = typename Mesh_t::mesh_id_t;

            boost::mpi::communicator world;

            std::ofstream logs;
            // DEBUG 
            logs.open( "log_" + std::to_string( world.rank() ) + ".dat", std::ofstream::app );
            logs << "# New load balancing" << std::endl;

            std::string smt = "# " + std::to_string( _rank ) + " [Diffusion_LoadBalancer_interval]::load_balance_impl ";
            SAMURAI_TRACE( smt + "Entering function ..." );

            // give access to rank & mesh of neighbour
            std::vector<mpi_subdomain_t> & neighbourhood = mesh.mpi_neighbourhood();
            size_t n_neighbours = neighbourhood.size();

            // get the load to neighbours (geometrical neighbour)
            std::vector<int> fluxes = samurai::cmptFluxes<samurai::BalanceElement_t::INTERVAL>( mesh );

            {
                logs << "load : " << samurai::cmptLoad<samurai::BalanceElement_t::INTERVAL>( mesh ) << std::endl;
                logs << "nneighbours : " << n_neighbours << std::endl;
                logs << "neighbours : ";
                for( size_t in=0; in<neighbourhood.size(); ++in )
                    logs << neighbourhood[ in ].rank << ", ";
                logs << std::endl << "fluxes : ";
                for( size_t in=0; in<neighbourhood.size(); ++in )
                    logs << fluxes[ in ] << ", ";
                logs << std::endl;
            }

            // pour échanger les intervalles, calcul des barycentres des cellules d'interfaces et pas le barycentre des
            // domaines en eux mêmes !

            // Interface for each neighbour as cell_array
            auto interface = samurai::_computeCartesianInterface<dim, samurai::Direction_t::FACE>( mesh );

            // compute some point of reference in mesh and interval-based interface
            // Coord_t barycenter = _cmpIntervalBarycenter( mesh[ mesh_id_t::cells ] );
            Coord_t barycenter = samurai::_cmpCellBarycenter<dim>( mesh[ mesh_id_t::cells ] );
            logs << "Domain barycenter : " << fmt::format( " barycenter : ({}, {})", barycenter(0), barycenter(1) ) << std::endl;

            std::vector<double> invloads;
            double my_load = static_cast<double>( samurai::cmptLoad<samurai::BalanceElement_t::INTERVAL>( mesh ) );
            boost::mpi::all_gather( world, my_load, invloads );
            for(size_t il=0; il<invloads.size(); ++il ){
                invloads[ il ] = 1. / invloads[ il ];
            }

            // std::vector<Coord_t> barycenter_interface_neighbours( n_neighbours );
            std::vector<Coord_t> barycenter_neighbours( n_neighbours );

            for(size_t nbi=0; nbi<n_neighbours; ++nbi ){
                // barycenter_interface_neighbours[ nbi ] = _cmpIntervalBarycenter( interface[ nbi ] );
                // barycenter_interface_neighbours[ nbi ] = _cmpCellBarycenter<dim>( interface[ nbi ] );
                barycenter_neighbours[ nbi ] = samurai::_cmpCellBarycenter<dim>( neighbourhood[ nbi ].mesh[ mesh_id_t::cells ] );

                // debug
                // auto s_ = fmt::format( "Barycenter neighbour : ({}, {})", 
                //                barycenter_interface_neighbours[ nbi ]( 0 ),
                //                barycenter_interface_neighbours[ nbi ]( 1 ) );

                // logs << s_ << std::endl;
            }

            struct Data {
                size_t level;
                interval_t interval;
                xt::xtensor_fixed<samurai::default_config::value_t, xt::xshape<dim - 1>> indices;
                int rank;
            };

            // build map of interval that needs to be sent
            std::multimap<double, Data> repartition;

            constexpr auto fdist = samurai::Distance_t::GRAVITY;

            for_each_interval( mesh[ Mesh_t::mesh_id_t::cells ],
                               [&]( std::size_t level, const auto& interval, const auto& index ){

                // cartesian coordinates
                auto mid_point = _getIntervalMidPoint( level, interval, index );

                // process that might get the interval
                int winner_id       = -1;
                // double winner_dist = std::numeric_limits<double>::max();
                double winner_dist = samurai::getDistance<dim, fdist>( mid_point, barycenter ) * invloads[ _rank ];

                // select the neighbour
                for( size_t ni=0; ni<n_neighbours; ++ni ){ // for each neighbour

                    auto neighbour_rank = neighbourhood[ ni ].rank;

                    if( fluxes[ ni ] >= 0 ) continue; // skip neighbour that will recv

                    // this might fix ilots but require neighbour update
                    // double dist = std::min( distance_inf<dim>( mid_point, barycenter_interface_neighbours[ ni ] ),
                    //                         distance_inf<dim>( mid_point, barycenter_neighbours[ ni ] ) );

                    double dist = samurai::getDistance<dim, fdist>( mid_point, barycenter_neighbours[ ni ] ) * invloads[ neighbour_rank ] ;
                    // double dist = samurai::distance_inf<dim>( mid_point, barycenter_interface_neighbours[ ni ] );

                    if( dist < winner_dist ){
                        winner_id   = ni;
                        winner_dist = dist;
                    }
                    
                }

                if( winner_id >= 0 ){

                    if( repartition.find( winner_dist ) != repartition.find( winner_dist ) ){
                        std::cerr << "\t> WARNING: Key conflict for std::map !" << std::endl;
                    }

                    repartition.insert ( std::pair<double, Data>( winner_dist, Data { level, interval, index, static_cast<int>( winner_id ) } ) );

                }

            });

            std::vector<CellList_t> cl_to_send( n_neighbours );
            std::vector<CellArray_t> ca_to_send( n_neighbours );

            // distribute intervals based on ordered distance to neighbours
            // rank is not the rank but the offset in the neighbourhood
            std::vector<int> given_ ( n_neighbours, 0 );
            // for( auto & it : repartition ){
            for( auto it = repartition.begin(); it != repartition.end(); it++ ){

                auto nrank = neighbourhood[ it->second.rank ].rank;
                auto lvl   = it->second.level;

                logs << fmt::format("\t> Interval {} --> to rank {} ( level : {} , dist : {}", it->second.interval, nrank, lvl, it->first) << std::endl;

                // shouldn't we give it to the second closest neighbour ?!
                if( given_[ it->second.rank ] + it->second.interval.size() <= ( - fluxes[ it->second.rank ] ) ){
                    cl_to_send[ it->second.rank ][ it->second.level ][ it->second.indices ].add_interval( it->second.interval );
                    given_[ it->second.rank ] += it->second.interval.size();
                }

            }

            for(size_t nbi=0; nbi<n_neighbours; ++nbi ) {
                ca_to_send[ nbi ] = { cl_to_send[ nbi ], false };
                logs << "\t\t> Number of cells to send to process # " << neighbourhood[ nbi ].rank
                          << " : " << ca_to_send[ nbi ].nb_cells() << std::endl;
            }

            // actual data transfer occurs here 
            for(size_t ni=0; ni<n_neighbours; ++ni ){
                
                if( fluxes [ ni ] == 0 ) continue; 

                if( fluxes[ ni ] > 0 ) { // receive data
                    samurai::CellArray<dim> to_rcv;
                    world.recv( neighbourhood[ ni ].rank, 42, to_rcv );
                    mesh.merge( to_rcv );
                }else{ // send data to 
                    world.send( neighbourhood[ ni ].rank, 42, ca_to_send[ ni ] );
                    mesh.remove( ca_to_send[ ni ] );
                }
            }

            // update neighbour mesh
            mesh.update_mesh_neighbour();
        }

};