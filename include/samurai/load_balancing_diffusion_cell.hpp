#pragma once

#include <map>
#include "load_balancing.hpp"

template<int dim>
class Diffusion_LoadBalancer_cell : public samurai::LoadBalancer<Diffusion_LoadBalancer_cell<dim>> {

    using Coord_t = xt::xtensor_fixed<double, xt::xshape<dim>>;

    private:
        int _ndomains;
        int _rank;

        template<class Mesh_t>
        double getSurfaceOrVolume( Mesh_t & mesh ) const {

            double s_ = 0.;
            for_each_cell( mesh[ Mesh_t::mesh_id_t::cells ], [&]( const auto& cell ){
                
                double s = 1.;

                for(int idim=0; idim<dim; ++idim){
                   s *= samurai::cell_length( cell.level );
                }

                s_ += s;
            });

            return s_;
        }

    public:

        Diffusion_LoadBalancer_cell() {
#ifdef SAMURAI_WITH_MPI
            boost::mpi::communicator world;
            _ndomains = world.size();
            _rank     = world.rank();
#else
            _ndomains = 1;
            _rank     = 0;
#endif
        }

        inline std::string getName() const { return "Gravity_LB"; } 

        template<class Mesh_t>
        Mesh_t load_balance_impl( Mesh_t & mesh ){

            using mpi_subdomain_t = typename Mesh_t::mpi_subdomain_t;
            using CellList_t      = typename Mesh_t::cl_type;
            using CellArray_t     = samurai::CellArray<dim>;
            using Cell_t          = typename Mesh_t::cell_t;
            using mesh_id_t       = typename Mesh_t::mesh_id_t;

            boost::mpi::communicator world;

            // For debug purpose
            std::ofstream logs;
            logs.open( "log_" + std::to_string( world.rank() ) + ".dat", std::ofstream::app );
            logs << "# New load balancing" << std::endl;

            // give access to rank & mesh of neighbour
            std::vector<mpi_subdomain_t> & neighbourhood = mesh.mpi_neighbourhood();
            
            std::size_t n_neighbours = neighbourhood.size();

            std::vector<double> loads;
            double my_load = static_cast<double>( samurai::cmptLoad<samurai::BalanceElement_t::CELL>( mesh ) );
            boost::mpi::all_gather( world, my_load, loads );

            // get the load to neighbours (geometrical neighbour)
            std::vector<int> fluxes = samurai::cmptFluxes<samurai::BalanceElement_t::CELL>( mesh );

            {
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

            // pour échanger les intervalles, calcul des barycentres des cellules d'interfaces et pas le barycentre des
            // domaines en eux mêmes !

            // Interface for each neighbour as cell_array
            auto interface = samurai::_computeCartesianInterface<dim, samurai::Direction_t::FACE_AND_DIAG>( mesh );

            std::vector<size_t> ncells_interface ( interface.size(), 0 );
            for(int ni=0; ni<interface.size(); ni++){
                CellArray_t _tmp = { interface[ ni ] };
                samurai::for_each_interval( _tmp, [&](std::size_t level, const auto & interval, const auto & index ){
                    ncells_interface[ ni ] ++;
                });
            }

            // compute some point of reference in mesh and interval-based interface
            // Coord_t barycenter = _cmpIntervalBarycenter( mesh[ mesh_id_t::cells ] );
            Coord_t barycenter = samurai::_cmpCellBarycenter<dim>( mesh[ mesh_id_t::cells ] );
            logs << "Domain barycenter : " << fmt::format( " barycenter : ({}, {})", barycenter(0), barycenter(1) ) << std::endl;

            // std::vector<Coord_t> barycenter_interface_neighbours( n_neighbours );
            std::vector<Coord_t> barycenter_neighbours( n_neighbours );
            std::vector<double> sv( n_neighbours );

            for(size_t nbi=0; nbi<n_neighbours; ++nbi ){
                // barycenter_interface_neighbours[ nbi ] = _cmpIntervalBarycenter( interface[ nbi ] );
                // barycenter_interface_neighbours[ nbi ] = _cmpCellBarycenter<dim>( interface[ nbi ] );
                barycenter_neighbours[ nbi ] = samurai::_cmpCellBarycenter<dim>( neighbourhood[ nbi ].mesh[ mesh_id_t::cells ] );
                
                sv[ nbi ] = getSurfaceOrVolume( neighbourhood[ nbi ].mesh );

                // debug
                auto s_ = fmt::format( "Barycenter neighbour # {}: ({}, {})", neighbourhood[ nbi ].rank,
                               barycenter_neighbours[ nbi ]( 0 ),
                               barycenter_neighbours[ nbi ]( 1 ) );

                logs << s_ << std::endl;
            }

            struct Data {
                Cell_t cell;
                int rank;
            };

            // build map of interval that needs to be sent. Warning, it does not work with classical std::map !!
            std::multimap<double, Data> repartition;

            constexpr auto fdist = samurai::Distance_t::GRAVITY;

            double currentSV = getSurfaceOrVolume( mesh );

            for_each_cell( mesh[ Mesh_t::mesh_id_t::cells ], [&]( const auto& cell ){

                auto cc = cell.center();

                // process that might get the interval
                int winner_id = -1;

                // double winner_dist = std::numeric_limits<double>::max();
                // double winner_dist = samurai::getDistance<dim, fdist>( cell, barycenter ) / loads[ world.rank() ];
                
                double coeff_current = currentSV / loads[ static_cast<size_t>( world.rank() ) ];
                double winner_dist = samurai::getDistance<dim, fdist>( cc, barycenter ) * coeff_current;

                std::vector<size_t> mload( world.size(), 0 );

                // select the neighbour
                for( std::size_t ni=0; ni<n_neighbours; ++ni ){ // for each neighbour

                    auto neighbour_rank = static_cast<std::size_t>( neighbourhood[ ni ].rank );

                    if( fluxes[ ni ] >= 0 ) continue; // skip neighbour that will recv

                    // this might fix ilots but require neighbour update
                    // double dist = std::min( distance_inf<dim>( mid_point, barycenter_interface_neighbours[ ni ] ),
                    //                         distance_inf<dim>( mid_point, barycenter_neighbours[ ni ] ) );

                    // double dist = samurai::getDistance<dim, fdist>( cell, barycenter_interface_neighbours[ ni ] ) / loads[ neighbour_rank ];
                    double coeff = sv[ ni ] / ( loads[ neighbour_rank ] + mload[ neighbour_rank ] ) ; // / sv[ ni ];
                    double dist = samurai::getDistance<dim, fdist>( cell.center(), barycenter_neighbours[ ni ] ) * coeff;

                    if( dist < winner_dist && ncells_interface[ ni ] > 0 ){
                        winner_id   = static_cast<int>( ni );
                        winner_dist = dist;

                        mload[ neighbour_rank ] += 1;
                    }
                    
                }

                if( winner_id >= 0 ){

                    if( repartition.find( winner_dist ) != repartition.find( winner_dist ) ){
                        std::cerr << "\t> WARNING: Key conflict for std::map !" << std::endl;
                    }

                    repartition.insert ( std::pair<double, Data>( winner_dist, Data { cell, static_cast<int>( winner_id ) } ) );

                }

            });


            std::vector<CellList_t> cl_to_send( n_neighbours );
            std::vector<CellArray_t> ca_to_send( n_neighbours );

            // distribute intervals based on ordered distance to neighbours
            // rank is not the rank but the offset in the neighbourhood
            std::vector<int> given_ ( n_neighbours, 0 );
            // for( auto & it : repartition ){
            for( auto it = repartition.begin(); it != repartition.end(); it++ ){

                std::size_t rank = static_cast<std::size_t>( it->second.rank );

                // shouldn't we give it to the second closest neighbour ?!
                if( given_[ rank ] + 1 <= ( - fluxes[ rank ] ) ){
                    
                    if constexpr ( dim == 3 ) {
                        auto i = it->second.cell.indices[ 0 ];
                        auto j = it->second.cell.indices[ 1 ];
                        auto k = it->second.cell.indices[ 2 ];
                        cl_to_send[ rank ][ it->second.cell.level ][ { j, k } ].add_point( i );
                    }else{
                        auto i = it->second.cell.indices[ 0 ];
                        auto j = it->second.cell.indices[ 1 ];
                        cl_to_send[ rank ][ it->second.cell.level ][ { j } ].add_point( i );
                    }

                    given_[ rank ] += 1;

                }

            }

            for(size_t nbi=0; nbi<n_neighbours; ++nbi ) {
                ca_to_send[ nbi ] = { cl_to_send[ nbi ], false };
                logs << "\t\t> Number of cells to send to process # " << neighbourhood[ nbi ].rank
                          << " : " << ca_to_send[ nbi ].nb_cells() << std::endl;
            }

            /* ---------------------------------------------------------------------------------------------------------- */
            /* ------- Data transfer between processes ------------------------------------------------------------------ */ 
            /* ---------------------------------------------------------------------------------------------------------- */

            CellList_t new_cl, need_remove;
            for(size_t ni=0; ni<n_neighbours; ++ni ){
                
                if( fluxes [ ni ] == 0 ) continue; 

                if( fluxes[ ni ] > 0 ) { // receive data
                    samurai::CellArray<dim> to_rcv;
                    world.recv( neighbourhood[ ni ].rank, 42, to_rcv );

                    logs << "Receiving data from # " << neighbourhood[ ni ].rank
                         << ", nbCells : " << to_rcv.nb_cells() << std::endl;

                    // old strategy: modifying the current mesh - not working, breaks some internals 
                    // mesh.merge( to_rcv );

                    // new strategy: build a whole new mesh from a cl_t
                    samurai::for_each_interval(to_rcv,
                            [&](std::size_t level, const auto& interval, const auto& index)
                            {
                                new_cl[ level ][ index ].add_interval( interval );
                            });

                }else{ // send data to 
                    world.send( neighbourhood[ ni ].rank, 42, ca_to_send[ ni ] );
                    
                    logs << "Sending data to # " << neighbourhood[ ni ].rank 
                         << ", nbCells : " << ca_to_send[ ni ].nb_cells() << std::endl;

                    // old strategy: modifying the current mesh - not working, breaks some internals 
                    // mesh.remove( ca_to_send[ ni ] );

                    samurai::for_each_interval(ca_to_send[ ni ],
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