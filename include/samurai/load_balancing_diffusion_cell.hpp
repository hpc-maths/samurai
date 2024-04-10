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
        void load_balance_impl( Mesh_t & mesh ){

            using mpi_subdomain_t = typename Mesh_t::mpi_subdomain_t;
            using CellList_t      = typename Mesh_t::cl_type;
            using CellArray_t     = samurai::CellArray<dim>;
            using Cell_t          = typename Mesh_t::cell_t;
            using mesh_id_t       = typename Mesh_t::mesh_id_t;
            using Coord_t         = xt::xtensor_fixed<double, xt::xshape<dim>>;

            boost::mpi::communicator world;

            // For debug purpose
            std::ofstream logs;
            logs.open( "log_" + std::to_string( world.rank() ) + ".dat", std::ofstream::app );
            logs << "# New load balancing" << std::endl;

            // give access to rank & mesh of neighbour
            std::vector<mpi_subdomain_t> & neighbourhood = mesh.mpi_neighbourhood();
            
            size_t n_neighbours = neighbourhood.size();

            std::vector<double> loads;
            double my_load = static_cast<double>( cmptLoad<samurai::BalanceElement_t::CELL>( mesh ) );
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
            auto interface = _computeCartesianInterface<dim, samurai::Direction_t::FACE_AND_DIAG>( mesh );

            // compute some point of reference in mesh and interval-based interface
            // Coord_t barycenter = _cmpIntervalBarycenter( mesh[ mesh_id_t::cells ] );
            Coord_t barycenter = _cmpCellBarycenter<dim>( mesh[ mesh_id_t::cells ] );
            logs << "Domain barycenter : " << fmt::format( " barycenter : ({}, {})", barycenter(0), barycenter(1) ) << std::endl;

            // std::vector<Coord_t> barycenter_interface_neighbours( n_neighbours );
            std::vector<Coord_t> barycenter_neighbours( n_neighbours );
            std::vector<double> sv( n_neighbours );

            for(size_t nbi=0; nbi<n_neighbours; ++nbi ){
                // barycenter_interface_neighbours[ nbi ] = _cmpIntervalBarycenter( interface[ nbi ] );
                // barycenter_interface_neighbours[ nbi ] = _cmpCellBarycenter<dim>( interface[ nbi ] );
                barycenter_neighbours[ nbi ] = _cmpCellBarycenter<dim>( neighbourhood[ nbi ].mesh[ mesh_id_t::cells ] );
                
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
                
                double coeff_current = currentSV / loads[ world.rank() ];
                double winner_dist = samurai::getDistance<dim, fdist>( cc, barycenter ) * coeff_current;

                // select the neighbour
                for( size_t ni=0; ni<n_neighbours; ++ni ){ // for each neighbour

                    auto neighbour_rank = neighbourhood[ ni ].rank;

                    if( fluxes[ ni ] >= 0 ) continue; // skip neighbour that will recv

                    // this might fix ilots but require neighbour update
                    // double dist = std::min( distance_inf<dim>( mid_point, barycenter_interface_neighbours[ ni ] ),
                    //                         distance_inf<dim>( mid_point, barycenter_neighbours[ ni ] ) );

                    // double dist = samurai::getDistance<dim, fdist>( cell, barycenter_interface_neighbours[ ni ] ) / loads[ neighbour_rank ];
                    double coeff = sv[ ni ] / loads[ neighbour_rank ]; // / sv[ ni ];
                    double dist = samurai::getDistance<dim, fdist>( cell.center(), barycenter_neighbours[ ni ] ) * coeff;

                    // double dist = std::max( d1, d2 );

                    if( dist < winner_dist ){
                        winner_id   = ni;
                        winner_dist = dist;
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

                // shouldn't we give it to the second closest neighbour ?!
                if( given_[ it->second.rank ] + 1 <= ( - fluxes[ it->second.rank ] ) ){
                    
                    if constexpr ( dim == 3 ) {
                        auto i = it->second.cell.indices[ 0 ];
                        auto j = it->second.cell.indices[ 1 ];
                        auto k = it->second.cell.indices[ 2 ];
                        cl_to_send[ it->second.rank ][ it->second.cell.level ][ { j, k } ].add_point( i );
                    }else{
                        auto i = it->second.cell.indices[ 0 ];
                        auto j = it->second.cell.indices[ 1 ];
                        cl_to_send[ it->second.rank ][ it->second.cell.level ][ { j } ].add_point( i );
                    }

                    given_[ it->second.rank ] += 1;

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

                    logs << "Receiving data from # " << neighbourhood[ ni ].rank
                         << ", nbCells : " << to_rcv.nb_cells() << std::endl;

                    logs << "Merging cells, before : " << mesh.nb_cells( mesh_id_t::cells ) << std::endl;
                    mesh.merge( to_rcv );
                    logs << "Merging cells, after : " << mesh.nb_cells( mesh_id_t::cells ) << std::endl;

                }else{ // send data to 
                    world.send( neighbourhood[ ni ].rank, 42, ca_to_send[ ni ] );
                    
                    logs << "Sending data to # " << neighbourhood[ ni ].rank 
                         << ", nbCells : " << ca_to_send[ ni ].nb_cells() << std::endl;

                    logs << "Removing cells, before : " << mesh.nb_cells( mesh_id_t::cells ) << std::endl;

                    mesh.remove( ca_to_send[ ni ] );

                    logs << "Removing cells, after : " << mesh.nb_cells( mesh_id_t::cells ) << std::endl;

                }
            }

            // update neighbourhood
            // send current process mesh to neighbour and get neighbour mesh
            mesh.update_mesh_neighbour();

            // discover neighbours, since it might have changed
            bool requireNextIter = true; 
            
            // while( requireNextIter ){
                requireNextIter = samurai::discover_neighbour<dim>( mesh );
                requireNextIter = samurai::discover_neighbour<dim>( mesh );
            // }

        }
        
};