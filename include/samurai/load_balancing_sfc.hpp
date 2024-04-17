#pragma once

#include "assertLogTrace.hpp"

#include "load_balancing.hpp"

template<int dim, class SFC_type_t>
class SFC_LoadBalancer_interval : public samurai::LoadBalancer<SFC_LoadBalancer_interval<dim, SFC_type_t>> {

    private:
        SFC_type_t _sfc;
        int _ndomains;
        int _rank;

        const double TRANSFER_PERCENT = 0.42;

    public:

        SFC_LoadBalancer_interval() {

#ifdef SAMURAI_WITH_MPI
            boost::mpi::communicator world;
            _ndomains = world.size();
            _rank     = world.rank();
#else
            _ndomains = 1;
            _rank     = 0;
#endif
        }

        inline std::string getName() const { return "SFC_" + _sfc.getName() + "_LB"; }

        template<class Mesh, class... Fields>
        Mesh load_balance_impl( Mesh & mesh, Fields&... data ){
            
            using Config  = samurai::MRConfig<dim>;
            using Mesh_t  = samurai::MRMesh<Config>;
            using inter_t = samurai::Interval<int, long long>;
            using CellList_t      = typename Mesh_t::cl_type;
            using CellArray_t     = samurai::CellArray<dim>;

            struct Data_t {
                size_t level;
                inter_t interval;
                xt::xtensor_fixed<samurai::default_config::value_t, xt::xshape<dim - 1>> indices;
                bool given;
            };

            boost::mpi::communicator world;

            // For debug
            std::ofstream logs; 
            logs.open( "log_" + std::to_string( _rank ) + ".dat", std::ofstream::app );
            logs << "# New load balancing (load_balancing_sfc)" << std::endl;

            // SFC key (used for implicit sorting through map mechanism)
            std::map<SFC_key_t, Data_t> sfc_map;

            size_t ninterval = 0;
            samurai::for_each_interval( mesh, [&]( std::size_t level, const auto& inter, const auto& index ){

                // get Logical coordinate or first cell
                xt::xtensor_fixed<int, xt::xshape<dim>> icell;
                
                // first element of interval
                icell( 0 ) = inter.start; 
                for(int idim=0; idim<dim-1; ++idim){
                    icell( idim + 1 ) = index( idim );
                }

                // convert logical coordinate to max level logical coordinates
                for( int idim=0; idim<dim; ++idim ){
                    icell( idim ) = icell( idim ) << ( mesh.max_level() - level + 1 );
                }

                // this is where think can get nasty,  we expect indices to be positive values !!
                xt::xtensor_fixed<uint32_t, xt::xshape<dim>> ijk;
                for(size_t idim=0; idim<dim; ++idim ){
                    ijk( idim ) = static_cast<uint32_t>( icell( idim ) );
                }
                
                sfc_map[ _sfc.template getKey<dim>( ijk ) ] = { level, inter, index, false };

                ninterval ++;
            });

            assert( ninterval == sfc_map.size() );

            // Key boundaries of current process - unused for now
            // SFC_key_t interval[ 2 ] = { sfc_map.begin()->first, sfc_map.rbegin()->first };
            // logs << "Boundaries [" << interval[ 0 ] << ", " << interval[ 1 ] << "]" << std::endl;

            std::vector<int> load_interval;
            int my_load_i = static_cast<int>( samurai::cmptLoad<samurai::BalanceElement_t::INTERVAL>( mesh ) );
            boost::mpi::all_gather( world, my_load_i, load_interval );

            // compute load to transfer to neighbour rank-1, rank+1
            int neighbour_rank_prev = -1, neighbour_rank_next = -1;
            int transfer_load_prev = 0, transfer_load_next = 0;

            // define neighbour processes for load-balancing, not geometrical neighbour !
            if( _rank > 0 ) { 
                neighbour_rank_prev = _rank - 1;
                // transfer TRANSFER_PERCENT % max of difference
                transfer_load_prev = - static_cast<int>( ( my_load_i - load_interval[ static_cast<std::size_t>( neighbour_rank_prev ) ] ) * TRANSFER_PERCENT );
            }

            if( _rank < _ndomains - 1 ) { 
                neighbour_rank_next = _rank + 1;
                // transfer TRANSFER_PERCENT % max of difference
                transfer_load_next = - static_cast<int>( ( my_load_i - load_interval[ static_cast<std::size_t>( neighbour_rank_next ) ] ) * TRANSFER_PERCENT );
            }

            logs << "Neighbour prev : " << neighbour_rank_prev << ", transfer of loads : " << transfer_load_prev << std::endl;
            logs << "Neighbour next : " << neighbour_rank_next << ", transfer of loads : " << transfer_load_next << std::endl;

            /**
             * In this section we update the current process mesh data based on what is sent and receive
             * 
            */

            CellList_t new_cl; // this will contains the final mesh of the current process

            // need send data to prev neighbour
            if( neighbour_rank_prev >= 0 && transfer_load_prev != 0 ){

                if( transfer_load_prev < 0 ){
                    CellList_t cl_to_send;

                    // give n-smallest morton keys to prev neighbour
                    size_t niter_send = 0;
                    for (auto iter = sfc_map.begin(); iter != sfc_map.end(); ++iter) {
                        if( transfer_load_prev < 0 && my_load_i > 0 ){
                            cl_to_send[ iter->second.level][ iter->second.indices ].add_interval( iter->second.interval );
                            iter->second.given = true; // flag for "interval" has been sent
                            my_load_i -= 1;
                            transfer_load_prev += 1;
                            niter_send ++;
                        }else{
                            break;
                        }
                    }

                    CellArray_t ca_to_send = { cl_to_send, false };

                    logs << "\t> Sending nbCells {" << ca_to_send.nb_cells() << "} to process : " << neighbour_rank_prev << std::endl;

                    world.send( neighbour_rank_prev, 42, ca_to_send );

                    // auto new_mesh = samurai::load_balance::remove( mesh, ca_to_send );
                    // logs << "\t> New mesh : " << new_mesh.nb_cells() << std::endl;

                    // mesh.remove( ca_to_send );

                }else{
                    // need recv
                    CellArray_t ca_to_rcv;
                    world.recv( neighbour_rank_prev, 42, ca_to_rcv );

                    logs << "\t> Receiving nbCells {" << ca_to_rcv.nb_cells() << "} from process : " << neighbour_rank_prev << std::endl;

                    // mesh.merge( ca_to_rcv );
                    // add to CL what we just receive
                    samurai::for_each_interval( ca_to_rcv, [&](std::size_t level, const auto & interval, const auto & index ){
                        new_cl[ level ][ index ].add_interval( interval );
                    });
                    
                }

            }

            if( neighbour_rank_next > 0 && transfer_load_next != 0 ){

                if( transfer_load_next < 0 ){
                    CellList_t cl_to_send;

                    // give n-smallest morton keys to prev neighbour
                    size_t niter_send = 0;
                    for (auto iter = sfc_map.rbegin(); iter != sfc_map.rend(); ++iter) {
                        if( transfer_load_next < 0 && my_load_i > 0 ){
                            cl_to_send[ iter->second.level][ iter->second.indices ].add_interval( iter->second.interval );
                            iter->second.given = true;
                            my_load_i -= 1;
                            transfer_load_next += 1;
                            niter_send ++;
                        }else{
                            break;
                        }
                    }

                    CellArray_t ca_to_send = { cl_to_send, false };

                    logs << "\t> Sending nbCells {" << ca_to_send.nb_cells() << "} to process : " << neighbour_rank_next << std::endl;

                    world.send( neighbour_rank_next, 42, ca_to_send );
                    
                    // auto new_mesh = samurai::load_balance::remove( mesh, ca_to_send );
                    // logs << "\t> New mesh : " << new_mesh.nb_cells() << std::endl;

                    mesh.remove( ca_to_send );

                }else{
                    // need recv
                    CellArray_t ca_to_rcv;
                    world.recv( neighbour_rank_next, 42, ca_to_rcv );

                    logs << "\t> Receiving nbCells {" << ca_to_rcv.nb_cells() << "} from process : " << neighbour_rank_next << std::endl;

                    // mesh.merge( ca_to_rcv );
                    // add to CL what we just receive
                    samurai::for_each_interval( ca_to_rcv, [&](std::size_t level, const auto & interval, const auto & index ){
                        new_cl[ level ][ index ].add_interval( interval );
                    });

                }

            }

            // last loop over the map to add what is left
            for (auto iter = sfc_map.rbegin(); iter != sfc_map.rend(); ++iter) {
                if( ! iter->second.given ){
                    new_cl[ iter->second.level][ iter->second.indices ].add_interval( iter->second.interval );
                }
            }

            Mesh_t new_mesh( new_cl, mesh.min_level(), mesh.max_level(), mesh.mpi_neighbourhood() ); 
            // mesh.update_mesh_neighbour(); done already in new_mesh constructor

            // update neighbour connectivity
            samurai::discover_neighbour<dim>( new_mesh );
            samurai::discover_neighbour<dim>( new_mesh );


            return new_mesh;

        }

};