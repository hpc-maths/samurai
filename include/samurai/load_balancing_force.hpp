#pragma once

#include <map>
#include "load_balancing.hpp"

template<size_t dim>
class Diffusion_LoadBalancer_cell : public samurai::LoadBalancer<Diffusion_LoadBalancer_cell<dim>> {

    using Coord_t = xt::xtensor_fixed<double, xt::xshape<dim>>;

    private:
        int _ndomains;
        int _rank;

        const double _unbalance_threshold = 0.05; // 5 %

        template<class Mesh_t>
        double getSurfaceOrVolume( Mesh_t & mesh ) const {

            double s_ = 0.;
            for_each_cell( mesh[ Mesh_t::mesh_id_t::cells ], [&]( const auto& cell ){
                
                double s = 1.;

                for(size_t idim=0; idim<Mesh_t::dim; ++idim){
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

        template <class Mesh_t>
        bool require_balance_impl( Mesh_t & mesh ) {

            boost::mpi::communicator world;

            // logs << fmt::format("\n# [SFC_LoadBalancer_interval::Morton] required_balance_impl ") << std::endl;

            double nbCells_tot = 0;
            std::vector<double> nbCellsPerProc;
            boost::mpi::all_gather( world, static_cast<double>( mesh.nb_cells( Mesh_t::mesh_id_t::cells ) ), nbCellsPerProc );

            for(size_t ip=0; ip<nbCellsPerProc.size(); ++ip ) {
                nbCells_tot += nbCellsPerProc[ ip ];
            }

            // no weight while computing load 
            double dc = nbCells_tot  / static_cast<double> ( world.size() );

            for(size_t ip=0; ip<nbCellsPerProc.size(); ++ip ) {
                double diff = std::abs( nbCellsPerProc[ ip ] - dc ) / dc;

                if( diff > _unbalance_threshold ) return true;
            }

            return false;
        }

        template<class Mesh_t>
        auto reordering_impl( Mesh_t & mesh ){
            auto flags = samurai::make_field<int, 1>("rank", mesh);
            flags.fill( _rank );
            return flags;
        }

        template<class Mesh_t>
        auto load_balance_impl( Mesh_t & mesh ){

            using mpi_subdomain_t = typename Mesh_t::mpi_subdomain_t;

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

            // get the load to neighbours (geometrical neighbour) with 5 iterations max
            std::vector<int> fluxes = samurai::cmptFluxes<samurai::BalanceElement_t::CELL>( mesh, 1 );

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

            // Interface for each neighbour: std::vector<Cell_array_t>
            auto interface = samurai::_computeCartesianInterface<dim, samurai::Direction_t::FACE_AND_DIAG>( mesh );

            // Invalidate fluxes with non adjacents neighbours, should not happen or might happen if load balancing exchange does not consider some
            // direction.
            std::vector<size_t> ncells_interface ( n_neighbours, 0 );
            for(size_t ni=0; ni<n_neighbours; ni++){
                samurai::for_each_interval( interface[ ni ], [&]([[maybe_unused]] std::size_t level, [[maybe_unused]] const auto & interval, 
                                                      [[maybe_unused]] const auto & index ){
                    ncells_interface[ ni ] ++;
                });

                if( ncells_interface[ ni ] == 0 ) fluxes[ ni ] = 0;
            }

            // bary center of current mesh - no weight on cells
            // Coord_t barycenter = _cmpIntervalBarycenter( mesh[ mesh_id_t::cells ] );
            Coord_t barycenter = samurai::_cmpCellBarycenter<dim>( mesh );
            logs << "Domain barycenter : " << fmt::format( " barycenter : ({}, {})", barycenter(0), barycenter(1) ) << std::endl;

            // std::vector<Coord_t> barycenter_interface_neighbours( n_neighbours );
            std::vector<Coord_t> barycenter_neighbours( n_neighbours );
            std::vector<double> sv( n_neighbours );

            for(size_t nbi=0; nbi<n_neighbours; ++nbi ){
                // barycenter_interface_neighbours[ nbi ] = _cmpIntervalBarycenter( interface[ nbi ] );
                // barycenter_interface_neighbours[ nbi ] = _cmpCellBarycenter<dim>( interface[ nbi ] );
                barycenter_neighbours[ nbi ] = samurai::_cmpCellBarycenter<dim>( neighbourhood[ nbi ].mesh );
                
                // surface or volume depending on dim
                sv[ nbi ] = getSurfaceOrVolume( neighbourhood[ nbi ].mesh );

                // debug
                auto s_ = fmt::format( "Barycenter neighbour # {}: ({}, {})", neighbourhood[ nbi ].rank,
                               barycenter_neighbours[ nbi ]( 0 ),
                               barycenter_neighbours[ nbi ]( 1 ) );

                logs << s_ << std::endl;
            }

            // build map of interval that needs to be sent. Warning, it does not work with classical std::map !!
            auto flags = samurai::make_field<int, 1>("rank", mesh);
            flags.fill( world.rank() );

            constexpr auto fdist = samurai::Distance_t::L1;

            double currentSV = getSurfaceOrVolume( mesh );

            std::vector<int> mload( static_cast<size_t>( world.size() ), 0 );
            for_each_cell( mesh[ Mesh_t::mesh_id_t::cells ], [&]( const auto& cell ){

                auto cc = cell.center();

                // process that might get the interval
                int winner_id = -1;

                // double winner_dist = std::numeric_limits<double>::max();
                // double winner_dist = samurai::getDistance<dim, fdist>( cell, barycenter ) / loads[ world.rank() ];
                
                // double coeff_current = currentSV / loads[ static_cast<size_t>( world.rank() ) ];
                double coeff_current = 1.; //  loads[ static_cast<size_t>( world.rank() ) ];
                double winner_dist = samurai::getDistance<dim, fdist>( cc, barycenter ) * coeff_current;

                // logs << fmt::format("\t\t> Cell : ({},{}), current mesh dist : {}", cc(0), cc(1), winner_dist ) << std::endl;

                // select the neighbour
                for( std::size_t ni=0; ni<n_neighbours; ++ni ){ // for each neighbour

                    auto neighbour_rank = static_cast<std::size_t>( neighbourhood[ ni ].rank );

                    if( fluxes[ ni ] >= 0 ) continue; // skip neighbour that will recv

                    // this might fix ilots but require neighbour update
                    // double dist = std::min( distance_inf<dim>( mid_point, barycenter_interface_neighbours[ ni ] ),
                    //                         distance_inf<dim>( mid_point, barycenter_neighbours[ ni ] ) );

                    // double dist = samurai::getDistance<dim, fdist>( cell, barycenter_interface_neighbours[ ni ] ) / loads[ neighbour_rank ];
                    // double coeff = sv[ ni ] / ( loads[ neighbour_rank ] ) ; // / sv[ ni ];
                    double coeff = 1.; // loads[ neighbour_rank ] ; // mload[ neighbour_rank ];
                    double dist = samurai::getDistance<dim, fdist>( cell.center(), barycenter_neighbours[ ni ] ) * coeff;

                    // logs << fmt::format("\t\t\t> Dist to neighbour {} : {}", neighbour_rank, dist ) << std::endl;

                    if( dist < winner_dist && ncells_interface[ ni ] > 0 ){
                        winner_id   = static_cast<int>( ni );
                        winner_dist = dist;

                        mload[ neighbour_rank ] += 1;

                        // if( mload[ neighbour_rank ] >= ( - fluxes[ ni ] ) ) fluxes[ ni ] = 0;
                    }
                    
                }

                // logs << fmt::format("\t\t\t> Cell given to process #{}", neighbourhood[ static_cast<size_t>( winner_id ) ].rank ) << std::endl;

                if( winner_id >= 0 ){
                    assert( winner_id < neighbourhood.size() );
                    flags[ cell ] = neighbourhood[ static_cast<size_t>( winner_id ) ].rank; 
                }

            });

            return flags;
        }
        
};