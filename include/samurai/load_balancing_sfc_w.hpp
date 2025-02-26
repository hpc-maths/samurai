/**
 *
 * This class implements SFC-based load balancing. The difference between this class
 * and the one in the load_balancing_sfc.hpp is that, here we try to use level based "weight"
 * to dispatch cell between processes in addition to the 1D-SFC key.
 *
 * This is a try to fix the unbalanced due to the difference of load induced by the difference of
 * level between two cells. With the traditional (load_balancing_sfc.hpp) strategy even if the
 * load, in term of number of cells on a process, is good, unbalanced on computational time may
 * appear.
 *
 */
#pragma once

#include "assertLogTrace.hpp"

#include "load_balancing.hpp"
#ifdef SAMURAI_WITH_MPI
namespace Load_balancing{ 

    template <int dim, class SFC_type_t>
    class SFCw : public samurai::LoadBalancer<SFCw<dim, SFC_type_t>>
    {
    private:

        SFC_type_t _sfc;
        int _ndomains;
        int _rank;

        const double _unbalance_threshold = 0.05; // 5 %

    public:

        using samurai::LoadBalancer<SFCw<dim, SFC_type_t>>::logs;

        SFCw()
        {
#ifdef SAMURAI_WITH_MPI
            boost::mpi::communicator world;
            _ndomains = world.size();
            _rank     = world.rank();
#else
            _ndomains = 1;
            _rank     = 0;
#endif
        }

        inline std::string getName() const
        {
            return "SFCw_" + _sfc.getName() + "_LB";
        }

        /**
        * Compute weights for cell at each level. We expect that small cell (high level)
        * cost more computational power that largest cell (low level). But this is impacted
        * by numerical scheme used. As default value, we use power of two starting from levelmin
        * (low level).
        */
        std::vector<double> getWeights( size_t levelmin, size_t levelmax ) {
            // Computing weights based on maxlevel
            std::vector<double> weights( levelmax + 1 );
            
            for( size_t ilvl=levelmin; ilvl<=levelmax; ++ilvl ){
                // weights[ ilvl ] = ( 1 << ( ilvl  - levelmin ) );       // prioritize small cell
                weights[ ilvl ] = 1 << (ilvl * ilvl);
                // weights[ ilvl ] = 1 << ( levelmax - ilvl ) ; // prioritize large cell
                // weights[ ilvl ] = 1.;                                // all equal
                logs << fmt::format("\t\t> Level {}, weight for cell : {}", ilvl, weights[ ilvl ] ) << std::endl;
            }

            return weights;
        }

        template <class Mesh_t>
        bool require_balance_impl( Mesh_t & mesh ) {

            boost::mpi::communicator world;

            logs << fmt::format("\n# [SFCw_LoadBalancer::Morton] required_balance_impl ") << std::endl;

            std::vector<double> weights = getWeights( mesh.min_level(), mesh.max_level() );

            double load = 0;
            samurai::for_each_cell( mesh[Mesh_t::mesh_id_t::cells], [&](const auto & cell ){
                load +=  weights[ cell.level ];
            });

            std::vector<double> nbLoadPerProc;
            boost::mpi::all_gather( world, load, nbLoadPerProc );
            
            double load_tot = 0.;
            for(size_t i=0; i<nbLoadPerProc.size(); ++i){
                load_tot += nbLoadPerProc[ i ];
            }

            double dc = load_tot / static_cast<double>( world.size() );

            for(size_t ip=0; ip<nbLoadPerProc.size(); ++ip ) {
                double diff = std::abs( nbLoadPerProc[ ip ] - dc ) / dc;

                if( diff > _unbalance_threshold ) return true;
            }

            return false;
        }

        /**
        * Re-order cells on MPI processes based on the given SFC curve. This need to be
        * called once at the beginning (unless ordering is fixed in partition mesh).
        * After, load balancing will exchange data based on SFC order and thus only
        * boundaries will be moved.
        *
        */
        template<class Mesh_t>
        auto reordering_impl( Mesh_t & mesh ) {

            boost::mpi::communicator world;

            // For debug
            // std::ofstream logs;
            // logs.open("log_" + std::to_string(_rank) + ".dat", std::ofstream::app);
            logs << "# [SFCw_LoadBalancer::Morton] Reordering cells using SFC" << std::endl;

            // SFC 1D key for cells
            auto sfc_keys = samurai::make_field<SFC_key_t, 1>( "keys", mesh );
            sfc_keys.fill( 0 );

            auto flags = samurai::make_field<int, 1>("rank", mesh);
            flags.fill( world.rank() );
            
            logs << fmt::format("\t> Computing SFC ({}) 1D indices ( cell ) ... ", _sfc.getName() ) << std::endl;

            SFC_key_t mink = std::numeric_limits<SFC_key_t>::max(), maxk = std::numeric_limits<SFC_key_t>::min();

            samurai::for_each_cell(mesh[Mesh_t::mesh_id_t::cells], [&]( const auto & cell ) {

                // this is where things can get nasty, we expect indices to be positive values !!
                xt::xtensor_fixed<uint32_t, xt::xshape<Mesh_t::dim>> ijk;
                for (size_t idim = 0; idim < dim; ++idim) {

                    // FIX need shift to get only positive index
                    assert( cell.indices( idim ) >= 0 );
                    ijk( idim ) = static_cast<uint32_t>( cell.indices( idim ) ) << ( mesh.max_level() - cell.level );
                }

                auto key = _sfc.template getKey<dim>( ijk );

                sfc_keys[ cell ] = key;
                
                mink = std::min( key, mink );
                maxk = std::max( key, maxk );

            });

            // Key boundaries of current process - unused for now
            std::vector<SFC_key_t> bounds = { mink, maxk };
            logs << "\t\t> Local key bounds [" << bounds[ 0 ] << ", " << bounds[ 1 ] << "]" << std::endl;

            std::vector<SFC_key_t> boundaries;
            boost::mpi::all_gather( world, bounds.data(), static_cast<int>( bounds.size() ), boundaries );

            logs << "\t\t> Global key boundaries [";
            for(const auto & ik : boundaries )
                logs << ik << ",";
            logs << "]" << std::endl;

            // Check overlap with previous/next process. Does not mean that there is no overlap, but at least between "adjacent"
            // (MPI-1), (MPI+1) there is not overlap found
            std::vector<SFC_key_t> boundaries_new( static_cast<size_t>( world.size() + 1 ) );

            // find max value for boundaries
            SFC_key_t globalMax = boundaries[ 0 ];
            for(size_t ip=0; ip<boundaries.size(); ++ip){
                globalMax = std::max( boundaries[ ip ], globalMax );
            }

            // cmpt max theoretical key
            globalMax += 1;

            // evenly spaced intervals
            SFC_key_t ds = globalMax / static_cast<size_t>( world.size() );
            boundaries_new[ 0 ] = 0;
            for(size_t ip=1; ip<boundaries_new.size(); ++ip){
                boundaries_new[ ip ] = boundaries_new[ ip - 1 ] + ds;
            }

            logs << "\t\t> Global key evenly distrib boundaries [";
            for(const auto & ik : boundaries_new )
                logs << ik << ",";
            logs << "]" << std::endl;

            // distribute cell based on boundaries & sfc key
            std::map<int, bool> comm;
            samurai::for_each_cell( mesh[Mesh_t::mesh_id_t::cells], [&](const auto & cell ) {
                auto key = sfc_keys[ cell ];

                // optimize using bisect - find proc that should have this cell
                for( size_t ip=0; ip< static_cast<size_t>( world.size() ); ++ip ){
                    if( key >= boundaries_new[ ip ] &&  key < boundaries_new[ ip + 1 ] ){
                        flags[ cell ] = static_cast<int>( ip );

                        // unique list of process that should be contacted
                        if( comm.find( static_cast<int>( ip ) ) == comm.end() ){
                            comm[ static_cast<int>( ip ) ] = true;
                        }

                        break;
                    }
                }

            });

            return flags;
        }

        template <class Mesh_t>
        auto load_balance_impl( Mesh_t & mesh )
        {

            boost::mpi::communicator world;
            
            // std::ofstream logs;
            // logs.open( "log_" + std::to_string( world.rank() ) + ".dat", std::ofstream::app );
            logs << fmt::format("\n# [SFCw_LoadBalancer::Morton] Load balancing cells ") << std::endl;

            // SFC 1D key for each cell
            std::map<SFC_key_t, typename Mesh_t::cell_t> sfc_map;
            auto sfc_keys = samurai::make_field<SFC_key_t, 1>( "keys", mesh );
            sfc_keys.fill( 0 );

            // MPI destination rank of process for each cell
            auto flags = samurai::make_field<int, 1>("rank", mesh);
            flags.fill( world.rank() );

            // Computing weights based on maxlevel
            // std::vector<double> weights( mesh.max_level() + 1 );
            // for( size_t ilvl=mesh.min_level(); ilvl<=mesh.max_level(); ++ilvl ){
            //     weights[ ilvl ] = 1 << ( ( ilvl  - mesh.min_level() ) ) ;
            //     // weights[ ilvl ] = 1 << ( mesh.max_level() - ilvl ) ;
            //     // weights[ ilvl ] = 1.;
            //     logs << fmt::format("\t\t> Level {}, weight for cell : {}", ilvl, weights[ ilvl ] ) << std::endl;
            // }

            std::vector<double> weights = getWeights( mesh.min_level(), mesh.max_level() );
            
            
            logs << fmt::format("\t> Computing SFC ({}) 1D indices ( cell ) ... ", _sfc.getName() ) << std::endl;

            // SFC_key_t mink = std::numeric_limits<SFC_key_t>::max(), maxk = std::numeric_limits<SFC_key_t>::min();

            // compute SFC key for each cell
            samurai::for_each_cell( mesh[ Mesh_t::mesh_id_t::cells ], [&]( const auto & cell ) {

                // this is where things can get nasty, we expect indices to be positive values !!
                xt::xtensor_fixed<uint32_t, xt::xshape<Mesh_t::dim>> ijk;
                for (size_t idim = 0; idim < Mesh_t::dim; ++idim) {

                    // FIX need shift to get only positive index
                    assert( cell.indices( idim ) >= 0 );
                    ijk( idim ) = static_cast<uint32_t>( cell.indices( idim ) ) << ( mesh.max_level() - cell.level );
                }

                auto key = _sfc.template getKey<dim>( ijk );

                sfc_keys[ cell ] = key;

                if( sfc_map.find( key ) != sfc_map.end() ) {
                    assert( false );
                    std::cerr << fmt::format("Rank # {}, Error computing SFC, index not uniq ! ", world.rank()) << std::endl;
                }

                sfc_map[ key ] = cell;

                // mink = std::min( mink, key );
                // maxk = std::max( maxk, key ); 

            });

            assert( mesh.nb_cells( Mesh_t::mesh_id_t::cells ) == sfc_map.size() );

            size_t nload_tot = 0, dc = 0;
            std::vector<size_t> nbLoadPerProc;
            std::vector<size_t> globIdx( static_cast<size_t>( world.size() + 1 ) );
            std::vector<size_t> globIdxNew( static_cast<size_t>( world.size() + 1 ) );

            // get load for each MPI process
            size_t load = 0;
            samurai::for_each_cell( mesh[Mesh_t::mesh_id_t::cells], [&](const auto & cell ){
                load +=  weights[ cell.level ];
            });

            boost::mpi::all_gather( world, load, nbLoadPerProc );

            for(size_t i=0; i<static_cast<size_t>( world.size() ); ++i){
                globIdx[ i + 1 ] = globIdx[ i ] + nbLoadPerProc[ i ];
                nload_tot += nbLoadPerProc[ i ];
            }
            dc = nload_tot / static_cast<size_t>( world.size() );

            logs << fmt::format("\t\t> Load of cells (weighted) : {}, dc : {}", load, dc) << std::endl;

            // load balanced globIdx -> new theoretical key boundaries based on number of cells per proc
            globIdxNew[ 0 ] = 0;
            for(size_t i=0; i<static_cast<size_t>( world.size() ); ++i){
                globIdxNew[ i + 1 ] = globIdxNew[ i ] + dc;
            }

            {
                logs << "\t\t> GlobalIdx : ";
                for(const auto & i : globIdx )
                    logs << i << ", ";
                logs << std::endl;

                logs << "\t\t> GlobalIdx balanced : ";
                for(const auto & i : globIdxNew )
                    logs << i << ", ";
                logs << std::endl;
                
            }   

            size_t start = 0;
            while( globIdx[ static_cast<size_t>( world.rank() ) ] >= ( start + 1 ) * dc ){
                start ++;
            }

            logs << "\t\t> Start @ rank " << start << std::endl;

            size_t count = globIdx[ static_cast<size_t>( world.rank() ) ];
            for( auto & it : sfc_map ) {

                if( count >= ( start + 1 ) * dc ){
                    start ++;
                    start = std::min( static_cast<size_t>( world.size() - 1 ) , start );
                    logs << "\t\t> Incrementing Start @ rank " << start << ", count " << count << std::endl;
                }
                
                flags[ it.second ] = static_cast<int>( start );
                
                count += weights[ it.second.level ];
            }

            return flags;
        }

    };

}
#endif
