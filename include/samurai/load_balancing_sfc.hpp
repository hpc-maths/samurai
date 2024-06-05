#pragma once

#include "assertLogTrace.hpp"

#include "load_balancing.hpp"

template <int dim, class SFC_type_t>
class SFC_LoadBalancer_interval : public samurai::LoadBalancer<SFC_LoadBalancer_interval<dim, SFC_type_t>>
{
  private:

    SFC_type_t _sfc;
    int _ndomains;
    int _rank;

    const double TRANSFER_PERCENT = 0.5;

  public:

    SFC_LoadBalancer_interval()
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
        return "SFC_" + _sfc.getName() + "_LB";
    }

    /**
    * Re-order cells on MPI processes based on the given SFC curve. This need to be
    * called once at the beginning (unless ordering is fixed in partition mesh).
    * After, load balancing will exchange data based on SFC order and thus only
    * boundaries will be moved.
    *
    */
    template<class Mesh_t>
    Mesh_t reordering_impl( Mesh_t & mesh ) {
        using CellList_t  = typename Mesh_t::cl_type;
        using CellArray_t = samurai::CellArray<dim>;

        boost::mpi::communicator world;

        // For debug
        std::ofstream logs;
        logs.open("log_" + std::to_string(_rank) + ".dat", std::ofstream::app);
        logs << "# [SFC_LoadBalancer_interval::Morton] Reordering cells using SFC" << std::endl;

        // SFC 1D key for cells
        auto sfc_keys = samurai::make_field<SFC_key_t, 1>( "keys", mesh );
        sfc_keys.fill( 0 );

        auto flags = samurai::make_field<int, 1>("rank", mesh);
        flags.fill( world.rank() );
         
        logs << fmt::format("\t> Computing SFC ({}) 1D indices ( cell ) ... ", _sfc.getName() ) << std::endl;

        SFC_key_t mink = std::numeric_limits<SFC_key_t>::max(), maxk = std::numeric_limits<SFC_key_t>::min();

        samurai::for_each_cell(mesh[Mesh_t::mesh_id_t::cells], [&]( const auto & cell ) {

            // this is where things can get nasty, we expect indices to be positive values !!
            xt::xtensor_fixed<uint32_t, xt::xshape<dim>> ijk;
            for (size_t idim = 0; idim < dim; ++idim) {

                // FIX need shift to get only positive index
                assert( cell.indices( 0 ) >= 0 );
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

        logs << "\t\t> Communication for exchange required with rank : [";
        for( const auto & it : comm )
            logs << it.first << ",";
        logs << "]" << std::endl;

        // /* ---------------------------------------------------------------------------------------------------------- */
        // /* ------- Data transfer between processes ------------------------------------------------------------------ */ 
        // /* ---------------------------------------------------------------------------------------------------------- */

        CellList_t new_cl;
        std::vector<CellList_t> payload( static_cast<size_t>( world.size() ) );

        samurai::for_each_cell( mesh[Mesh_t::mesh_id_t::cells], [&]( const auto & cell ){
        
            if( flags[ cell ] == world.rank() ){
                if constexpr ( Mesh_t::dim == 1 ){ new_cl[ cell.level ][ {} ].add_point( cell.indices[ 0 ] ); }
                if constexpr ( Mesh_t::dim == 2 ){ new_cl[ cell.level ][ { cell.indices[ 1 ] } ].add_point( cell.indices[ 0 ] ); }
                if constexpr ( Mesh_t::dim == 3 ){ new_cl[ cell.level ][ { cell.indices[ 1 ], cell.indices[ 2 ] } ].add_point( cell.indices[ 0 ] ); }                        
            }else{
                if constexpr ( Mesh_t::dim == 1 ){ payload[ static_cast<size_t>( flags[ cell ] ) ][ cell.level ][ {} ].add_point( cell.indices[ 0 ] ); }
                if constexpr ( Mesh_t::dim == 2 ){ payload[ static_cast<size_t>( flags[ cell ] ) ][ cell.level ][ { cell.indices[ 1 ] } ].add_point( cell.indices[ 0 ] ); }
                if constexpr ( Mesh_t::dim == 3 ){ payload[ static_cast<size_t>( flags[ cell ] ) ][ cell.level ][ { cell.indices[ 1 ], cell.indices[ 2 ] } ].add_point( cell.indices[ 0 ] ); }
            } 

        });

        // FIXME: this part involve a lot of communication since each process will communicate with all processes.
        // This should be improved for better scalability.
        for( int iproc=0; iproc<world.size(); ++iproc ){
            if( iproc == world.rank() ) continue;

            int reqExchg;
            comm.find( iproc ) != comm.end() ? reqExchg = 1 : reqExchg = 0;

            world.send( iproc, 17, reqExchg );

            if( reqExchg == 1 ) {
                CellArray_t to_send = { payload[ static_cast<size_t>( iproc ) ], false };
                world.send( iproc, 17, to_send );
            }
            
        }

        for( int iproc=0; iproc<world.size(); ++iproc ){

            if( iproc == world.rank() ) continue;

            int reqExchg = 0;
            world.recv( iproc, 17, reqExchg );

            if( reqExchg == 1 ) {
                CellArray_t to_rcv;
                world.recv( iproc, 17, to_rcv );

                samurai::for_each_interval(to_rcv, [&](std::size_t level, const auto & interval, const auto & index ){
                    new_cl[ level ][ index ].add_interval( interval );
                });
            }
            
        }

        // /* ---------------------------------------------------------------------------------------------------------- */
        // /* ------- Construct new mesh for current process ----------------------------------------------------------- */ 
        // /* ---------------------------------------------------------------------------------------------------------- */

        Mesh_t new_mesh( new_cl, mesh );

        return new_mesh;
    }

    template <class Mesh_t>
    auto load_balance_impl( Mesh_t & mesh )
    {

        boost::mpi::communicator world;

        // For debug
        std::ofstream logs;
        logs.open("log_" + std::to_string(_rank) + ".dat", std::ofstream::app);
        
        logs << fmt::format("\n# [SFC_LoadBalancer_interval::Morton] Load balancing cells ") << std::endl;

        // SFC 1D key for cells
        std::map<SFC_key_t, typename Mesh_t::cell_t> sfc_map;
        auto sfc_keys = samurai::make_field<SFC_key_t, 1>( "keys", mesh );
        sfc_keys.fill( 0 );

        auto flags = samurai::make_field<int, 1>("rank", mesh);
        flags.fill( world.rank() );
         
        logs << fmt::format("\t> Computing SFC ({}) 1D indices ( cell ) ... ", _sfc.getName() ) << std::endl;

        SFC_key_t mink = std::numeric_limits<SFC_key_t>::max(), maxk = std::numeric_limits<SFC_key_t>::min();
        samurai::for_each_cell( mesh[ Mesh_t::mesh_id_t::cells ], [&]( const auto & cell ) {

            // this is where things can get nasty, we expect indices to be positive values !!
            xt::xtensor_fixed<uint32_t, xt::xshape<dim>> ijk;
            for (size_t idim = 0; idim < dim; ++idim) {

                // FIX need shift to get only positive index
                assert( cell.indices( 0 ) >= 0 );
                ijk( idim ) = static_cast<uint32_t>( cell.indices( idim ) ) << ( mesh.max_level() - cell.level );
            }

            auto key = _sfc.template getKey<dim>( ijk );

            sfc_keys[ cell ] = key;

            if( sfc_map.find( key ) != sfc_map.end() ) {
                assert( false );
                std::cerr << fmt::format("Rank # {}, Error computing SFC, index not uniq ! ", world.rank()) << std::endl;
            }

            sfc_map[ key ] = cell;

            mink = std::min( mink, key );
            maxk = std::max( maxk, key ); 

        });

        assert( mesh.nb_cells( Mesh_t::mesh_id_t::cells ) == sfc_map.size() );

        size_t ncells_tot = 0, dc = 0;
        std::vector<size_t> nbCellsPerProc;
        std::vector<size_t> globIdx( static_cast<size_t>( world.size() + 1 ) );
        std::vector<size_t> globIdxNew( static_cast<size_t>( world.size() + 1 ) );
        boost::mpi::all_gather( world, mesh.nb_cells( Mesh_t::mesh_id_t::cells ), nbCellsPerProc );

        logs << "\t\t> Number of cells : " << mesh.nb_cells( Mesh_t::mesh_id_t::cells ) << std::endl;

        for(size_t i=0; i<static_cast<size_t>( world.size() ); ++i){
            globIdx[ i + 1 ] = globIdx[ i ] + nbCellsPerProc[ i ];
            ncells_tot += nbCellsPerProc[ i ];
        }
        dc = ncells_tot / static_cast<size_t>( world.size() );

        // load balanced globIdx
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

        int start = 0;
        while( globIdx[ static_cast<size_t>( world.rank() ) ] >= static_cast<size_t>( start + 1 ) * dc ){
            start ++;
        }

        logs << "\t\t> Start @ rank " << start << std::endl;

        size_t count = globIdx[ static_cast<size_t>( world.rank() ) ];
        for( auto & it : sfc_map ) {

            if( count >= ( start + 1 ) * dc ){
                start ++;
                start = std::min( world.size() - 1 , start );
                logs << "\t\t> Incrementing Start @ rank " << start << ", count " << count << std::endl;
            }
            
            flags[ it.second ] = start;
            
            count ++;
        }

        return flags;
    }

};