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
        logs << "# New load balancing (load_balancing_sfc)" << std::endl;

        // SFC 1D key for cells
        auto sfc_keys = samurai::make_field<SFC_key_t, 1>( "keys", mesh );
        sfc_keys.fill( 0 );

        auto flags = samurai::make_field<int, 1>("rank", mesh);
        flags.fill( world.rank() );
         
        logs << fmt::format("\t\t> Computing SFC ({}) 1D indices ( cell ) ... ", _sfc.getName() ) << std::endl;

        SFC_key_t mink = std::numeric_limits<SFC_key_t>::max(), maxk = std::numeric_limits<SFC_key_t>::min();

        size_t ncells = 0;
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

            ncells ++;

        });

        // Key boundaries of current process - unused for now
        std::vector<SFC_key_t> bounds = { mink, maxk };
        logs << "\t\t\t> Local key bounds [" << bounds[ 0 ] << ", " << bounds[ 1 ] << "]" << std::endl;

        std::vector<SFC_key_t> boundaries;
        boost::mpi::all_gather( world, bounds.data(), bounds.size(), boundaries );

        logs << "\t\t\t> Global key boundaries [";
        for(const auto & ik : boundaries )
            logs << ik << ",";
        logs << "]" << std::endl;

        // Check overlap with previous/next process. Does not mean that there is no overlap, but at least between "adjacent"
        // (MPI-1), (MPI+1) there is not overlap found
        std::vector<SFC_key_t> boundaries_new( world.size() + 1 );

        // find max value for boundaries
        SFC_key_t globalMax = boundaries[ 0 ];
        for(size_t ip=0; ip<boundaries.size(); ++ip){
            globalMax = std::max( boundaries[ ip ], globalMax );
        }

        // cmpt max theoretical key
        globalMax += 1;

        // evenly spaced intervals
        SFC_key_t ds = globalMax / world.size();
        boundaries_new[ 0 ] = 0;
        for(size_t ip=1; ip<world.size() + 1; ++ip){
            boundaries_new[ ip ] = boundaries_new[ ip - 1 ] + ds;
        }

        logs << "\t\t\t> Global key evenly distrib boundaries [";
        for(const auto & ik : boundaries_new )
            logs << ik << ",";
        logs << "]" << std::endl;

        // distribute cell based on boundaries & sfc key
        std::map<int, bool> comm;
        samurai::for_each_cell( mesh[Mesh_t::mesh_id_t::cells], [&](const auto & cell ) {
            auto key = sfc_keys[ cell ];

            // optimize using bisect - find proc that should have this cell
            for( size_t ip=0; ip<world.size(); ++ip ){
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

        logs << "\t\t\t> Comm required with processes : [";
        for( const auto & it : comm )
            logs << it.first << ",";
        logs << "]" << std::endl;

        // /* ---------------------------------------------------------------------------------------------------------- */
        // /* ------- Data transfer between processes ------------------------------------------------------------------ */ 
        // /* ---------------------------------------------------------------------------------------------------------- */

        CellList_t new_cl;
        std::vector<CellList_t> payload( world.size() );

        samurai::for_each_cell( mesh[Mesh_t::mesh_id_t::cells], [&]( const auto & cell ){
        
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

        // FIXME: this part involve a lot of communication since each process will communicate with all processes.
        // This should be improved for better scalability.
        for( int iproc=0; iproc<world.size(); ++iproc ){
            if( iproc == world.rank() ) continue;

            int reqExchg;
            comm.find( iproc ) != comm.end() ? reqExchg = 1 : reqExchg = 0;

            world.send( iproc, 17, reqExchg );

            if( reqExchg == 1 ) {
                CellArray_t to_send = { payload[ iproc ], false };
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
    Mesh_t load_balance_impl( Mesh_t & mesh )
    {
        using inter_t     = samurai::Interval<int, long long>;
        using CellList_t  = typename Mesh_t::cl_type;
        using CellArray_t = samurai::CellArray<dim>;

        boost::mpi::communicator world;

        // For debug
        std::ofstream logs;
        logs.open("log_" + std::to_string(_rank) + ".dat", std::ofstream::app);
        logs << "# New load balancing (load_balancing_sfc)" << std::endl;

        // SFC 1D key for cells
        std::map<SFC_key_t, typename Mesh_t::cell_t> sfc_map;
        auto sfc_keys = samurai::make_field<SFC_key_t, 1>( "keys", mesh );
        sfc_keys.fill( 0 );

        auto flags = samurai::make_field<int, 1>("rank", mesh);
        flags.fill( world.rank() );
         
        logs << fmt::format("\t\t> Computing SFC ({}) 1D indices ( cell ) ... ", _sfc.getName() ) << std::endl;

        SFC_key_t mink = std::numeric_limits<SFC_key_t>::max(), maxk = std::numeric_limits<SFC_key_t>::min();
        size_t ncells = 0;
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

            ncells ++;

        });

        assert( ncells == sfc_map.size() );

        size_t ncells_tot = 0, dc = 0;
        std::vector<size_t> nbCellsPerProc;
        std::vector<size_t> globIdx( world.size() + 1, 0 ), globIdxNew( world.size() + 1, 0 );
        boost::mpi::all_gather( world, mesh.nb_cells( Mesh_t::mesh_id_t::cells ), nbCellsPerProc );

        logs << "\t\t\t> Number of cells : " << mesh.nb_cells( Mesh_t::mesh_id_t::cells ) << std::endl;

        for(size_t i=0; i<world.size(); ++i){
            globIdx[ i + 1 ] = globIdx[ i ] + nbCellsPerProc[ i ];
            ncells_tot += nbCellsPerProc[ i ];
        }
        dc = ncells_tot / world.size();

        // load balanced globIdx
        globIdxNew[ 0 ] = 0;
        for(size_t i=0; i<world.size(); ++i){
            globIdxNew[ i + 1 ] = globIdxNew[ i ] + dc;
        }

        {
            logs << "\t\t\t> GlobalIdx : ";
            for(const auto & i : globIdx )
                logs << i << ", ";
            logs << std::endl;

            logs << "\t\t\t> GlobalIdx balanced : ";
            for(const auto & i : globIdxNew )
                logs << i << ", ";
            logs << std::endl;
            
        }   

        size_t start = 0;

        // for( size_t ip=0; ip<globIdx.size(); ++ip ){

        //     if( globIdxNew[ start ] > globIdx[ world.rank() ] ) break;
        //     start ++;
        // }
        // start = std::min( start, static_cast<size_t>( world.size() - 1 ) );
        while( globIdx[ world.rank() ] >= ( start + 1 ) * dc ){
            start ++;
        }

        logs << "Start @ rank " << start << std::endl;

        size_t count = globIdx[ world.rank() ];
        for( auto & it : sfc_map ) {

            if( count >= ( start + 1 ) * dc ){
                start ++;
                logs << "Incrementing Start @ rank " << start << ", count " << count << std::endl;
            }
            
            flags[ it.second ] = start;
            
            count ++;
        }

        // /* ---------------------------------------------------------------------------------------------------------- */
        // /* ------- Data transfer between processes ------------------------------------------------------------------ */ 
        // /* ---------------------------------------------------------------------------------------------------------- */

        // distribute cell based on boundaries & sfc key
        std::map<int, bool> comm;
        samurai::for_each_cell( mesh[Mesh_t::mesh_id_t::cells], [&](const auto & cell ) {

            if( comm.find( flags[ cell ] ) == comm.end() ) { 
                comm[ flags[ cell ] ] = true;
            }

        });

        logs << "\t\t\t> Comm required with processes : [";
        for( const auto & it : comm )
            logs << it.first << ",";
        logs << "]" << std::endl;

        // /* ---------------------------------------------------------------------------------------------------------- */
        // /* ------- Data transfer between processes ------------------------------------------------------------------ */ 
        // /* ---------------------------------------------------------------------------------------------------------- */

        CellList_t new_cl;
        std::vector<CellList_t> payload( world.size() );

        samurai::for_each_cell( mesh[Mesh_t::mesh_id_t::cells], [&]( const auto & cell ){
        
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

        for( int iproc=0; iproc<world.size(); ++iproc ){
            if( iproc == world.rank() ) continue;

            int reqExchg;
            comm.find( iproc ) != comm.end() ? reqExchg = 1 : reqExchg = 0;

            // logs << fmt::format("\t\t\t\t> send {} to # {}", reqExchg, iproc ) << std::endl;
            world.send( iproc, 17, reqExchg );

            if( reqExchg == 1 ) {
                CellArray_t to_send = { payload[ iproc ], false };
                world.send( iproc, 17, to_send );
            }
            
        }

        for( int iproc=0; iproc<world.size(); ++iproc ){

            if( iproc == world.rank() ) continue;

            int reqExchg = 0;
            world.recv( iproc, 17, reqExchg );

            // logs << fmt::format("\t\t\t\t> recv {} to # {}", reqExchg, iproc ) << std::endl;

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
    Mesh_t load_balance_impl_old( Mesh_t& mesh )
    {
        using inter_t     = samurai::Interval<int, long long>;
        using CellList_t  = typename Mesh_t::cl_type;
        using CellArray_t = samurai::CellArray<dim>;

        struct Data_t
        {
            size_t level;
            inter_t interval;
            xt::xtensor_fixed<samurai::default_config::value_t, xt::xshape<dim - 1>> indices;
            bool given;
        };

        boost::mpi::communicator world;

        // For debug
        std::ofstream logs;
        logs.open("log_" + std::to_string(_rank) + ".dat", std::ofstream::app);
        logs << "# New load balancing (load_balancing_sfc)" << std::endl;

        // SFC 1D key for cells
        std::map<SFC_key_t, Data_t> sfc_map;
         
        logs << fmt::format("\t\t> Computing SFC ({}) 1D indices (interval) ... ", _sfc.getName() ) << std::endl;

        size_t ninterval = 0;
        samurai::for_each_interval(mesh, [&]( std::size_t level, const auto & inter, const auto & index ) {
            // get Logical coordinate or first cell
            xt::xtensor_fixed<int, xt::xshape<dim>> icell;

            // first element of interval
            icell(0) = inter.start;
            for (int idim = 0; idim < dim - 1; ++idim)
            {
                icell(idim + 1) = index(idim);
            }

            // convert logical coordinate to max level logical coordinates
            for (int idim = 0; idim < dim; ++idim)
            {
                icell(idim) = icell(idim) << (mesh.max_level() - level ); // +1
            }

            // this is where things can get nasty,  we expect indices to be positive values !!
            xt::xtensor_fixed<uint32_t, xt::xshape<dim>> ijk;
            for (size_t idim = 0; idim < dim; ++idim)
            {
                ijk(idim) = static_cast<uint32_t>( icell(idim) );
            }

            sfc_map[_sfc.template getKey<dim>(ijk)] = {level, inter, index, false};

            ninterval++;
        });

        assert(ninterval == sfc_map.size());

        // Key boundaries of current process - unused for now
        SFC_key_t interval[ 2 ] = { sfc_map.begin()->first, sfc_map.rbegin()->first };
        logs << "Boundaries [" << interval[ 0 ] << ", " << interval[ 1 ] << "]" << std::endl;

        std::vector<int> load_interval;
        int my_load_i = static_cast<int>(samurai::cmptLoad<samurai::BalanceElement_t::CELL>(mesh));
        boost::mpi::all_gather(world, my_load_i, load_interval);

        // compute load to transfer to neighbour rank-1, rank+1
        int neighbour_rank_prev = -1, neighbour_rank_next = -1;
        int transfer_load_prev = 0, transfer_load_next = 0;

        // define neighbour processes for load-balancing, not geometrical neighbour !
        if (_rank > 0)
        {
            neighbour_rank_prev = _rank - 1;
            // transfer TRANSFER_PERCENT % max of difference
            transfer_load_prev = -static_cast<int>((my_load_i - load_interval[static_cast<std::size_t>(neighbour_rank_prev)])
                                                   * TRANSFER_PERCENT);
        }

        if (_rank < _ndomains - 1)
        {
            neighbour_rank_next = _rank + 1;
            // transfer TRANSFER_PERCENT % max of difference
            transfer_load_next = -static_cast<int>((my_load_i - load_interval[static_cast<std::size_t>(neighbour_rank_next)])
                                                   * TRANSFER_PERCENT);
        }

        logs << "Neighbour prev : " << neighbour_rank_prev << ", transfer of loads : " << transfer_load_prev << std::endl;
        logs << "Neighbour next : " << neighbour_rank_next << ", transfer of loads : " << transfer_load_next << std::endl;

        /* ---------------------------------------------------------------------------------------------------------- */
        /* ------- Data transfer between processes ------------------------------------------------------------------ */ 
        /* ---------------------------------------------------------------------------------------------------------- */

        CellList_t new_cl; // this will contains the final mesh of the current process

        // need send data to prev neighbour
        if (neighbour_rank_prev >= 0 && transfer_load_prev != 0)
        {
            if (transfer_load_prev < 0)
            {
                CellList_t cl_to_send;

                // give n-smallest morton keys to prev neighbour
                size_t niter_send = 0;
                for (auto iter = sfc_map.begin(); iter != sfc_map.end(); ++iter)
                {
                    if (transfer_load_prev < 0 && my_load_i > 0)
                    {
                        cl_to_send[iter->second.level][iter->second.indices].add_interval(iter->second.interval);
                        iter->second.given = true; // flag "interval" has been sent
                        my_load_i -= 1;
                        transfer_load_prev += 1;
                        niter_send++;
                    }
                    else
                    {
                        break;
                    }
                }

                CellArray_t ca_to_send = {cl_to_send, false};

                logs << "\t> Sending nbCells {" << ca_to_send.nb_cells() << "} to process : " << neighbour_rank_prev << std::endl;

                world.send(neighbour_rank_prev, 42, ca_to_send);

                // auto new_mesh = samurai::load_balance::remove( mesh, ca_to_send );
                // logs << "\t> New mesh : " << new_mesh.nb_cells() << std::endl;

                // mesh.remove( ca_to_send );
            }
            else
            {
                // need recv
                CellArray_t ca_to_rcv;
                world.recv(neighbour_rank_prev, 42, ca_to_rcv);

                logs << "\t> Receiving nbCells {" << ca_to_rcv.nb_cells() << "} from process : " << neighbour_rank_prev << std::endl;

                // mesh.merge( ca_to_rcv );
                // add to CL what we just receive
                samurai::for_each_interval(ca_to_rcv,
                                           [&](std::size_t level, const auto& interval, const auto& index)
                                           {
                                               new_cl[level][index].add_interval(interval);
                                           });
            }
        }

        if (neighbour_rank_next > 0 && transfer_load_next != 0)
        {
            if (transfer_load_next < 0)
            {
                CellList_t cl_to_send;

                // give n-smallest morton keys to prev neighbour
                size_t niter_send = 0;
                for (auto iter = sfc_map.rbegin(); iter != sfc_map.rend(); ++iter)
                {
                    if (transfer_load_next < 0 && my_load_i > 0)
                    {
                        cl_to_send[iter->second.level][iter->second.indices].add_interval(iter->second.interval);
                        iter->second.given = true;
                        my_load_i -= 1;
                        transfer_load_next += 1;
                        niter_send++;
                    }
                    else
                    {
                        break;
                    }
                }

                CellArray_t ca_to_send = {cl_to_send, false};

                logs << "\t> Sending nbCells {" << ca_to_send.nb_cells() << "} to process : " << neighbour_rank_next << std::endl;

                world.send(neighbour_rank_next, 42, ca_to_send);

                // auto new_mesh = samurai::load_balance::remove( mesh, ca_to_send );
                // logs << "\t> New mesh : " << new_mesh.nb_cells() << std::endl;

                // mesh.remove( ca_to_send );
            }
            else
            {
                // need recv
                CellArray_t ca_to_rcv;
                world.recv(neighbour_rank_next, 42, ca_to_rcv);

                logs << "\t> Receiving nbCells {" << ca_to_rcv.nb_cells() << "} from process : " << neighbour_rank_next << std::endl;

                // mesh.merge( ca_to_rcv );
                // add to CL what we just receive
                samurai::for_each_interval(ca_to_rcv,
                                           [&](std::size_t level, const auto& interval, const auto& index)
                                           {
                                               new_cl[level][index].add_interval(interval);
                                           });
            }
        }

        // last loop over the map to add what wasn't given to another process
        for (auto iter = sfc_map.rbegin(); iter != sfc_map.rend(); ++iter)
        {
            if (!iter->second.given)
            {
                new_cl[iter->second.level][iter->second.indices].add_interval(iter->second.interval);
            }
        }

        /* ---------------------------------------------------------------------------------------------------------- */
        /* ------- Construct new mesh for current process ----------------------------------------------------------- */ 
        /* ---------------------------------------------------------------------------------------------------------- */

        Mesh_t new_mesh( new_cl, mesh );

        return new_mesh;
    }
};