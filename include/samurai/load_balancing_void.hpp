/**
 * Empty class, used for test to compare with and without load balancing. 
 *  
*/

#pragma once

#include <map>
#include "load_balancing.hpp"

template<int dim>
class Void_LoadBalancer: public samurai::LoadBalancer<Void_LoadBalancer<dim>> {

    private:
        int _ndomains;
        int _rank;

    public:

        Void_LoadBalancer() {
#ifdef SAMURAI_WITH_MPI
            boost::mpi::communicator world;
            _ndomains = world.size();
            _rank     = world.rank();
#else
            _ndomains = 1;
            _rank     = 0;
#endif
        }

        inline std::string getName() const { return "Void_LB"; }

        template<class Mesh_t>
        Mesh_t reordering_impl( Mesh_t & mesh ) { return mesh; }

        template<class Mesh_t>
        Mesh_t load_balance_impl( Mesh_t & mesh ){ return mesh; }

};