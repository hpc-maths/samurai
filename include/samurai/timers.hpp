#pragma once

#include <vector>
#include <map>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <limits>

#include "assertLogTrace.hpp"

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#else
#include <time.h>
#include <sys/time.h>
#endif

struct Timer{
    double start, elapsed;
};

class Timers {

    public:

        Timers() {}

        ~ Timers() = default;

        inline double getElapsedTime( const std::string & tname ) const {
            SAMURAI_ASSERT( _times.find( tname ) != _times.end(), "[Timers::getElapsedTime] Requested timer not found '" + tname + "' !" );

            if( _times.find( tname ) != _times.end() ){
                return _times.at( tname ).elapsed;
            }else{
                return 0.0;
            }
        }

        inline void start( const std::string & tname ) { 
            if( _times.find( tname ) != _times.end() ){
                _times.at( tname ).start = _getTime();
            } else {
                _times.emplace( std::make_pair( tname, Timer { _getTime(), 0.0} ) );
            }
        }

        inline void stop( const std::string & tname ) {
            SAMURAI_ASSERT( _times.find( tname ) != _times.end(), "[Timers::stop] Requested timer not found '" + tname + "' !" );
            if( _times.find( tname ) != _times.end() ){
                _times.at( tname ).elapsed += ( _getTime() - _times.at( tname ).start );
            } else {
                _times.emplace( std::make_pair( tname, Timer {0.0, 0.0} ) );
            }
        }

#ifdef SAMURAI_WITH_MPI
        void print() const{

            boost::mpi::communicator world;

            double total_ptime = 0.0;
            for( const auto &timer : _times )
                total_ptime += timer.second.elapsed;

            double total_perc = 0.0;
            int setwSize = 20;

            if( world.rank() == 0 ){
                std::cout << "\n\t> [Master] Timers " << std::endl;
                std::cout << "\t" << std::setw(setwSize) << "Name " << std::setw(setwSize) << "Min time (s)"
                          << std::setw(8) << "" << std::setw(setwSize) << "Max time (s)"
                          << std::setw(8) << "" << std::setw(setwSize) << "Ave time (s)"
                          << std::setw(setwSize) << "Std dev"
                          << std::endl;
            }

            for (const auto &timer: _times) {
                int minrank = -1, maxrank = -1;
                double min = std::numeric_limits<double>::max(), max = std::numeric_limits<double>::min();
                double ave = 0., std = 0.;

                std::vector<double> all( static_cast<std::size_t>( world.size() ), 0.0 );
                boost::mpi::all_gather( world, timer.second.elapsed, all );

                for( size_t iproc = 0; iproc < all.size(); ++iproc ){
                    if( all[ iproc ] < min ) {
                        min     = all[ iproc ];
                        minrank = static_cast<int>( iproc );
                    }

                    if( all[ iproc ] > max ) {
                        max     = all[ iproc ];
                        maxrank = static_cast<int>( iproc );
                    }

                    ave += timer.second.elapsed;
                }

                ave /= static_cast<double>( world.size() );

                for( size_t iproc = 0; iproc < all.size(); ++iproc ){
                    std = ( all[ iproc ] - ave ) * ( all[ iproc ] - ave );
                }
                std /= static_cast<double>( world.size() );
                std = std::sqrt( std );

                // boost::mpi::reduce( world, timer.second.elapsed, min, boost::mpi::minimum<double>(), root );
                // boost::mpi::reduce( world, timer.second.elapsed, max, boost::mpi::maximum<double>(), root );
                // boost::mpi::reduce( world, timer.second.elapsed, ave, std::plus<double>(), root );

                if( world.rank() == 0 ){
                    std::cout << std::fixed << std::setprecision(5) << "\t" << std::setw(setwSize) << timer.first 
                              << std::setw(setwSize) << min << std::setw(2) << "[" << std::setw(4) << minrank << std::setw(2) << "]"
                              << std::setw(setwSize) << max << std::setw(2) << "[" << std::setw(4) << maxrank << std::setw(2) << "]"
                              << std::setw(setwSize) << ave << std::setw(setwSize) << std 
                              << std::endl;
                }

                total_perc += timer.second.elapsed * 100.0 / total_ptime;
            }

            if( world.rank() == 0 ){
                std::string msg = "------------------------";
                std::cout << "\t" << std::setw(setwSize) << msg << std::setw(setwSize) << msg
                        << std::setw(setwSize) << msg << std::endl;
                std::cout << "\t" << std::setw(setwSize) << "Total" << std::setw(setwSize) << total_ptime
                        << std::setw(setwSize) << total_perc << std::endl;

                std::flush(std::cout);
            }
        }

#else
        void print() const{

            double total_ptime = 0.0;
            for( const auto &timer : _times )
                total_ptime += timer.second.elapsed;

            double total_perc = 0.0;
            std::cout << "\n\t> [Process] Timers " << std::endl;
            int setwSize = 28;

            std::cout << "\t" << std::setw(setwSize) << "Name " << std::setw(setwSize) << " Total Elapsed (s)"
                        << std::setw(setwSize) << " Fraction (%) " << std::endl;
            for (const auto &timer: _times) {
                std::cout << "\t" << std::setw(setwSize) << timer.first << std::setw(setwSize)
                            << timer.second.elapsed << std::setw(setwSize) << std::setprecision(5)
                            << timer.second.elapsed * 100.0 / total_ptime
                            << std::endl;
                total_perc += timer.second.elapsed * 100.0 / total_ptime;
            }

            // std::string msg = "------------------------";
            // std::cout << "\t" << std::setw(setwSize) << msg << std::setw(setwSize) << msg
            //         << std::setw(setwSize) << msg << std::endl;
            // std::cout << "\t" << std::setw(setwSize) << "Total" << std::setw(setwSize) << total_ptime
            //         << std::setw(setwSize) << total_perc << std::endl;

            std::flush(std::cout);
        }
#endif
    private:
        std::map<std::string, Timer> _times;

#ifdef SAMURAI_WITH_MPI
        inline double _getTime() const { return MPI_Wtime(); }
#else
        inline double _getTime() const { 
            timeval_t now;
            SAMURAI_ASSERT( -1 != gettimeofday(&now, 0), "[Timers::_getTime()] Error getting timeofday !" );
            return double( now.tv_sec ) +  ( double( now.tv_usec ) * 1e-6 );
        }
#endif

};