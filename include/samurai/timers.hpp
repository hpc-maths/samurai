#pragma once

#include <cassert>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <vector>

#include "assert_log_trace.hpp"

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#else
#include <chrono>
#include <sys/time.h>
#include <time.h>
#endif

namespace samurai
{
    struct Timer
    {
#ifdef SAMURAI_WITH_MPI
        double start, elapsed;
#else
        std::chrono::time_point<std::chrono::high_resolution_clock> start;
        std::chrono::microseconds elapsed;
#endif
        uint32_t ntimes;
    };

    class Timers
    {
      public:

        Timers()
        {
        }

        ~Timers() = default;

        inline auto getElapsedTime(const std::string& tname) const
        {
            SAMURAI_ASSERT(_times.find(tname) != _times.end(), "[Timers::getElapsedTime] Requested timer not found '" + tname + "' !");

            if (_times.find(tname) != _times.end())
            {
                return _times.at(tname).elapsed;
            }
            else
            {
#ifdef SAMURAI_WITH_MPI
                return 0.0;
#else
                return std::chrono::microseconds(0);
#endif
            }
        }

        inline void start(const std::string& tname)
        {
            if (_times.find(tname) != _times.end())
            {
                _times.at(tname).start = _getTime();
            }
            else
            {
                _times.emplace(std::make_pair(tname, Timer{_getTime(), _zero_duration(), 0}));
            }
        }

        inline void stop(const std::string& tname)
        {
            SAMURAI_ASSERT(_times.find(tname) != _times.end(), "[Timers::stop] Requested timer not found '" + tname + "' !");
            // if (_times.find(tname) != _times.end())
            // {
#ifdef SAMURAI_WITH_MPI
            _times.at(tname).elapsed += (_getTime() - _times.at(tname).start);
#else
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(_getTime() - _times.at(tname).start);
            _times.at(tname).elapsed += duration;
#endif
            _times.at(tname).ntimes++;
            // }
            // else
            // {
            //     _times.emplace(std::make_pair(tname, Timer{0.0, 0.0, 0}));
            // }
        }

#ifdef SAMURAI_WITH_MPI
        void print() const
        {
            boost::mpi::communicator world;

            int setwSize = 20;

            if (world.rank() == 0)
            {
                std::cout << "\n\t> [Master] Timers " << std::endl;
                std::cout << "\t" << std::setw(setwSize) << "Name " << std::setw(setwSize) << "Min time (s)" << std::setw(8) << ""
                          << std::setw(setwSize) << "Max time (s)" << std::setw(8) << "" << std::setw(setwSize) << "Ave time (s)"
                          << std::setw(setwSize) << "Std dev" << std::setw(setwSize) << "Calls" << std::endl;
            }

            for (const auto& timer : _times)
            {
                int minrank = -1, maxrank = -1;
                double min = std::numeric_limits<double>::max(), max = std::numeric_limits<double>::min();
                double ave = 0., std = 0.;

                std::vector<double> all(static_cast<std::size_t>(world.size()), 0.0);
                boost::mpi::all_gather(world, timer.second.elapsed, all);

                for (size_t iproc = 0; iproc < all.size(); ++iproc)
                {
                    if (all[iproc] < min)
                    {
                        min     = all[iproc];
                        minrank = static_cast<int>(iproc);
                    }

                    if (all[iproc] > max)
                    {
                        max     = all[iproc];
                        maxrank = static_cast<int>(iproc);
                    }

                    ave += all[iproc];
                }

                ave /= static_cast<double>(world.size());

                for (size_t iproc = 0; iproc < all.size(); ++iproc)
                {
                    std = (all[iproc] - ave) * (all[iproc] - ave);
                }
                std /= static_cast<double>(world.size());
                std = std::sqrt(std);

                // boost::mpi::reduce( world, timer.second.elapsed, min, boost::mpi::minimum<double>(), root );
                // boost::mpi::reduce( world, timer.second.elapsed, max, boost::mpi::maximum<double>(), root );
                // boost::mpi::reduce( world, timer.second.elapsed, ave, std::plus<double>(), root );

                if (world.rank() == 0)
                {
                    std::cout << std::fixed << std::setprecision(5) << "\t" << std::setw(setwSize) << timer.first << std::setw(setwSize)
                              << min << std::setw(2) << "[" << std::setw(4) << minrank << std::setw(2) << "]" << std::setw(setwSize) << max
                              << std::setw(2) << "[" << std::setw(4) << maxrank << std::setw(2) << "]" << std::setw(setwSize) << ave
                              << std::setw(setwSize) << std << std::setw(setwSize) << timer.second.ntimes << std::endl;
                }
            }
        }

#else
        void print() const
        {
            std::chrono::microseconds total_runtime(0);
            std::chrono::microseconds total_measured(0);
            std::size_t max_name_length = 0;
            bool has_total_runtime      = false;
            for (const auto& timer : _times)
            {
                if (timer.first == "total runtime")
                {
                    has_total_runtime = true;
                    total_runtime     = timer.second.elapsed;
                }
                else
                {
                    total_measured += timer.second.elapsed;
                    max_name_length = std::max(max_name_length, timer.first.length());
                }
            }
            std::chrono::microseconds total = has_total_runtime ? total_runtime : total_measured;

            std::vector<std::pair<std::string, Timer>> sorted_data(_times.begin(), _times.end());
            std::sort(sorted_data.begin(),
                      sorted_data.end(),
                      [](const std::pair<std::string, Timer>& a, const std::pair<std::string, Timer>& b)
                      {
                          return a.second.elapsed > b.second.elapsed;
                      });

            double total_perc = 0.0;
            // std::cout << "\n\t> [Process] Timers " << std::endl;
            int setwSizeName = static_cast<int>(max_name_length) + 4;
            int setwSizeData = 16;

            // std::cout << "\t";
            std::cout << std::setw(setwSizeName) << " ";
            std::cout << std::setw(setwSizeData) << "Elapsed (s)";
            std::cout << std::setw(setwSizeData) << "Fraction (%)";
            // std::cout << std::setw(setwSizeData) << "Calls";
            std::cout << std::endl;

            std::cout << std::fixed;
            for (const auto& timer : sorted_data)
            {
                if (timer.first != "total runtime")
                {
                    auto elapsedInSeconds = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(timer.second.elapsed);
                    // std::cout << "\t";
                    std::cout << std::setw(setwSizeName) << timer.first;
                    std::cout << std::setw(setwSizeData) << std::setprecision(3) << elapsedInSeconds.count();
                    std::cout << std::setw(setwSizeData) << std::setprecision(1) << _percent(timer.second.elapsed, total);
                    // std::cout << std::setw(setwSizeData) << timer.second.ntimes;
                    std::cout << std::endl;
                    total_perc += timer.second.elapsed * 100.0 / total;
                }
            }

            for (const auto& timer : _times)
            {
                if (timer.first == "total runtime")
                {
                    std::string msg = "----------------";
                    std::cout << std::setw(setwSizeName) << msg /*<< std::setw(setwSizeData) << msg << std::setw(setwSizeData) << msg
                              << std::setw(setwSizeData)*/
                              << std::endl;

                    auto untimed          = total_runtime - total_measured;
                    auto untimedInSeconds = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(untimed);
                    // std::cout << "\t";
                    std::cout << std::setw(setwSizeName) << "(untimed)";
                    std::cout << std::setw(setwSizeData) << std::setprecision(3) << untimedInSeconds.count();
                    std::cout << std::setw(setwSizeData) << std::setprecision(1) << _percent(untimed, total);
                    std::cout << std::endl;

                    std::cout << std::setw(setwSizeName) << msg << std::setw(setwSizeData) << msg << std::setw(setwSizeData) << msg
                              << std::setw(setwSizeData) << std::endl;

                    auto totalInSeconds = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(timer.second.elapsed);
                    // std::cout << "\t";
                    std::cout << std::setw(setwSizeName) << timer.first;
                    std::cout << std::setw(setwSizeData) << std::setprecision(3) << totalInSeconds.count();
                    std::cout << std::setw(setwSizeData) << std::setprecision(1) << 100.0;
                    std::cout << std::endl;
                    total_perc += _percent(timer.second.elapsed, total);
                }
            }

            if (!has_total_runtime)
            {
                auto totalInSeconds = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(total_measured);
                std::cout << std::setw(setwSizeName) << "Total";
                std::cout << std::setw(setwSizeData) << std::setprecision(3) << totalInSeconds.count();
                std::cout << std::setw(setwSizeData) << std::setprecision(1) << total_perc;
                std::cout << std::endl;
            }

            std::cout << std::endl;
        }
#endif

      private:

        std::map<std::string, Timer> _times;

#ifdef SAMURAI_WITH_MPI
        inline double _getTime() const
        {
            return MPI_Wtime();
        }

        inline double _zero_duration() const
        {
            return 0.0;
        }
#else
        inline std::chrono::time_point<std::chrono::high_resolution_clock> _getTime() const
        {
            // timeval now;
            // SAMURAI_ASSERT(-1 != gettimeofday(&now, 0), "[Timers::_getTime()] Error getting timeofday !");
            // return double(now.tv_sec) + (double(now.tv_usec) * 1e-6);
            // double durationInmicroseconds = chrono::duration_cast<chrono::duration<double, std::milli>>(_elapsed_stop -
            // _elapsed_start).count();
            return std::chrono::high_resolution_clock::now();
        }

        inline std::chrono::microseconds _zero_duration() const
        {
            return std::chrono::microseconds(0);
        }
#endif
        inline double _percent(const std::chrono::microseconds& value, const std::chrono::microseconds& total) const
        {
            return static_cast<double>(value.count() * 100) / static_cast<double>(total.count());
        }
    };

    namespace times
    {

        static Timers timers;

    }
}
