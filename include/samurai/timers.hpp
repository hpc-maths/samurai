#pragma once

#include <algorithm>
#include <cassert>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include <fmt/printf.h>

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

            // Compute dynamic widths
            std::size_t max_name_length = 0;
            for (const auto& kv : _times)
            {
                max_name_length = std::max<std::size_t>(max_name_length, kv.first.size());
            }
            const int nameWidth  = static_cast<int>(std::max<std::size_t>(24, max_name_length + 2));
            const int timeWidth  = 16;
            const int rankWidth  = 7; // prints like "[  12]"
            const int callsWidth = 10;

            if (world.rank() == 0)
            {
                fmt::printf("\n > [Master] Timers \n");
                fmt::printf(" %-*s%*s%*s%*s%*s%*s%*s%*s\n",
                            nameWidth,
                            "Name",
                            timeWidth,
                            "Min time (s)",
                            rankWidth,
                            "[r]",
                            timeWidth,
                            "Max time (s)",
                            rankWidth,
                            "[r]",
                            timeWidth,
                            "Ave time (s)",
                            timeWidth,
                            "Std dev",
                            callsWidth,
                            "Calls");
            }

            for (const auto& timer : _times)
            {
                int minrank = -1, maxrank = -1;
                double min = std::numeric_limits<double>::max(), max = std::numeric_limits<double>::lowest();
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

                double sqsum = 0.0;
                for (size_t iproc = 0; iproc < all.size(); ++iproc)
                {
                    const double d = all[iproc] - ave;
                    sqsum += d * d;
                }
                std = std::sqrt(sqsum / static_cast<double>(world.size()));

                // boost::mpi::reduce( world, timer.second.elapsed, min, boost::mpi::minimum<double>(), root );
                // boost::mpi::reduce( world, timer.second.elapsed, max, boost::mpi::maximum<double>(), root );
                // boost::mpi::reduce( world, timer.second.elapsed, ave, std::plus<double>(), root );

                if (world.rank() == 0)
                {
                    fmt::printf(" %-*s%*.5f%*s%*.5f%*s%*.5f%*.5f%*u\n",
                                nameWidth,
                                timer.first,
                                timeWidth,
                                min,
                                rankWidth,
                                "[" + std::to_string(minrank) + "]",
                                timeWidth,
                                max,
                                rankWidth,
                                "[" + std::to_string(maxrank) + "]",
                                timeWidth,
                                ave,
                                timeWidth,
                                std,
                                callsWidth,
                                timer.second.ntimes);
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
            // std::cout << "\n > [Process] Timers " << std::endl;
            int setwSizeName = static_cast<int>(max_name_length) + 2;
            int setwSizeData = 12;

            // std::cout << " ";
            fmt::printf("%*s%*s%*s\n", setwSizeName, " ", setwSizeData, "Elapsed (s)", setwSizeData, "Fraction (%)");

            for (const auto& timer : sorted_data)
            {
                if (timer.first != "total runtime")
                {
                    auto elapsedInSeconds = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(timer.second.elapsed);
                    // std::cout << " ";
                    fmt::printf("%*s%*.3f%*.1f\n",
                                setwSizeName,
                                timer.first,
                                setwSizeData,
                                elapsedInSeconds.count(),
                                setwSizeData,
                                _percent(timer.second.elapsed, total));
                    // std::cout << std::setw(setwSizeData) << timer.second.ntimes;
                    total_perc += timer.second.elapsed * 100.0 / total;
                }
            }

            for (const auto& timer : _times)
            {
                if (timer.first == "total runtime")
                {
                    std::string msg = "--------";
                    fmt::printf("%*s\n", setwSizeName, msg);

                    auto untimed          = total_runtime - total_measured;
                    auto untimedInSeconds = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(untimed);
                    // std::cout << " ";
                    fmt::printf("%*s%*.3f%*.1f\n",
                                setwSizeName,
                                "(untimed)",
                                setwSizeData,
                                untimedInSeconds.count(),
                                setwSizeData,
                                _percent(untimed, total));

                    fmt::printf("%*s%*s%*s\n", setwSizeName, msg, setwSizeData, msg, setwSizeData, msg);

                    auto totalInSeconds = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(timer.second.elapsed);
                    // std::cout << " ";
                    fmt::printf("%*s%*.3f%*.1f\n", setwSizeName, timer.first, setwSizeData, totalInSeconds.count(), setwSizeData, 100.0);
                    total_perc += _percent(timer.second.elapsed, total);
                }
            }

            if (!has_total_runtime)
            {
                auto totalInSeconds = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(total_measured);
                fmt::printf("%*s%*.3f%*.1f\n", setwSizeName, "Total", setwSizeData, totalInSeconds.count(), setwSizeData, total_perc);
            }

            fmt::printf("\n");
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
