#include <cstddef>
#include <iostream>
#include <chrono>
#include <string>

#include "DynamicLevelData.h"
#include "LevelData.h"

/// Timer used in tic & toc
auto tic_timer = std::chrono::high_resolution_clock::now();

/// Launching the timer
void tic()
{
    tic_timer = std::chrono::high_resolution_clock::now();
}

/// Stopping the timer and returning the duration in seconds
double toc()
{
    const auto toc_timer = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> time_span = toc_timer - tic_timer;
    return time_span.count();
}

template <std::size_t N>
struct Carac
{
    using Index = std::size_t;
    static constexpr std::size_t dim = N;
    using TC = long long int;
};

int main(int argc, char* argv[])
{
    constexpr std::size_t N_run = 5;
    constexpr std::size_t dim = 3;
    using Config = Carac<dim>;

    using coord_index_t = Config::TC;
    using dcl_t = DynamicLevelData<Config>;
    using dca_t = LevelData<Config>;

    const coord_index_t cross_size      = std::stoull(argv[1]);
    const coord_index_t cross_tickness  = std::stoull(argv[2]);
    const coord_index_t box_size = cross_size + cross_tickness;

    for (std::size_t i = 0; i < N_run; ++i)
    {
        std::cout << "Run #" << i << std::endl;

        tic();
        dcl_t dcl;
        dcl_t::YzCoords min_corner; min_corner.fill(0);
        dcl_t::YzCoords max_corner; max_corner.fill(cross_size + cross_tickness);
        dcl.extend(min_corner, max_corner);

        Config::Index cnt = 0;
        for (coord_index_t i = 0; i < cross_size; ++i)
        {
            if (i < (cross_size - cross_tickness)/2 || i >= (cross_size + cross_tickness)/2)
            {
                dcl.add_x_range({i,i}, i, i+cross_tickness+1);
                dcl.add_x_range({i,i}, cross_size-i-1, cross_size-i+cross_tickness);
                dcl.add_x_range({cross_size-i-1,i}, i, i+cross_tickness+1);
                dcl.add_x_range({cross_size-i-1,i}, cross_size-i-1, cross_size-i+cross_tickness);
            }
        }
        auto duration = toc();
        std::cout << "\tCreating level cell list in " << duration << "s" << std::endl;

        tic();
        dca_t dca(dcl);
        duration = toc();
        std::cout << "\tConvertion in " << duration << "s" << std::endl;

        std::size_t interval_cnt = 0;
        std::size_t coord_sum = 0;
        std::size_t interval_sum = 0;
        std::size_t index_sum = 0;
        auto counter = [&] (auto index, auto interval) { ++interval_cnt; coord_sum += index[0] + index[1]; interval_sum += interval.beg + interval.end; index_sum += interval.x0_index; };

        tic();
        dca.for_each_x_range(counter);
        duration = toc();
        std::cout << "\tTraversal in " << duration << "s (#interval=" << interval_cnt << ", sum(yz)=" << coord_sum << ", sum([a,b[)=" << interval_sum << ", sum(index)=" << index_sum << ")" << std::endl;
    }
    return 0;
}
