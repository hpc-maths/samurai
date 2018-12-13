#include <cstddef>
#include <iostream>
#include <chrono>
#include <string>

#include <mure/box.hpp>
#include <mure/mr_config.hpp>
#include <mure/level_cell_list.hpp>
//#include <mure/level_cell_array_other.hpp>
//#include <mure/level_cell_array_other_nolist.hpp>
#include <mure/level_cell_array.hpp>

#include <xtensor/xview.hpp>

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

int main(int argc, char* argv[])
{
    constexpr std::size_t N_run = 5;
    constexpr std::size_t dim = 3;
    using Config = mure::MRConfig<dim>;
    using coord_index_t = Config::coord_index_t;
    const coord_index_t cross_size      = std::stoull(argv[1]);
    const coord_index_t cross_tickness  = std::stoull(argv[2]);
    const coord_index_t box_size = cross_size + cross_tickness;

    std::cout << "Creating level cell list..." << std::endl;
    mure::Box<coord_index_t, dim> box{{0, 0, 0}, {box_size, box_size, box_size}};
    mure::LevelCellList<Config> dcl;
    dcl.extend(xt::view(box.min_corner(), xt::drop(0)), xt::view(box.max_corner(), xt::drop(0)));
    
    std::size_t cnt = 0;
    for (Config::coord_index_t i = 0; i < cross_size; ++i)
    {
        if (i < (cross_size - cross_tickness)/2 || i >= (cross_size + cross_tickness)/2)
        {
            dcl[{i,i}].add_interval({i, i+cross_tickness+1, cnt++});
            dcl[{i,i}].add_interval({cross_size-i-1, cross_size-i+cross_tickness, cnt++});
            dcl[{cross_size-i-1,i}].add_interval({i, i+1+cross_tickness, cnt++});
            dcl[{cross_size-i-1,i}].add_interval({cross_size-i-1, cross_size-i+cross_tickness, cnt++});
        }
    }

    for (std::size_t i = 0; i < N_run; ++i)
    {
        std::cout << "Run #" << i << std::endl;
        
        tic();
        mure::LevelCellArray<Config> dca(dcl);
        auto duration = toc();
        std::cout << "\tConvertion in " << duration << "s" << std::endl;

        std::size_t cnt = 0;
        auto counter = [&cnt] (auto index, auto interval) { cnt += index[0] + index[1] + interval.index; };

        tic();
        dca.for_each_interval_in_x(counter);
        duration = toc();
        std::cout << "\tTraversal in " << duration << "s (" << cnt << ")" << std::endl;
    }

    return 0;
}
