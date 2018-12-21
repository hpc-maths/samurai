#include <iostream>
#include <chrono>
#include <vector>

#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include "xtensor/xnoalias.hpp"

#include <mure/box.hpp>
#include <mure/level_cell_array.hpp>
#include <mure/mr_config.hpp>

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

int main()
{
    constexpr size_t dim = 2;
    constexpr size_t level = 12;
    constexpr size_t nrun = 10;
    constexpr size_t end = std::pow(2, level);
    using Config = mure::MRConfig<dim>;
    mure::Box<int, dim> box({0, 0}, {end, end});


    mure::LevelCellArray<Config> lca = {box};

    auto array_1 = xt::xtensor<double, 1>::from_shape({lca.nb_cells()});
    array_1.fill(1.);
    auto array_2 = xt::xtensor<double, 1>::from_shape({lca.nb_cells()});

    std::cout << "Mure:\n";
    for(size_t n=0; n<nrun; ++n)
    {
        tic();
        lca.for_each_block([&](auto load, auto restore){
            auto view = load(array_1);
            auto tmp = xt::xtensor<double, 2>::from_shape(view.shape());

            xt::noalias(xt::view(tmp, xt::range(1, view.shape()[0]-1), xt::range(1, view.shape()[1]-1))) = 
                2*xt::view(view, xt::range(1, view.shape()[0]-1), xt::range(1, view.shape()[1]-1))
                -   xt::view(view, xt::range(2, view.shape()[0]), xt::range(1, view.shape()[1]-1))
                -   xt::view(view, xt::range(0, view.shape()[0]-2), xt::range(1, view.shape()[1]-1))
                -   xt::view(view, xt::range(1, view.shape()[0]-1), xt::range(2, view.shape()[1]))
                -   xt::view(view, xt::range(1, view.shape()[0]-1), xt::range(0, view.shape()[1]-2));

            restore(array_2, tmp);
        });

        auto duration = toc();

        std::cout << "\tRun #" << n << " in " << duration << "s (" << std::accumulate(array_2.begin(), array_2.end(), 0) << ")\n";
    }

    auto view = xt::reshape_view(array_1, {end, end});
    auto array_3 = xt::xtensor<double, 1>::from_shape({lca.nb_cells()});
    auto array_3_ = xt::reshape_view(array_3, {end, end});

    std::cout << "xtensor:\n";
    for(size_t n=0; n<nrun; ++n)
    {
        tic();

        xt::noalias(xt::view(array_3_, xt::range(1, array_3_.shape()[0]-1), xt::range(1, array_3_.shape()[1]-1))) = 
                2*xt::view(view, xt::range(1, view.shape()[0]-1), xt::range(1, view.shape()[1]-1))
            -   xt::view(view, xt::range(2, view.shape()[0]), xt::range(1, view.shape()[1]-1))
            -   xt::view(view, xt::range(0, view.shape()[0]-2), xt::range(1, view.shape()[1]-1))
            -   xt::view(view, xt::range(1, view.shape()[0]-1), xt::range(2, view.shape()[1]))
            -   xt::view(view, xt::range(1, view.shape()[0]-1), xt::range(0, view.shape()[1]-2));

        auto duration = toc();
        std::cout << "\tRun #" << n << " in " << duration << "s (" << std::accumulate(array_3.begin(), array_3.end(), 0) << ")\n";
    }

    std::vector<double> vector_1(lca.nb_cells(), 1.);
    std::vector<double> vector_2(lca.nb_cells(), 0  );

    std::cout << "std::vector:\n";
    for(size_t n=0; n<nrun; ++n)
    {
        tic();
        for(size_t j=1; j<end-1; ++j)
        {
            for(size_t i=1; i<end-1; ++i)
            {
                vector_2[i + j*end] = 2*vector_1[i + j*end]
                                    -   vector_1[i+1 + j*end]
                                    -   vector_1[i-1 + j*end]
                                    -   vector_1[i + (j+1)*end]
                                    -   vector_1[i + (j-1)*end];
            }
        }

        auto duration = toc();
        std::cout << "\tRun #" << n << " in " << duration << "s (" << std::accumulate(vector_2.begin(), vector_2.end(), 0) << ")\n";
    }
    // for(size_t j=0; j<end; ++j)
    // {
    //     for(size_t i=0; i<end; ++i)
    //     {
    //         std::cout << vector_2[i + j*end] << " ";
    //     }
    //     std::cout << "\n";
    // }
    // std::cout << array_3_ << "\n";
    // std::cout << array_1 << "\n";
    // xt::xarray<double> a = {{1, 2, 3},
    //                         {4, 5, 6},
    //                         {7, 8, 9}};
    
    // auto t = xt::xtensor<int, 2>::from_shape({3, 2});
    // for(size_t i = 0; i < t.shape()[0]; ++i)
    // {
    //     xt::view(t, i, xt::all()) = xt::view(a, i, xt::range(1, 2));
    // }
    // t[{0, 1}] = 1000;
    // std::cout << a << "\n";
}