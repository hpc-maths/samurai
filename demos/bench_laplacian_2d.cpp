#include <iostream>
#include <chrono>
#include <vector>
#include <string>

#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include "xtensor/xnoalias.hpp"

#include <samurai/samurai.hpp>

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
    // Default parameters
    constexpr std::size_t dim = 2;
    std::size_t level = 12;
    std::size_t nrun = 10;

    // Command-line parameters
    if (argc > 1)
        level = std::stoull(argv[1]);

    if (argc > 2)
        nrun = std::stoull(argv[2]);

    const size_t end = std::pow(2, level) +2;
    using Config = samurai::MRConfig<dim>;
    samurai::Box<double, dim> box({0, 0}, {1, 1});
    samurai::Mesh<Config> mesh{box, 0, level};

    samurai::Field<Config> array_1{"array_1", mesh};
    array_1.array().fill(1.);
    samurai::Field<Config> array_2{"array_2", mesh};
    array_2.array().fill(0.);

    std::cout << "Samurai:\n";
    for(size_t n=0; n<nrun; ++n)
    {
        tic();
        auto subset = samurai::intersection(mesh[samurai::MeshType::cells][level],
                                         mesh[samurai::MeshType::cells][level]);

        subset([&](auto &index, auto &interval, auto)
        {
            auto i = interval[0];
            auto j = index[0];
            xt::noalias(array_2(level, i, j)) = 2.*array_1(level, i , j)
                                              - array_1(level, i - 1 , j)
                                              - array_1(level, i + 1 , j)
                                              - array_1(level, i     , j - 1)
                                              - array_1(level, i     , j + 1);
        });
        auto duration = toc();

        std::cout << "\tRun #" << n << " in " << duration << "s (" << xt::sum(array_2.array()) << ")\n";
    }


    auto array_3 = xt::xtensor<double, 1>::from_shape({end*end});
    auto array_3_ = xt::reshape_view(array_3, {end, end});
    auto array_4 = xt::xtensor<double, 1>::from_shape({end*end});
    auto array_4_ = xt::reshape_view(array_4, {end, end});
    array_4_.fill(1.);

    std::cout << "xtensor:\n";
    for(size_t n=0; n<nrun; ++n)
    {
        tic();

        xt::noalias(xt::view(array_3_, xt::range(1, array_3_.shape()[0]-1), xt::range(1, array_3_.shape()[1]-1))) =
                2*xt::view(array_4_, xt::range(1, array_4_.shape()[0]-1), xt::range(1, array_4_.shape()[1]-1))
            -   xt::view(array_4_, xt::range(2, array_4_.shape()[0]), xt::range(1, array_4_.shape()[1]-1))
            -   xt::view(array_4_, xt::range(0, array_4_.shape()[0]-2), xt::range(1, array_4_.shape()[1]-1))
            -   xt::view(array_4_, xt::range(1, array_4_.shape()[0]-1), xt::range(2, array_4_.shape()[1]))
            -   xt::view(array_4_, xt::range(1, array_4_.shape()[0]-1), xt::range(0, array_4_.shape()[1]-2));

        auto duration = toc();
        std::cout << "\tRun #" << n << " in " << duration << "s (" << xt::sum(array_3) << ")\n";
    }

    std::vector<double> vector_1(end*end, 1.);
    std::vector<double> vector_2(end*end, 0  );

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
        std::cout << "\tRun #" << n << " in " << duration << "s (" << std::accumulate(vector_2.begin(), vector_2.end(), 0.) << ")\n";
    }
}
