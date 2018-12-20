#include <iostream>

#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

#include <mure/box.hpp>
#include <mure/level_cell_array.hpp>
#include <mure/mr_config.hpp>

int main()
{
    constexpr size_t dim = 2;
    constexpr size_t level = 1;
    constexpr size_t end = std::pow(2, 2);
    using Config = mure::MRConfig<dim>;
    mure::Box<int, dim> box({0, 0, 0}, {end, end, end});


    mure::LevelCellArray<Config> lca = {box};

    auto array_1 = xt::xtensor<double, 1>::from_shape({lca.nb_cells()});
    array_1.fill(1.);
    auto array_2 = xt::xtensor<double, 1>::from_shape({lca.nb_cells()});

    lca.for_each_block([&](auto load, auto restore){
        auto view = load(array_1);
        auto tmp = xt::xtensor<double, 2>::from_shape(view.shape());

        xt::view(tmp, xt::range(1, view.shape()[0]-1), xt::range(1, view.shape()[1]-1)) = 
              2*xt::view(view, xt::range(1, view.shape()[0]-1), xt::range(1, view.shape()[1]-1));
            -   xt::view(view, xt::range(2, view.shape()[0]), xt::range(1, view.shape()[1]-1))
            -   xt::view(view, xt::range(0, view.shape()[0]-2), xt::range(1, view.shape()[1]-1))
            -   xt::view(view, xt::range(1, view.shape()[0]-1), xt::range(2, view.shape()[1]))
            -   xt::view(view, xt::range(1, view.shape()[0]-1), xt::range(0, view.shape()[1]-2));

        restore(array_2, tmp);
    });
    std::cout << xt::reshape_view(array_2, {4, 4}) << "\n";

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