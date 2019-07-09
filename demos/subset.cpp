// #include <mure/intervals_operator.hpp>
#include <mure/level_cell_array.hpp>
#include <mure/level_cell_list.hpp>
#include <mure/mr_config.hpp>
// #include <mure/subset.hpp>
#include <mure/subset/subset_op.hpp>

int main()
{
    constexpr size_t dim = 2;

    mure::LevelCellList<dim, 0> lcl1;
    // lcl1[{}].add_interval({-5, 5});
    lcl1[{0}].add_interval({-5, 5});

    mure::LevelCellList<dim, 1> lcl2;
    // lcl2[{}].add_interval({-2, 3});
    lcl2[{0}].add_interval({1, 3});
    lcl2[{1}].add_interval({1, 3});

    mure::LevelCellArray<dim, 0> lca1{lcl1};
    mure::LevelCellArray<dim, 1> lca2{lcl2};

    // // std::cout << lca1 << "\n";
    // // std::cout << lca2 << "\n";
    // auto expr = mure::intersection(lca1, translate(lca1));
    // // auto expr = mure::intersection(lca1, lca2);
    // expr([](auto &index, auto &interval) {
    //     std::cout << index << " " << interval << "\n";
    // });

    auto tmp = mure::intersection(lca1, lca1);
    auto expr = mure::intersection(translate(lca1), tmp);
    // auto expr = mure::intersection(translate(lca1), lca1);
    // auto expr = mure::intersection(lca1, translate(translate(lca1)));
    // auto expr = mure::intersection(lca1, mure::intersection(lca1, lca1));
    auto expr1 = expr.on<2>();
    expr1([](auto &index, auto &interval) {
        std::cout << index << " " << interval << "\n";
    });

    // mure::LevelCellList<dim, 0> lcl2;
    // lcl2[{0}].add_interval({-10, 0});
    // lcl2[{1}].add_interval({0, 10});

    // mure::LevelCellArray<dim, 0> lca1{lcl1};
    // mure::LevelCellArray<dim, 0> lca2{lcl2};

    // // std::cout << lca2 << "\n";
    // // auto clean_expr = mure::intersection(mure::_1, mure::_2);
    // // // auto clean_expr = mure::union_(mure::_2, mure::difference(mure::_1,
    // // // mure::_2));
    // // std::array<mure::LevelCellArray<Config>, 2> set{lca1, lca2};
    // // auto clean_set = mure::make_subset<Config>(clean_expr, 1, {0, 1},
    // set);

    // // clean_set.apply([&](auto &index, auto &interval, auto &interval_index)
    // {
    // //     std::cout << index << " " << interval << "\n";
    // // });

    // // auto clean_expr = mure::intersection(lca1, lca2);
    // // auto clean_expr = mure::intersection(lca1, mure::union_(lca1, lca2));
    // // auto clean_expr = mure::union_(lca2, mure::difference(lca1, lca2));
    // auto test = mure::difference(lca1, lca2);
    // auto clean_expr = mure::union_(lca2, test);
    // clean_expr([](auto &index, auto &interval) {
    //     std::cout << index << " " << interval << "\n";
    // });

    // // clean_expr([](auto &index, auto &interval) {
    // //     std::cout << index << " " << interval << "\n";
    // // });
    // // // auto clean_expr = mure::union_(mure::_2, mure::difference(mure::_1,
    // // // mure::_2));
    // // // std::array<mure::LevelCellArray<Config>, 2> set{lca1, lca2};
    // // // auto clean_set = mure::make_subset<Config>(clean_expr, 1, {0, 1},
    // // set);

    // // // clean_set.apply([&](auto &index, auto &interval, auto
    // &interval_index)
    // // {
    // // //     std::cout << index << " " << interval << "\n";
    // // // });
    return 0;
}