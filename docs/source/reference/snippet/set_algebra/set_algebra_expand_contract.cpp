#include <iostream>

#include <samurai/level_cell_array.hpp>
#include <samurai/level_cell_list.hpp>
#include <samurai/samurai.hpp>
#include <samurai/subset/node.hpp>

template <class Set>
void print(const char* name, const Set& set)
{
    std::cout << name << ":" << std::endl;
    set(
        [](const auto& interval, const auto& index)
        {
            std::cout << "    y = " << index[0] << ": " << interval
                      << std::endl;
        });
}

int main(int argc, char* argv[])
{
    samurai::initialize(argc, argv);

    constexpr std::size_t dim = 2;

    // A plus sign of five cells centered on (1, 2).
    samurai::LevelCellList<dim> list(0);
    list[{1}].add_interval({1, 2});
    list[{2}].add_interval({0, 3});
    list[{3}].add_interval({1, 2});
    const samurai::LevelCellArray<dim> plus(list);

    print("plus", samurai::self(plus));
    print("expand(plus, 1)", samurai::expand(plus, 1));
    print("expand(plus, 1, {false, true})",
          samurai::expand(plus, 1, {false, true}));
    print("contract(plus, 1)", samurai::contract(plus, 1));

    xt::xtensor_fixed<int, xt::xshape<dim>> shift{1, -1};
    print("translate(plus, {1, -1})", samurai::translate(plus, shift));

    samurai::finalize();
    return 0;
}
