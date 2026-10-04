#include <iostream>

#include <samurai/level_cell_array.hpp>
#include <samurai/level_cell_list.hpp>
#include <samurai/samurai.hpp>
#include <samurai/subset/node.hpp>

template <class Set>
void print(const char* name, const Set& set)
{
    std::cout << name << ":";
    set(
        [](const auto& interval, const auto&)
        {
            std::cout << " " << interval;
        });
    std::cout << std::endl;
}

int main(int argc, char* argv[])
{
    samurai::initialize(argc, argv);

    constexpr std::size_t dim = 1;

    // Two 1D sets at level 0: A = [0,5[ U [10,13[ and B = [4,8[.
    samurai::LevelCellList<dim> list_a(0);
    list_a[{}].add_interval({0, 5});
    list_a[{}].add_interval({10, 13});
    const samurai::LevelCellArray<dim> a(list_a);

    samurai::LevelCellList<dim> list_b(0);
    list_b[{}].add_interval({4, 8});
    const samurai::LevelCellArray<dim> b(list_b);

    print("union_(a, b)", samurai::union_(a, b));
    print("intersection(a, b)", samurai::intersection(a, b));
    print("difference(a, b)", samurai::difference(a, b));
    print("difference(b, a)", samurai::difference(b, a));

    samurai::finalize();
    return 0;
}
