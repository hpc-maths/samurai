#include <iostream>

#include <samurai/level_cell_array.hpp>
#include <samurai/level_cell_list.hpp>
#include <samurai/samurai.hpp>
#include <samurai/subset/node.hpp>

template <class Set>
void print(const char* name, const Set& set)
{
    std::cout << name << " (level " << set.level() << "):";
    set(
        [](const auto& interval, const auto& /* index */)
        {
            std::cout << " " << interval;
        });
    std::cout << std::endl;
}

int main(int argc, char* argv[])
{
    samurai::initialize(argc, argv);

    constexpr std::size_t dim = 1;

    samurai::LevelCellList<dim> fine_list(1);
    fine_list[{}].add_interval({0, 4});
    const samurai::LevelCellArray<dim> fine(fine_list);

    samurai::LevelCellList<dim> coarse_list(0);
    coarse_list[{}].add_interval({1, 3});
    const samurai::LevelCellArray<dim> coarse(coarse_list);

    auto set = samurai::intersection(fine, coarse);
    print("intersection", set);
    print("intersection.on(0)", set.on(0));
    print("intersection.on(3)", set.on(3));

    set.on(0).apply_op(
        [](auto level, const auto& interval, const auto& /* index */)
        {
            std::cout << "apply_op: level " << level;
            std::cout << ", interval " << interval << std::endl;
        });

    samurai::finalize();
    return 0;
}
