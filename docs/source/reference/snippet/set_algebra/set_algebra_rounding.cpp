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

    samurai::LevelCellList<dim> list(2);
    list[{}].add_interval({3, 6});
    const samurai::LevelCellArray<dim> a(list);

    // A set projection to a coarser level rounds outwards
    print("a", samurai::self(a));
    print("a.on(1)", samurai::self(a).on(1));
    print("a.on(0)", samurai::self(a).on(0));

    samurai::finalize();
    return 0;
}
