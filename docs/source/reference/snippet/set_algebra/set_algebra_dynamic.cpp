#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <samurai/level_cell_array.hpp>
#include <samurai/level_cell_list.hpp>
#include <samurai/samurai.hpp>
#include <samurai/subset/dynamic/dynamic.hpp>

int main(int argc, char* argv[])
{
    samurai::initialize(argc, argv);

    constexpr std::size_t dim = 1;
    using lca_t               = samurai::LevelCellArray<dim>;
    using set_t               = samurai::DynamicSet<dim, lca_t::interval_t>;

    std::vector<lca_t> blocks;
    for (int start : {0, 6, 12})
    {
        samurai::LevelCellList<dim> list(0);
        list[{}].add_interval({start, start + 3});
        blocks.emplace_back(list);
    }

    // The number of operands is only known at runtime.
    std::vector<set_t> operands;
    for (const auto& block : blocks)
    {
        operands.push_back(samurai::dyn::self(block));
    }
    const set_t set = samurai::dyn::union_(operands).on(1);

    // One clone per thread: a DynamicSet holds its traversal state
    std::vector<std::string> results(2);
    std::vector<std::thread> threads;
    for (std::size_t t = 0; t < results.size(); ++t)
    {
        threads.emplace_back(
            [&results, t, local = set.clone()]()
            {
                local(
                    [&](const auto& interval, const auto&)
                    {
                        const auto a = std::to_string(interval.start);
                        const auto b = std::to_string(interval.end);
                        results[t] += " [" + a + "," + b + ")";
                    });
            });
    }
    for (auto& thread : threads)
    {
        thread.join();
    }

    for (std::size_t t = 0; t < results.size(); ++t)
    {
        std::cout << "thread " << t << ":" << results[t] << std::endl;
    }

    samurai::finalize();
    return 0;
}
