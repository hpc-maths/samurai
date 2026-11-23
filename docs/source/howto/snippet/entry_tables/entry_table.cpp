#include <cmath>
#include <iostream>

#include <samurai/algorithm.hpp>
#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/entry_table.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>

// The cells of one level whose next cell along x is also a cell of that level
struct cell_and_next
{
    using entry_type = samurai::Rows<2>;

    std::size_t level;

    void produce(const auto& mesh, auto& emit) const
    {
        using mesh_id_t = typename std::decay_t<decltype(mesh)>::mesh_id_t;

        const auto& cells = mesh[mesh_id_t::cells][level];
        const xt::xtensor_fixed<int, xt::xshape<2>> to_previous{-1, 0};
        auto set = samurai::intersection(cells,
                                         samurai::translate(cells, to_previous));
        set(
            [&](const auto& i, const auto& index)
            {
                // the row of cells of the interval i, and the row of their next
                // cells
                emit(samurai::make_rows(mesh,
                                        i.size(),
                                        {
                                            {level, i.start,     index},
                                            {level, i.start + 1, index}
                }));
            });
    }
};

int main(int argc, char* argv[])
{
    samurai::initialize(argc, argv);

    static constexpr std::size_t dim = 2;

    samurai::Box<double, dim> box({0.0, 0.0}, {1.0, 1.0});
    auto config = samurai::mesh_config<dim>();
    config.min_level(2).max_level(6);
    auto mesh = samurai::mra::make_mesh(box, config);

    auto u = samurai::make_scalar_field<double>("u", mesh);
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               const double x = cell.center(0) - 0.5;
                               const double y = cell.center(1) - 0.5;
                               u[cell] = std::exp(-200 * (x * x + y * y));
                           });
    samurai::make_bc<samurai::Neumann<1>>(u, 0.);
    samurai::make_MRAdapt(u)(samurai::mra_config().epsilon(1e-3));

    auto du = samurai::make_scalar_field<double>("du", mesh);
    du.fill(0.);

    // the kernel: the difference between a cell and its next cell
    std::size_t n_differences = 0;
    for (int step = 0; step < 3; ++step)
    {
        for (std::size_t level = mesh.min_level(); level <= mesh.max_level();
             ++level)
        {
            samurai::for_each_entry(
                mesh,
                cell_and_next{level},
                [&](const auto& rows)
                {
                    const auto* cells = u.data() + rows.at[0];
                    const auto* next  = u.data() + rows.at[1];
                    auto* result      = du.data() + rows.at[0];
                    for (std::size_t k = 0; k < rows.len; ++k)
                    {
                        result[k] = next[k] - cells[k];
                    }
                    n_differences += rows.len;
                });
        }
    }
    std::cout << n_differences / 3 << " differences per step" << std::endl;

    samurai::finalize();
    return 0;
}
