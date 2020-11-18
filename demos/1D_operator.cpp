#include <samurai/subset.hpp>
#include <samurai/intervals_operator.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/mesh.hpp>
#include <samurai/mr_config.hpp>

int main()
{
    constexpr std::size_t dim = 1;
    using config = samurai::MRConfig<dim, 1, 1, 2>;

    samurai::CellList<config> cell_list;
    cell_list[0][{}].add_interval({0, 1});
    cell_list[0][{}].add_interval({4, 6});
    cell_list[1][{}].add_interval({2, 3});
    cell_list[1][{}].add_interval({6, 8});
    cell_list[2][{}].add_interval({6, 12});

    samurai::Mesh<config> mesh(cell_list);

    auto cells = mesh.get_cells(1);
    std::cout << mesh << "\n";

    mesh.projection();
    // mesh.prediction();
    // mesh.coarsening();
}