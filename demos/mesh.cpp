#include <mure/box.hpp>
#include <mure/mesh.hpp>
#include <mure/mr_config.hpp>
#include <mure/level_cell_list.hpp>
#include <mure/level_cell_array.hpp>

int main()
{
    constexpr size_t dim = 3;
    using config = mure::MRConfig<dim>;

    mure::Box<double, dim> box{{0, 0, 0}, {1, 1, 1}};
    // mure::Mesh<mure::MRConfig<dim>> mesh{box, 1};

    std::size_t init_level = 1;
    mure::Box<int, dim>::point_t start = box.min_corner()*std::pow(2, init_level);
    mure::Box<int, dim>::point_t end = box.max_corner()*std::pow(2, init_level);

    mure::LevelCellList<config> dcl;
    dcl.extend(xt::view(start, xt::drop(0)), xt::view(end, xt::drop(0)));
    dcl.fill({start[0], end[0]});
    std::cout << dcl.min_corner_yz() << " " << dcl.max_corner_yz() << "\n";
    mure::LevelCellArray<config> dca = dcl;

    return 0;
}