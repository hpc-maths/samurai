#include <mure/box.hpp>
#include <mure/hdf5.hpp>
#include <mure/mesh.hpp>
#include <mure/mr_config.hpp>
#include <mure/level_cell_list.hpp>
#include <mure/level_cell_array.hpp>

int main()
{
    constexpr size_t dim = 2;
    using config = mure::MRConfig<dim>;

    mure::Box<double, dim> box{{0, 0, 0}, {1, 1, 1}};
    mure::Mesh<mure::MRConfig<dim>> mesh{box, 1};

    // std::cout << mesh << "\n";
    auto h5file = mure::Hdf5("test.h5");
    h5file.add_mesh(mesh);
    return 0;
}