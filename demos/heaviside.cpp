#include <mure/box.hpp>
#include <mure/field.hpp>
#include <mure/hdf5.hpp>
#include <mure/mesh.hpp>
#include <mure/mr_config.hpp>

int main()
{
    constexpr size_t dim = 2;
    using Config = mure::MRConfig<dim>;
    mure::Box<double, dim> box({0, 0, 0}, {1, 1, 1});
    mure::Mesh<Config> mesh{box, 6};

    mure::Field<Config> u{"u", mesh};

    mesh.for_each_cell([&](auto& cell){
        auto center = cell.center();
        if (center[0] < .5)
            u[cell] = 0;
        else
            u[cell] = 1;
    });

    auto h5file = mure::Hdf5("heaviside");
    h5file.add_mesh(mesh);
    h5file.add_field(u);
    return 0;
}