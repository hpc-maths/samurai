#include <samurai/box.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/mr/mesh.hpp>

int main()
{
    static constexpr std::size_t dim = 2;
    using config_t                   = samurai::MRConfig<dim>;

    samurai::Box<double, dim> box({0.0, 0.0}, {1.0, 1.0});
    samurai::MRMesh<config_t> mesh(box, 2, 5); // min level 2, max level 5

    samurai::save("output_path", "mesh_filename", mesh);
    // or
    samurai::save("mesh_filename", mesh);

    return 0;
}
