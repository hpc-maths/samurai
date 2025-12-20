#include <filesystem>
namespace fs = std::filesystem;

#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/mr/mesh.hpp>

int main()
{
    static constexpr std::size_t dim = 2;

    samurai::Box<double, dim> box({0.0, 0.0}, {1.0, 1.0});
    auto config = samurai::mesh_config<dim>().min_level(2).max_level(5);
    auto mesh   = samurai::mra::make_mesh(box, config);

    auto field_1 = samurai::make_scalar_field<double>("u", mesh);
    auto field_2 = samurai::make_vector_field<double, 3>("v", mesh);

    samurai::save("output_path", "fields", {true, true}, mesh, field_1, field_2);
    // or
    samurai::save(fs::current_path(), "fields", {true, true}, mesh, field_1, field_2);

    return 0;
}
