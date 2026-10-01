#include <filesystem>
namespace fs = std::filesystem;

#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/io/restart.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>

int main(int argc, char** argv)
{
    samurai::initialize(argc, argv);

    static constexpr std::size_t dim = 2;

    samurai::Box<double, dim> box({0.0, 0.0}, {1.0, 1.0});
    auto config = samurai::mesh_config<dim>().min_level(2).max_level(5);
    auto mesh   = samurai::mra::make_mesh(box, config);

    auto field_1 = samurai::make_scalar_field<double>("u", mesh);
    auto field_2 = samurai::make_vector_field<double, 3>("v", mesh);
    field_1.fill(1.0);
    field_2.fill(2.0);

    // Write ./restart_file.h5 (the directory must already exist)
    samurai::dump(fs::current_path(), "restart_file", mesh, field_1, field_2);
    // or, with the same result
    samurai::dump("restart_file", mesh, field_1, field_2);

    // Read ./restart_file.h5 back into the mesh and the fields
    samurai::load(fs::current_path(), "restart_file", mesh, field_1, field_2);
    // or, with the same result
    samurai::load("restart_file", mesh, field_1, field_2);

    samurai::finalize();
    return 0;
}
