#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>

int main()
{
    static constexpr std::size_t dim = 2;
    using config_t                   = samurai::MRConfig<dim>;

    samurai::Box<double, dim> box({0.0, 0.0}, {1.0, 1.0});
    samurai::MRMesh<config_t> mesh(box, 2, 5); // min level 2, max level 5

    auto field = samurai::make_vector_field<double, 3>("v", mesh); // 3 components

    return 0;
}
