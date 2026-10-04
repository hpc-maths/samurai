#include <samurai/domain_builder.hpp>
#include <samurai/samurai.hpp>

int main(int argc, char* argv[])
{
    samurai::initialize(argc, argv);

    static constexpr std::size_t dim = 2;

    samurai::DomainBuilder<dim> domain({0.0, 0.0}, {2.0, 1.0});
    domain.add({0.0, 1.0}, {1.0, 2.0});

    samurai::finalize();
    return 0;
}
