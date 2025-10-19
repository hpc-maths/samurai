#include <samurai/domain_builder.hpp>

int main()
{
    static constexpr std::size_t dim = 2;

    samurai::DomainBuilder<dim> domain({-1., -1.}, {1., 1.});
    domain.remove({0.0, 0.0}, {0.4, 0.4});

    return 0;
}
