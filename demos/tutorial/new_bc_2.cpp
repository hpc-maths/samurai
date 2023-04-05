#include <iostream>

#include <xtensor/xfixed.hpp>

#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/level_cell_array.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/subset/subset_op.hpp>
#include <samurai/uniform_mesh.hpp>

template <class Field, std::size_t dim = Field::dim, class interval_t = typename Field::interval_t, class T = typename Field::type, std::size_t size = Field::size>
void apply(const samurai::Dirichlet<dim, interval_t, T, size>& dirichlet, Field&)
{
    std::cout << "Dirichlet" << std::endl;
    std::cout << dirichlet.p_bcvalue->apply({1, 2})[0] << std::endl;
}

template <class Field, std::size_t dim = Field::dim, class interval_t = typename Field::interval_t, class T = typename Field::type, std::size_t size = Field::size>
void apply(const samurai::Neumann<dim, interval_t, T, size>& neumann, Field&)
{
    std::cout << "Neumann" << std::endl;
    std::cout << neumann.p_bcvalue->apply({1, 2})[0] << std::endl;
}

int main()
{
    constexpr std::size_t dim     = 2;
    std::size_t start_level       = 4;
    samurai::Box<double, dim> box = {
        {0, 0, 0},
        {1, 1, 1}
    };

    // samurai::LevelCellArray<dim> mesh = {start_level, box};

    // using Config = samurai::UniformConfig<dim>;
    // samurai::UniformMesh<Config> mesh = {box, start_level};

    using Config = samurai::MRConfig<dim>;
    samurai::MRMesh<Config> mesh(box, start_level, start_level);

    auto u = samurai::make_field<double, 1>("u", mesh);
    u.fill(0);
    // std::cout << "ite -> " << ite << std::endl;
    // auto bc = samurai::make_bc<samurai::Dirichlet>(u, 1., 2.);
    auto bc = samurai::make_bc<samurai::Dirichlet>(u,
                                                   [](auto& coords)
                                                   {
                                                       // return
                                                       // xt::xtensor_fixed<double,
                                                       // xt::xshape<2>>{coords[0],
                                                       // coords[1]};
                                                       return coords[0];
                                                   });
    bc->on(
        [](auto& coords)
        {
            return (coords[0] >= .25 && coords[0] <= .75);
        });

    // std::cout << bc->get_lca() << std::endl;

    bc->on(samurai::Everywhere<dim, typename Config::interval_t>());
    // std::cout << bc->get_lca() << std::endl;
    // // samurai::save("domain",  u.mesh().domain());
    // // samurai::save("boundary",  bc.get_lca());

    // std::cout << u.get_bc().back().get()->get_lca() << std::endl;

    samurai::make_bc<samurai::Neumann>(u,
                                       [](auto&)
                                       {
                                           return 1;
                                           // return xt::xtensor_fixed<double,
                                           // xt::xshape<1>>(1);
                                       });

    // auto uvec = samurai::make_field<double, 4>("u", mesh);

    // samurai::make_bc<samurai::Dirichlet>(uvec, 1., 2., 3., 0.);
    // auto bcn = samurai::make_bc<samurai::Neumann>(uvec, [](auto&)
    // {
    //     return xt::ones<double>({4});
    // });
    // // .on(samurai::difference(mesh, samurai::translate(mesh,
    // xt::xtensor_fixed<int, xt::xshape<1>>{1}))); std::cout << bcn.get_lca()
    // << std::endl;

    samurai::update_bc(u);
    samurai::save(fs::current_path(), "boundary", {true, true}, mesh, u);

    // Robin<2, double, 3> robin(ConstantBc<2, double, 3>{4});
    // f.attach(robin);

    // for(auto& p: f.p_bc)
    // {
    //     if (dynamic_cast<Dirichlet<2, double, 3>*>(p.get()))
    //     {
    //         apply(*dynamic_cast<Dirichlet<2, double, 3>*>(p.get()), f);
    //     }
    //     else if (dynamic_cast<Neumann<2, double, 3>*>(p.get()))
    //     {
    //         apply(*dynamic_cast<Neumann<2, double, 3>*>(p.get()), f);
    //     }
    //     else
    //     {
    //         std::cout << "not known" << std::endl;
    //     }
    // }

    return 0;
}