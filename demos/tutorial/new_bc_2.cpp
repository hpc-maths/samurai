#include <iostream>
#include <array>
#include <vector>
#include <memory>
#include <functional>

#include <samurai/level_cell_array.hpp>
#include <samurai/subset/subset_op.hpp>

// BcValue
/////////////////////////////////////////////////////////////////////////////////////////
template<std::size_t dim, class T, std::size_t size>
struct BcValue
{
    BcValue(){}
    virtual ~BcValue() = default;

    virtual std::array<T, size>& apply(const std::array<T, dim>&) = 0;
    virtual BcValue* clone() const = 0;
};

template<std::size_t dim, class T, std::size_t size>
struct ConstantBc: public BcValue<dim, T, size>
{
    ConstantBc(T v)
    {
        m_v.fill(v);
    }

    ConstantBc(const std::array<T, size>& v)
    : m_v(v)
    {}

    virtual ~ConstantBc() = default;

    inline std::array<T, size>& apply(const std::array<T, dim>&) override
    {
        std::cout << "constant" << std::endl;
        return m_v;
    }

    ConstantBc* clone() const override
    {
        return new ConstantBc(m_v);
    }

    std::array<T, size> m_v;
};

template<std::size_t dim, class T, std::size_t size>
struct FunctionBc: public BcValue<dim, T, size>
{
    template <class Func>
    FunctionBc(Func &&f)
    :m_func(std::forward<Func>(f))
    {}

    virtual ~FunctionBc() = default;

    inline std::array<T, size>& apply(const std::array<T, dim>& coords) override
    {
        std::cout << "function" << std::endl;

        m_v = m_func(coords);
        return m_v;
    }

    FunctionBc* clone() const override
    {
        return new FunctionBc(m_func);
    }

    std::function<std::array<T, size>(const std::array<double, dim>&)> m_func;
    std::array<T, size> m_v;
};

// BcRegion
/////////////////////////////////////////////////////////////////////////////////////////
template<std::size_t dim>
struct BcRegion
{
    BcRegion(){}
    virtual ~BcRegion() = default;

    virtual const samurai::LevelCellArray<dim> apply(const samurai::LevelCellArray<dim>&) const = 0;
    virtual BcRegion* clone() const = 0;
};

template<std::size_t dim>
struct Everywhere: public BcRegion<dim>
{
    Everywhere(){}

    const samurai::LevelCellArray<dim> apply(const samurai::LevelCellArray<dim>& mesh) const override
    {
        return samurai::difference(mesh, samurai::contraction(mesh));
    }

    Everywhere* clone() const override
    {
        return new Everywhere();
    }
};

template<std::size_t dim>
struct CoordsRegion: public BcRegion<dim>
{
    template <class Func>
    CoordsRegion(Func &&f)
    :m_func(std::forward<Func>(f))
    {}

    CoordsRegion* clone() const override
    {
        return new CoordsRegion(m_func);
    }

    const samurai::LevelCellArray<dim> apply(const samurai::LevelCellArray<dim>& mesh) const override
    {
        samurai::LevelCellList<dim> lcl{mesh.level()};
        auto set = samurai::difference(mesh, samurai::contraction(mesh));
        samurai::for_each_cell(mesh, set, [&](auto& cell)
        {
            if (m_func(cell.center()))
            {
                lcl.add_cell(cell);
            }
            // samurai::static_nested_loop<dim, 0, 2>([&](auto stencil)
            // {
            //     if (m_func(cell.corner() + cell.length*stencil))
            //     {
            //         lcl.add_cell(cell);
            //     }
            // });
        });
        return lcl;
    }

    std::function<bool(const xt::xtensor_fixed<double, xt::xshape<dim>>&)> m_func;
};

template<std::size_t dim, class Set>
struct SetRegion: public BcRegion<dim>
{
    SetRegion(const Set& set)
    : m_set(set)
    {}

    SetRegion* clone() const override
    {
        return new SetRegion(m_set);
    }

    const samurai::LevelCellArray<dim> apply(const samurai::LevelCellArray<dim>& mesh) const override
    {
        return samurai::intersection(mesh, m_set);
    }

    Set m_set;
};

template<std::size_t dim, class F, class... CT>
auto make_region(samurai::subset_operator<F, CT...> region)
{
    return std::make_unique<SetRegion<dim, samurai::subset_operator<F, CT...>>>(region);
}

template<std::size_t dim, class Func>
auto make_region(Func&& func)
{
    return std::make_unique<CoordsRegion<dim>>(std::forward<Func>(func));
}

// Bc
/////////////////////////////////////////////////////////////////////////////////////////
template<std::size_t dim, class T, std::size_t size>
struct Bc
{
    virtual ~Bc() = default;

    template<class Bcvalue>
    Bc(const Bcvalue& bcv)

    {
        p_bcvalue = std::make_unique<Bcvalue>(bcv);
        p_bcregion = std::make_unique<Everywhere<dim>>();
    }

    Bc(const Bc& bc)
    : p_bcvalue(bc.p_bcvalue->clone())
    , p_bcregion(bc.p_bcregion->clone())
    {
    }

    template<class Region>
    auto on(const Region& region)
    {
        p_bcregion = make_region<dim>(region);
        return *this;
    }

    auto get_lca(const samurai::LevelCellArray<dim>& mesh)
    {
        return p_bcregion->apply(mesh);
    }

    std::unique_ptr<BcValue<dim, T, size>> p_bcvalue;
    std::unique_ptr<BcRegion<dim>> p_bcregion;
};


template<std::size_t dim, class T, std::size_t size>
struct Dirichlet: public Bc<dim, T, size>
{
    using Bc<dim, T, size>::Bc;
};

template<std::size_t dim, class T, std::size_t size>
struct Neumann: public Bc<dim, T, size>
{
    using Bc<dim, T, size>::Bc;
};

template<std::size_t dim, class T, std::size_t size>
struct Robin: public Bc<dim, T, size>
{
    using Bc<dim, T, size>::Bc;
};

template<std::size_t dim, class T, std::size_t size>
struct Field
{
    Field(){}

    template<class Bc_derived>
    void attach(const Bc_derived& bc)
    {
        //p_bc.push_back(std::unique_ptr<Bc_derived>(bc.clone()));
        p_bc.push_back(std::make_unique<Bc_derived>(bc));
    }
    std::vector<std::unique_ptr<Bc<dim, T, size>>> p_bc;
};

template<class Field, std::size_t dim=Field::dim, class T = typename Field::type, std::size_t size=Field::size>
void apply(const Dirichlet<dim, T, size>& dirichlet, Field& field)
{
    std::cout << "Dirichlet" << std::endl;
    std::cout << dirichlet.p_bcvalue->apply({1, 2})[0] << std::endl;
}

template<class Field, std::size_t dim=Field::dim, class T = typename Field::type, std::size_t size=Field::size>
void apply(const Neumann<dim, T, size>& neumann, Field& field)
{
    std::cout << "Neumann" << std::endl;
    std::cout << neumann.p_bcvalue->apply({1, 2})[0] << std::endl;
}

int main()
{
    constexpr std::size_t dim = 2;
    samurai::Box<double, dim> box = {{0, 0}, {1, 1}};
    samurai::LevelCellArray<dim> lca = {2, box};

    Field<2, double, 3> f;
    Dirichlet<2, double, 3> dirichlet(ConstantBc<2, double, 3>{4});
    dirichlet.on([](auto& coords)
    {
        return (coords[0] >= .25 && coords[0] <= .75);
    });

    f.attach(dirichlet);

    // Neumann<2, double, 3> neumann(ConstantBc<2, double, 3>{4});
    Neumann<2, double, 3> neumann(FunctionBc<2, double, 3>([](auto& coords) -> std::array<double, 3>
    {
        return {coords[0], 2*coords[1], 3*coords[0]*coords[1]};
    }));
    neumann.on(samurai::difference(lca, samurai::translate(lca, xt::xtensor_fixed<int, xt::xshape<1>>{1})));
    f.attach(neumann);

    std::cout << "lca" << std::endl;
    std::cout << lca << std::endl;
    std::cout << "dirichlet" << std::endl;
    std::cout << dirichlet.get_lca(lca) << std::endl;
    std::cout << "neumann" << std::endl;
    std::cout << neumann.get_lca(lca) << std::endl;

    Robin<2, double, 3> robin(ConstantBc<2, double, 3>{4});
    f.attach(robin);

    for(auto& p: f.p_bc)
    {
        if (dynamic_cast<Dirichlet<2, double, 3>*>(p.get()))
        {
            apply(*dynamic_cast<Dirichlet<2, double, 3>*>(p.get()), f);
        }
        else if (dynamic_cast<Neumann<2, double, 3>*>(p.get()))
        {
            apply(*dynamic_cast<Neumann<2, double, 3>*>(p.get()), f);
        }
        else
        {
            std::cout << "not known" << std::endl;
        }
    }

    return 0;
}