#include <iostream>
#include <array>
#include <vector>
#include <memory>
#include <functional>
#include <type_traits>

#include <xtensor/xfixed.hpp>

#include <samurai/level_cell_array.hpp>
#include <samurai/subset/subset_op.hpp>
#include <samurai/field.hpp>

// BcValue
/////////////////////////////////////////////////////////////////////////////////////////
template<std::size_t dim, class T, std::size_t size>
struct BcValue
{
    using value_t = xt::xtensor_fixed<T, xt::xshape<size>>;
    using coords_t = xt::xtensor_fixed<T, xt::xshape<dim>>;

    BcValue(){}
    virtual ~BcValue() = default;

    virtual value_t& get_value(const coords_t&) = 0;
    virtual BcValue* clone() const = 0;
};

template<std::size_t dim, class T, std::size_t size>
struct ConstantBc: public BcValue<dim, T, size>
{
    using base_t = BcValue<dim, T, size>;
    using value_t = typename base_t::value_t;
    using coords_t = typename base_t::coords_t;

    ConstantBc(T v)
    {
        m_v.fill(v);
    }

    ConstantBc(const value_t& v)
    : m_v(v)
    {}

    virtual ~ConstantBc() = default;

    inline value_t& get_value(const coords_t&) override
    {
        std::cout << "constant" << std::endl;
        return m_v;
    }

    ConstantBc* clone() const override
    {
        return new ConstantBc(m_v);
    }

    value_t m_v;
};

template<std::size_t dim, class T, std::size_t size>
struct FunctionBc: public BcValue<dim, T, size>
{
    using base_t = BcValue<dim, T, size>;
    using value_t = typename base_t::value_t;
    using coords_t = typename base_t::coords_t;

    template <class Func>
    FunctionBc(Func &&f)
    :m_func(std::forward<Func>(f))
    {}

    virtual ~FunctionBc() = default;

    inline value_t& get_value(const coords_t& coords) override
    {
        std::cout << "function" << std::endl;

        m_v = m_func(coords);
        return m_v;
    }

    FunctionBc* clone() const override
    {
        return new FunctionBc(m_func);
    }

    std::function<value_t (const coords_t&)> m_func;
    value_t m_v;
};

// BcRegion
/////////////////////////////////////////////////////////////////////////////////////////
template<std::size_t dim>
struct BcRegion
{
    BcRegion(){}
    virtual ~BcRegion() = default;

    virtual const samurai::LevelCellArray<dim> get_region(const samurai::LevelCellArray<dim>&) const = 0;
    virtual BcRegion* clone() const = 0;
};

template<std::size_t dim>
struct Everywhere: public BcRegion<dim>
{
    Everywhere(){}

    const samurai::LevelCellArray<dim> get_region(const samurai::LevelCellArray<dim>& mesh) const override
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

    const samurai::LevelCellArray<dim> get_region(const samurai::LevelCellArray<dim>& mesh) const override
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

    const samurai::LevelCellArray<dim> get_region(const samurai::LevelCellArray<dim>& mesh) const override
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

template<std::size_t dim>
auto make_region(Everywhere<dim>)
{
    return std::make_unique<Everywhere<dim>>();
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
        return p_bcregion->get_region(mesh);
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

template<class mesh_t_, class value_t = double, std::size_t size_ = 1, bool SOA=false>
struct MyField: public samurai::Field<mesh_t_, value_t, size_, SOA>
{
    using base_type = samurai::Field<mesh_t_, value_t, size_, SOA>;
    static constexpr std::size_t dim = base_type::dim;

    using base_type::base_type;

    template<class Bc_derived>
    auto attach(const Bc_derived& bc)
    {
        //p_bc.push_back(std::unique_ptr<Bc_derived>(bc.clone()));
        p_bc.push_back(std::make_unique<Bc_derived>(bc));
        return *p_bc.back().get();
    }

    std::vector<std::unique_ptr<Bc<dim, value_t, size_>>> p_bc;
};

template <class value_t, std::size_t size, bool SOA=false, class mesh_t>
auto make_field(std::string name, mesh_t& mesh)
{
    using field_t = MyField<mesh_t, value_t, size, SOA>;
    return field_t(name, mesh);
}

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

template<template<std::size_t, class, std::size_t> class bc_type, class Field>
auto make_bc(Field& field, const std::function<xt::xtensor_fixed<typename Field::value_type, xt::xshape<Field::size>>(const xt::xtensor_fixed<typename Field::value_type, xt::xshape<Field::dim>>&)>& func)
{
    using value_t = typename Field::value_type;
    constexpr std::size_t dim = Field::dim;
    constexpr std::size_t size = Field::size;

    // static_assert(std::is_same_v<typename Field::value_type, std::common_type_t<T...>>, "The constant value type must be the same as the field value_type");
    // static_assert(Field::size == sizeof...(T), "The number of constant values should be equal to the number of element in the field");

    return field.attach(bc_type<dim, value_t, size>(FunctionBc<dim, value_t, size>(func)));
}

template<template<std::size_t, class, std::size_t> class bc_type, class Field, class... T>
auto make_bc(Field& field, typename Field::value_type v1, T... v)
{
    using value_t = typename Field::value_type;
    constexpr std::size_t dim = Field::dim;
    constexpr std::size_t size = Field::size;

    static_assert(std::is_same_v<typename Field::value_type, std::common_type_t<typename Field::value_type, T...>>, "The constant value type must be the same as the field value_type");
    static_assert(Field::size == sizeof...(T) + 1, "The number of constant values should be equal to the number of element in the field");

    return field.attach(bc_type<dim, value_t, size>(ConstantBc<dim, value_t, size>(v1, v...)));
}

int main()
{
    constexpr std::size_t dim = 2;
    samurai::Box<double, dim> box = {{0, 0}, {1, 1}};
    samurai::LevelCellArray<dim> lca = {2, box};

    auto u = ::make_field<double, 1>("u", lca);

    auto bc = make_bc<Dirichlet>(u, 1.);
    bc.on([](auto& coords)
    {
        return (coords[0] >= .25 && coords[0] <= .75);
    });

    std::cout << bc.get_lca(u.mesh()) << std::endl;

    bc.on(Everywhere<dim>());
    std::cout << bc.get_lca(u.mesh()) << std::endl;

    std::cout << u.p_bc.back().get()->get_lca(u.mesh()) << std::endl;

    make_bc<Dirichlet>(u, [](auto& coords)
    {
        // return 1;
        return xt::xtensor_fixed<double, xt::xshape<1>>(1);
    });

    // Field<2, double, 3> f;
    // Dirichlet<2, double, 3> dirichlet(ConstantBc<2, double, 3>{4});
    // dirichlet.on([](auto& coords)
    // {
    //     return (coords[0] >= .25 && coords[0] <= .75);
    // });

    // // Neumann<2, double, 3> neumann(ConstantBc<2, double, 3>{4});
    // Neumann<2, double, 3> neumann(FunctionBc<2, double, 3>([](auto& coords) -> std::array<double, 3>
    // {
    //     return {coords[0], 2*coords[1], 3*coords[0]*coords[1]};
    // }));
    // neumann.on(samurai::difference(lca, samurai::translate(lca, xt::xtensor_fixed<int, xt::xshape<1>>{1})));
    // f.attach(neumann);

    // std::cout << "lca" << std::endl;
    // std::cout << lca << std::endl;
    // std::cout << "dirichlet" << std::endl;
    // std::cout << dirichlet.get_lca(lca) << std::endl;
    // std::cout << "neumann" << std::endl;
    // std::cout << neumann.get_lca(lca) << std::endl;

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