// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <vector>
#include <utility>
#include <memory>
#include <functional>
#include <type_traits>

#include <xtensor/xfixed.hpp>
namespace samurai
{
    enum class BCType
    {
        dirichlet = 0,
        neumann = 1,
        periodic = 2,
        interpolation = 3 // Reconstruct the function by linear approximation
    };

    template<std::size_t Dim>
    struct BC
    {
        std::vector<std::pair<BCType, double>> type;
    };


    namespace detail
    {
        template<class T, std::size_t size>
        struct return_type
        {
            using type = xt::xtensor_fixed<T, xt::xshape<size>>;
        };

        template<class T>
        struct return_type<T, 1>
        {
            using type = T;
        };

        template<class T, std::size_t size>
        using return_type_t = typename return_type<T, size>::type;
    }

    ////////////////////////
    // BcValue definition //
    ////////////////////////
    template<std::size_t dim, class T, std::size_t size>
    struct BcValue
    {
        using value_t = detail::return_type_t<T, size>;
        using coords_t = xt::xtensor_fixed<T, xt::xshape<dim>>;

        BcValue(){}
        virtual ~BcValue() = default;

        virtual value_t& get_value(const coords_t&) = 0;
        virtual BcValue* clone() const = 0;
    };

    template<std::size_t dim, class T, std::size_t size>
    class ConstantBc: public BcValue<dim, T, size>
    {
    public:
        using base_t = BcValue<dim, T, size>;
        using value_t = typename base_t::value_t;
        using coords_t = typename base_t::coords_t;

        template<class... CT>
        ConstantBc(const CT... v);
        virtual ~ConstantBc() = default;

        value_t& get_value(const coords_t&) override;
        ConstantBc* clone() const override;

    private:
        value_t m_v;
    };

    ////////////////////////////
    // BcValue implementation //
    ////////////////////////////
    template<std::size_t dim, class T, std::size_t size>
    class FunctionBc: public BcValue<dim, T, size>
    {
    public:
        using base_t = BcValue<dim, T, size>;
        using value_t = typename base_t::value_t;
        using coords_t = typename base_t::coords_t;

        template <class Func>
        FunctionBc(Func &&f);
        virtual ~FunctionBc() = default;

        value_t& get_value(const coords_t& coords) override;
        FunctionBc* clone() const override;

    private:
        std::function<value_t (const coords_t&)> m_func;
        value_t m_v;
    };

    template<std::size_t dim, class T, std::size_t size>
    template<class... CT>
    ConstantBc<dim, T, size>::ConstantBc(const CT... v)
    {
        m_v = {v...};
    }

    template<std::size_t dim, class T, std::size_t size>
    inline auto ConstantBc<dim, T, size>::get_value(const coords_t&) -> value_t&
    {
        return m_v;
    }

    template<std::size_t dim, class T, std::size_t size>
    ConstantBc<dim, T, size>* ConstantBc<dim, T, size>::clone() const
    {
        return new ConstantBc(m_v);
    }

    template<std::size_t dim, class T, std::size_t size>
    template <class Func>
    FunctionBc<dim, T, size>::FunctionBc(Func &&f)
    :m_func(std::forward<Func>(f))
    {}

    template<std::size_t dim, class T, std::size_t size>
    inline auto FunctionBc<dim, T, size>::get_value(const coords_t& coords) -> value_t&
    {
        m_v = m_func(coords);
        return m_v;
    }

    template<std::size_t dim, class T, std::size_t size>
    FunctionBc<dim, T, size>* FunctionBc<dim, T, size>::clone() const
    {
        return new FunctionBc(m_func);
    }

    /////////////////////////
    // BcRegion definition //
    /////////////////////////
    template<std::size_t dim>
    struct BcRegion
    {
        BcRegion(){}
        virtual ~BcRegion() = default;

        virtual const LevelCellArray<dim> get_region(const LevelCellArray<dim>&) const = 0;
        virtual BcRegion* clone() const = 0;
    };

    template<std::size_t dim>
    struct Everywhere: public BcRegion<dim>
    {
        Everywhere() = default;

        const LevelCellArray<dim> get_region(const LevelCellArray<dim>& mesh) const override;
        Everywhere* clone() const override;
    };

    template<std::size_t dim>
    class CoordsRegion: public BcRegion<dim>
    {
    public:
        template <class Func>
        CoordsRegion(Func &&f);

        CoordsRegion* clone() const override;
        const LevelCellArray<dim> get_region(const LevelCellArray<dim>& mesh) const override;

    private:
        std::function<bool(const xt::xtensor_fixed<double, xt::xshape<dim>>&)> m_func;
    };

    template<std::size_t dim, class Set>
    class SetRegion: public BcRegion<dim>
    {
    public:
        SetRegion(const Set& set);

        SetRegion* clone() const override;
        const LevelCellArray<dim> get_region(const LevelCellArray<dim>& mesh) const override;

    private:
        Set m_set;
    };

    /////////////////////////////
    // BcRegion implementation //
    /////////////////////////////

    // Everywhere
    template<std::size_t dim>
    inline const LevelCellArray<dim> Everywhere<dim>::get_region(const LevelCellArray<dim>& mesh) const
    {
        return difference(mesh, contraction(mesh));
    }

    template<std::size_t dim>
    Everywhere<dim>* Everywhere<dim>::clone() const
    {
        return new Everywhere();
    }

    // CoordsRegion
    template<std::size_t dim>
    template <class Func>
    CoordsRegion<dim>::CoordsRegion(Func &&f)
    :m_func(std::forward<Func>(f))
    {}

    template<std::size_t dim>
    CoordsRegion<dim>* CoordsRegion<dim>::clone() const
    {
        return new CoordsRegion(m_func);
    }

    template<std::size_t dim>
    inline const LevelCellArray<dim> CoordsRegion<dim>::get_region(const LevelCellArray<dim>& mesh) const
    {
        LevelCellList<dim> lcl{mesh.level()};
        auto set = difference(mesh, contraction(mesh));
        for_each_cell(mesh, set, [&](auto& cell)
        {
            if (m_func(cell.center()))
            {
                lcl.add_cell(cell);
            }
            // static_nested_loop<dim, 0, 2>([&](auto stencil)
            // {
            //     if (m_func(cell.corner() + cell.length*stencil))
            //     {
            //         lcl.add_cell(cell);
            //     }
            // });
        });
        return lcl;
    }

    // SetRegion
    template<std::size_t dim, class Set>
    SetRegion<dim, Set>::SetRegion(const Set& set)
    : m_set(set)
    {}

    template<std::size_t dim, class Set>
    SetRegion<dim, Set>* SetRegion<dim, Set>::clone() const
    {
        return new SetRegion(m_set);
    }

    template<std::size_t dim, class Set>
    const LevelCellArray<dim> SetRegion<dim, Set>::get_region(const LevelCellArray<dim>& mesh) const
    {
        return intersection(mesh, m_set);
    }

    ///////////////////////////////
    // BcRegion helper functions //
    ///////////////////////////////
    template<std::size_t dim, class F, class... CT>
    auto make_region(subset_operator<F, CT...> region)
    {
        return std::make_unique<SetRegion<dim, subset_operator<F, CT...>>>(region);
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

    ///////////////////
    // Bc definition //
    ///////////////////
    template<std::size_t dim, class T, std::size_t size>
    class Bc
    {
    public:
        // virtual ~Bc() = default;

        template<class Bcvalue>
        Bc(const LevelCellArray<dim>& mesh, const Bcvalue& bcv);

        Bc(const Bc& bc);

        template<class Region>
        auto on(const Region& region);
        auto get_lca();

    private:
        std::unique_ptr<BcValue<dim, T, size>> p_bcvalue;
        std::unique_ptr<BcRegion<dim>> p_bcregion;
        const LevelCellArray<dim>& m_mesh;
    };

    ///////////////////
    // Bc definition //
    ///////////////////
    template<std::size_t dim, class T, std::size_t size>
    template<class Bcvalue>
    Bc<dim, T, size>::Bc(const LevelCellArray<dim>& mesh, const Bcvalue& bcv)
    : m_mesh(mesh)
    {
        p_bcvalue = std::make_unique<Bcvalue>(bcv);
        p_bcregion = std::make_unique<Everywhere<dim>>();
    }

    template<std::size_t dim, class T, std::size_t size>
    Bc<dim, T, size>::Bc(const Bc& bc)
    : p_bcvalue(bc.p_bcvalue->clone())
    , p_bcregion(bc.p_bcregion->clone())
    , m_mesh(bc.m_mesh)
    {
    }

    template<std::size_t dim, class T, std::size_t size>
    template<class Region>
    inline auto Bc<dim, T, size>::on(const Region& region)
    {
        p_bcregion = make_region<dim>(region);
        return *this;
    }

    template<std::size_t dim, class T, std::size_t size>
    inline auto Bc<dim, T, size>::get_lca()
    {
        return p_bcregion->get_region(m_mesh);
    }

    /////////////////////////
    // Bc helper functions //
    /////////////////////////
    template<std::size_t dim, class TInterval>
    class LevelCellArray;

    template<class D, class Config>
    class Mesh_base;

    template<class Config>
    class UniformMesh;

    namespace detail
    {
        template<std::size_t dim, class TInterval>
        decltype(auto) get_mesh(const LevelCellArray<dim, TInterval>& mesh)
        {
            return mesh;
        }

        template<class D, class Config>
        decltype(auto) get_mesh(const Mesh_base<D, Config>& mesh)
        {
            return mesh.domain();
        }

        template<class Config>
        decltype(auto) get_mesh(const UniformMesh<Config>& mesh)
        {
            using mesh_id_t = typename Config::mesh_id_t;
            return mesh[mesh_id_t::cells];
        }
    }

    template<template<std::size_t, class, std::size_t> class bc_type, class Field>
    auto make_bc(Field& field, const std::function<detail::return_type_t<typename Field::value_type, Field::size>(const xt::xtensor_fixed<typename Field::value_type, xt::xshape<Field::dim>>&)>& func)
    {
        using value_t = typename Field::value_type;
        constexpr std::size_t dim = Field::dim;
        constexpr std::size_t size = Field::size;

        auto& mesh = detail::get_mesh(field.mesh());
        return field.attach(bc_type<dim, value_t, size>(mesh, FunctionBc<dim, value_t, size>(func)));
    }

    template<template<std::size_t, class, std::size_t> class bc_type, class Field, class... T>
    auto make_bc(Field& field, typename Field::value_type v1, T... v)
    {
        using value_t = typename Field::value_type;
        constexpr std::size_t dim = Field::dim;
        constexpr std::size_t size = Field::size;

        static_assert(std::is_same_v<typename Field::value_type, std::common_type_t<typename Field::value_type, T...>>, "The constant value type must be the same as the field value_type");
        static_assert(Field::size == sizeof...(T) + 1, "The number of constant values should be equal to the number of element in the field");

        auto& mesh = detail::get_mesh(field.mesh());
        return field.attach(bc_type<dim, value_t, size>(mesh, ConstantBc<dim, value_t, size>(v1, v...)));
    }

    //////////////
    // BC Types //
    //////////////
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

} // namespace samurai