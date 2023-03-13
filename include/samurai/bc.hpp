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


    template<std::size_t dim, class TInterval>
    class LevelCellArray;

    template<std::size_t dim, class TInterval>
    class LevelCellList;

    template<class D, class Config>
    class Mesh_base;

    template<class Config>
    class UniformMesh;

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

        virtual ~BcValue() = default;
        BcValue(const BcValue&) = delete;
        BcValue& operator=(const BcValue&) = delete;
        BcValue(BcValue&&) = delete;
        BcValue& operator=(BcValue&&) = delete;

        virtual value_t& get_value(const coords_t&) = 0;
        virtual std::unique_ptr<BcValue> clone() const = 0;
    protected:
        BcValue(){}
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

        ConstantBc(const ConstantBc& bc);
        ConstantBc& operator=(const ConstantBc& bc);

        value_t& get_value(const coords_t&) override;
        std::unique_ptr<base_t> clone() const override;

    private:
        value_t m_v;
    };

    template<std::size_t dim, class T, std::size_t size>
    class FunctionBc: public BcValue<dim, T, size>
    {
    public:
        using base_t = BcValue<dim, T, size>;
        using value_t = typename base_t::value_t;
        using coords_t = typename base_t::coords_t;

        template <class Func>
        FunctionBc(Func &&f);

        FunctionBc(const FunctionBc& bc);
        FunctionBc& operator=(const FunctionBc& bc);

        value_t& get_value(const coords_t& coords) override;
        std::unique_ptr<base_t> clone() const override;

    private:
        std::function<value_t (const coords_t&)> m_func;
        value_t m_v;
    };

    ////////////////////////////
    // BcValue implementation //
    ////////////////////////////

    template<std::size_t dim, class T, std::size_t size>
    template<class... CT>
    ConstantBc<dim, T, size>::ConstantBc(const CT... v)
    {
        m_v = {v...};
    }

    template<std::size_t dim, class T, std::size_t size>
    ConstantBc<dim, T, size>::ConstantBc(const ConstantBc& bc)
    : m_v(bc.m_v)
    {}

    template<std::size_t dim, class T, std::size_t size>
    ConstantBc<dim, T, size>& ConstantBc<dim, T, size>::operator=(const ConstantBc& bc)
    {
        return {bc.m_v};
    }

    template<std::size_t dim, class T, std::size_t size>
    inline auto ConstantBc<dim, T, size>::get_value(const coords_t&) -> value_t&
    {
        std::cout << "constant" << std::endl;
        return m_v;
    }

    template<std::size_t dim, class T, std::size_t size>
    auto ConstantBc<dim, T, size>::clone() const -> std::unique_ptr<base_t>
    {
        return std::make_unique<ConstantBc>(*this);
    }

    template<std::size_t dim, class T, std::size_t size>
    template <class Func>
    FunctionBc<dim, T, size>::FunctionBc(Func &&f)
    :m_func(std::forward<Func>(f))
    {}

    template<std::size_t dim, class T, std::size_t size>
    FunctionBc<dim, T, size>::FunctionBc(const FunctionBc& bc)
    : m_func(bc.m_func)
    {}

    template<std::size_t dim, class T, std::size_t size>
    FunctionBc<dim, T, size>& FunctionBc<dim, T, size>::operator=(const FunctionBc& bc)
    {
        return {bc.m_func};
    }

    template<std::size_t dim, class T, std::size_t size>
    inline auto FunctionBc<dim, T, size>::get_value(const coords_t& coords) -> value_t&
    {
        std::cout << "function" << std::endl;
        m_v = m_func(coords);
        return m_v;
    }

    template<std::size_t dim, class T, std::size_t size>
    auto FunctionBc<dim, T, size>::clone() const -> std::unique_ptr<base_t>
    {
        return std::make_unique<FunctionBc>(*this);
    }

    /////////////////////////
    // BcRegion definition //
    /////////////////////////
    template<std::size_t dim, class TInterval>
    struct BcRegion
    {
        using direction_t = xt::xtensor_fixed<int, xt::xshape<dim>>;
        using lca_t = LevelCellArray<dim, TInterval>;
        using region_t = std::pair<std::vector<direction_t>, std::vector<lca_t>>;

        virtual ~BcRegion() = default;
        BcRegion(const BcRegion&) = delete;
        BcRegion& operator=(const BcRegion&) = delete;
        BcRegion(BcRegion&&) = delete;
        BcRegion& operator=(BcRegion&&) = delete;

        virtual const region_t get_region(const lca_t&) const = 0;
        virtual std::unique_ptr<BcRegion> clone() const = 0;

    protected:
        BcRegion(){}
    };

    template<std::size_t dim, class TInterval>
    struct Everywhere: public BcRegion<dim, TInterval>
    {
        using base_t = BcRegion<dim, TInterval>;
        using direction_t = typename base_t::direction_t;
        using lca_t = typename base_t::lca_t;
        using region_t = typename base_t::region_t;

        Everywhere(){}
        Everywhere(const Everywhere&){}
        Everywhere& operator=(const Everywhere&){}

        const region_t get_region(const lca_t& mesh) const override;
        std::unique_ptr<base_t> clone() const override;
    };

    template<std::size_t dim, class TInterval>
    class CoordsRegion: public BcRegion<dim, TInterval>
    {
    public:
        using base_t = BcRegion<dim, TInterval>;
        using direction_t = typename base_t::direction_t;
        using lca_t = typename base_t::lca_t;
        using region_t = typename base_t::region_t;

        template <class Func>
        CoordsRegion(Func &&f);

        CoordsRegion(const CoordsRegion& r);
        CoordsRegion& operator=(const CoordsRegion& r);

        std::unique_ptr<base_t> clone() const override;
        const region_t get_region(const lca_t& mesh) const override;

    private:
        std::function<bool(const xt::xtensor_fixed<double, xt::xshape<dim>>&)> m_func;
    };

    template<std::size_t dim, class TInterval, class Set>
    class SetRegion: public BcRegion<dim, TInterval>
    {
    public:
        using base_t = BcRegion<dim, TInterval>;
        using direction_t = typename base_t::direction_t;
        using lca_t = typename base_t::lca_t;
        using region_t = typename base_t::region_t;

        SetRegion(const Set& set);

        SetRegion(const SetRegion& r);
        SetRegion& operator=(const SetRegion& r);

        std::unique_ptr<base_t> clone() const override;
        const region_t get_region(const lca_t& mesh) const override;

    private:
        Set m_set;
    };

    /////////////////////////////
    // BcRegion implementation //
    /////////////////////////////

    namespace detail
    {
        template<std::size_t dim>
        auto get_direction();

        template<>
        auto get_direction<1>()
        {
            return std::vector<xt::xtensor_fixed<int, xt::xshape<1>>> {{-1}, {1}};
        }

        template<>
        auto get_direction<2>()
        {
            return std::vector<xt::xtensor_fixed<int, xt::xshape<2>>> {{ 1,  0},
                                                                       {-1,  0},
                                                                       { 0,  1},
                                                                       { 0, -1}};
        }

        template<>
        auto get_direction<3>()
        {
            return std::vector<xt::xtensor_fixed<int, xt::xshape<3>>> {{ 1,  0,  0},
                                                                       {-1,  0,  0},
                                                                       { 0,  1,  0},
                                                                       { 0, -1,  0},
                                                                       { 0,  0,  1},
                                                                       { 0,  0, -1}};
        }
    }
    // Everywhere
    template<std::size_t dim, class TInterval>
    inline auto Everywhere<dim, TInterval>::get_region(const lca_t& domain) const -> const region_t
    {
        std::vector<direction_t> dir;
        std::vector<lca_t> lca;
        for (auto& d: detail::get_direction<dim>())
        {
            dir.emplace_back(-d);
            lca.emplace_back(difference(translate(domain, d), domain));
        }
        return std::make_pair(dir, lca);
    }

    template<std::size_t dim, class TInterval>
    auto Everywhere<dim, TInterval>::clone() const -> std::unique_ptr<base_t>
    {
        return std::make_unique<Everywhere>(*this);
    }

    // CoordsRegion
    template<std::size_t dim, class TInterval>
    template <class Func>
    CoordsRegion<dim, TInterval>::CoordsRegion(Func &&f)
    :m_func(std::forward<Func>(f))
    {}

    template<std::size_t dim, class TInterval>
    CoordsRegion<dim, TInterval>::CoordsRegion(const CoordsRegion& r)
    : m_func{r.m_func}
    {}

    template<std::size_t dim, class TInterval>
    CoordsRegion<dim, TInterval>& CoordsRegion<dim, TInterval>::operator=(const CoordsRegion& r)
    {
        return {r.m_func};
    }

    template<std::size_t dim, class TInterval>
    auto CoordsRegion<dim, TInterval>::clone() const -> std::unique_ptr<base_t>
    {
        return std::make_unique<CoordsRegion>(*this);
    }

    // TODO: must be implemented
    template<std::size_t dim, class TInterval>
    inline auto CoordsRegion<dim, TInterval>::get_region(const lca_t&) const -> const region_t
    {
        return std::make_pair(std::vector<direction_t>(), std::vector<lca_t>());
    }

    // SetRegion
    template<std::size_t dim, class TInterval, class Set>
    SetRegion<dim, TInterval, Set>::SetRegion(const Set& set)
    : m_set(set)
    {}

    template<std::size_t dim, class TInterval, class Set>
    SetRegion<dim, TInterval, Set>::SetRegion(const SetRegion& r)
    : m_set{r.m_set}
    {}

    template<std::size_t dim, class TInterval, class Set>
    SetRegion<dim, TInterval, Set>& SetRegion<dim, TInterval, Set>::operator=(const SetRegion& r)
    {
        return {r.m_set};
    }
    template<std::size_t dim, class TInterval, class Set>
    auto SetRegion<dim, TInterval, Set>::clone() const -> std::unique_ptr<base_t>
    {
        return std::make_unique<SetRegion>(*this);
    }

    template<std::size_t dim, class TInterval, class Set>
    inline auto SetRegion<dim, TInterval, Set>::get_region(const lca_t& domain) const -> const region_t
    {
        std::vector<direction_t> dir;
        std::vector<lca_t> lca;
        for (auto& d: detail::get_direction<dim>())
        {
            lca_t lca_temp = intersection(m_set, difference(translate(domain, d), domain));
            if (!lca_temp.empty())
            {
                dir.emplace_back(-d);
                lca.emplace_back(std::move(lca_temp));
            }
        }
        return std::make_pair(dir, lca);
    }

    ///////////////////////////////
    // BcRegion helper functions //
    ///////////////////////////////
    template<std::size_t dim, class TInterval, class F, class... CT>
    auto make_region(subset_operator<F, CT...> region)
    {
        return std::make_unique<SetRegion<dim, TInterval, subset_operator<F, CT...>>>(region);
    }

    template<std::size_t dim, class TInterval, class Func>
    auto make_region(Func&& func)
    {
        return std::make_unique<CoordsRegion<dim, TInterval>>(std::forward<Func>(func));
    }

    template<std::size_t dim, class TInterval>
    auto make_region(Everywhere<dim, TInterval>)
    {
        return std::make_unique<Everywhere<dim, TInterval>>();
    }

    ///////////////////
    // Bc definition //
    ///////////////////
    template<std::size_t dim, class TInterval, class T, std::size_t size>
    class Bc
    {
    public:
        using bcvalue_t = BcValue<dim, T, size>;
        using bcvalue_impl = std::unique_ptr<bcvalue_t>;
        using bcregion_t = BcRegion<dim, TInterval>;
        using bcregion_impl = std::unique_ptr<bcregion_t>;
        using lca_t = typename bcregion_t::lca_t;

        virtual ~Bc() = default;

        Bc(const lca_t& domain, const bcvalue_t& bcv);
        Bc(const lca_t& domain, const bcvalue_t& bcv, const bcregion_t& bcr);

        Bc(const Bc& bc);
        Bc& operator=(const Bc& bc);

        Bc(Bc&& bc) = default;
        Bc& operator=(Bc&& bc) = default;

        virtual std::unique_ptr<Bc> clone() const = 0;

        template<class Region>
        auto on(const Region& region);
        auto get_lca();

    private:
        bcvalue_impl p_bcvalue;
        bcregion_impl p_bcregion;
        const lca_t& m_domain;
    };

    ///////////////////
    // Bc definition //
    ///////////////////
    template<std::size_t dim, class TInterval, class T, std::size_t size>
    Bc<dim, TInterval, T, size>::Bc(const lca_t& domain, const bcvalue_t& bcv, const bcregion_t& bcr)
    : p_bcvalue(bcv.clone())
    , p_bcregion(bcr.clone())
    , m_domain(domain)
    {
    }

    template<std::size_t dim, class TInterval, class T, std::size_t size>
    Bc<dim, TInterval, T, size>::Bc(const lca_t& domain, const bcvalue_t& bcv)
    : p_bcvalue(bcv.clone())
    , p_bcregion(make_region(Everywhere<dim, TInterval>()))
    , m_domain(domain)
    {
    }

    template<std::size_t dim, class TInterval, class T, std::size_t size>
    Bc<dim, TInterval, T, size>::Bc(const Bc& bc)
    : p_bcvalue(bc.p_bcvalue->clone())
    , p_bcregion(bc.p_bcregion->clone())
    , m_domain(bc.m_domain)
    {
    }

    template<std::size_t dim, class TInterval, class T, std::size_t size>
    Bc<dim, TInterval, T, size>& Bc<dim, TInterval, T, size>::operator=(const Bc& bc)
    {
        bcvalue_impl bcvalue = bc.p_bcvalue->clone();
        bcregion_impl bcregion = bc.p_bcregion->clone();
        std::swap(p_bcvalue, bcvalue);
        std::swap(p_bcregion, bcregion);
        m_domain = bc.m_domain;
        return *this;
    }

    template<std::size_t dim, class TInterval, class T, std::size_t size>
    template<class Region>
    inline auto Bc<dim, TInterval, T, size>::on(const Region& region)
    {
        p_bcregion = make_region<dim, TInterval>(region);
        return this;
    }

    template<std::size_t dim, class TInterval, class T, std::size_t size>
    inline auto Bc<dim, TInterval, T, size>::get_lca()
    {
        return p_bcregion->get_region(m_domain);
    }

    /////////////////////////
    // Bc helper functions //
    /////////////////////////
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

    template<template<std::size_t, class, class, std::size_t> class bc_type, class Field>
    auto make_bc(Field& field, const std::function<detail::return_type_t<typename Field::value_type, Field::size>(const xt::xtensor_fixed<typename Field::value_type, xt::xshape<Field::dim>>&)>& func)
    {
        using value_t = typename Field::value_type;
        using interval_t = typename Field::interval_t;
        constexpr std::size_t dim = Field::dim;
        constexpr std::size_t size = Field::size;

        auto& mesh = detail::get_mesh(field.mesh());
        return field.attach(bc_type<dim, interval_t, value_t, size>(mesh, FunctionBc<dim, value_t, size>(func)));
    }

    template<template<std::size_t, class, class, std::size_t> class bc_type, class Field, class... T>
    auto make_bc(Field& field, typename Field::value_type v1, T... v)
    {
        using value_t = typename Field::value_type;
        using interval_t = typename Field::interval_t;
        constexpr std::size_t dim = Field::dim;
        constexpr std::size_t size = Field::size;

        static_assert(std::is_same_v<typename Field::value_type, std::common_type_t<typename Field::value_type, T...>>, "The constant value type must be the same as the field value_type");
        static_assert(Field::size == sizeof...(T) + 1, "The number of constant values should be equal to the number of element in the field");

        auto& mesh = detail::get_mesh(field.mesh());
        return field.attach(bc_type<dim, interval_t, value_t, size>(mesh, ConstantBc<dim, value_t, size>(v1, v...)));
    }

    //////////////
    // BC Types //
    //////////////
    template<std::size_t dim, class TInterval, class T, std::size_t size>
    struct Dirichlet: public Bc<dim, TInterval, T, size>
    {
        using base_t = Bc<dim, TInterval, T, size>;
        using Bc<dim, TInterval, T, size>::Bc;

        virtual std::unique_ptr<base_t> clone() const override
        {
            return std::make_unique<Dirichlet>(*this);
        }
    };


    template<std::size_t dim, class TInterval, class T, std::size_t size>
    struct Neumann: public Bc<dim, TInterval, T, size>
    {
        using base_t = Bc<dim, TInterval, T, size>;
        using Bc<dim, TInterval, T, size>::Bc;

        virtual std::unique_ptr<base_t> clone() const override
        {
            return std::make_unique<Neumann>(*this);
        }
    };

    // template<std::size_t dim, class TInterval, class T, std::size_t size>
    // struct Robin: public Bc<dim, TInterval, T, size>
    // {
    //     using Bc<dim, TInterval, T, size>::Bc;
    // };

} // namespace samurai