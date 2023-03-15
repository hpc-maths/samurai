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
#include <xtensor/xnoalias.hpp>
#include <xtensor/xio.hpp>

namespace samurai
{
    enum class BCType
    {
        dirichlet = 0,
        neumann = 1,
        periodic = 2,
        interpolation = 3 // Reconstruct the function by linear approximation
    };

    enum class BCVType
    {
        constant = 0,
        function = 1,
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
            static constexpr std::size_t dim = 2;
        };

        template<class T>
        struct return_type<T, 1>
        {
            using type = T;
            static constexpr std::size_t dim = 1;
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

        virtual value_t get_value(const coords_t&) const = 0;
        virtual std::unique_ptr<BcValue> clone() const = 0;
        virtual BCVType type() const = 0;
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

        value_t get_value(const coords_t&) const override;
        std::unique_ptr<base_t> clone() const override;
        BCVType type() const override;
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

        value_t get_value(const coords_t& coords) const override;
        std::unique_ptr<base_t> clone() const override;
        BCVType type() const override;
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
    inline auto ConstantBc<dim, T, size>::get_value(const coords_t&) const -> value_t
    {
        return m_v;
    }

    template<std::size_t dim, class T, std::size_t size>
    auto ConstantBc<dim, T, size>::clone() const -> std::unique_ptr<base_t>
    {
        return std::make_unique<ConstantBc>(*this);
    }

    template<std::size_t dim, class T, std::size_t size>
    inline BCVType ConstantBc<dim, T, size>::type() const
    {
        return BCVType::constant;
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
    inline auto FunctionBc<dim, T, size>::get_value(const coords_t& coords) const -> value_t
    {
        return m_func(coords);
    }

    template<std::size_t dim, class T, std::size_t size>
    auto FunctionBc<dim, T, size>::clone() const -> std::unique_ptr<base_t>
    {
        return std::make_unique<FunctionBc>(*this);
    }

    template<std::size_t dim, class T, std::size_t size>
    inline BCVType FunctionBc<dim, T, size>::type() const
    {
        return BCVType::function;
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
        return SetRegion<dim, TInterval, subset_operator<F, CT...>>(region);
    }

    template<std::size_t dim, class TInterval, class Func>
    auto make_region(Func&& func)
    {
        return CoordsRegion<dim, TInterval>(std::forward<Func>(func));
    }

    template<std::size_t dim, class TInterval>
    auto make_region(Everywhere<dim, TInterval>)
    {
        return Everywhere<dim, TInterval>();
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
        using value_t = typename bcvalue_t::value_t;
        using coords_t = typename bcvalue_t::coords_t;


        using bcregion_t = BcRegion<dim, TInterval>;
        using lca_t = typename bcregion_t::lca_t;
        using region_t = typename bcregion_t::region_t;

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
        auto get_region() const;

        template<class Direction>
        void update_values(const Direction& d, std::size_t level, const TInterval& i, const xt::xtensor_fixed<typename TInterval::value_t, xt::xshape<dim-1>> index);

        value_t constant_value();
        const auto& value() const;
        BCVType get_value_type() const;

    private:
        bcvalue_impl p_bcvalue;
        const lca_t& m_domain;
        region_t m_region;
        xt::xtensor<T, detail::return_type<T, size>::dim> m_value;
    };

    ///////////////////
    // Bc definition //
    ///////////////////
    template<std::size_t dim, class TInterval, class T, std::size_t size>
    Bc<dim, TInterval, T, size>::Bc(const lca_t& domain, const bcvalue_t& bcv, const bcregion_t& bcr)
    : p_bcvalue(bcv.clone())
    , m_domain(domain)
    , m_region(bcr.get_region(domain))
    {
    }

    template<std::size_t dim, class TInterval, class T, std::size_t size>
    Bc<dim, TInterval, T, size>::Bc(const lca_t& domain, const bcvalue_t& bcv)
    : p_bcvalue(bcv.clone())
    , m_domain(domain)
    , m_region(Everywhere<dim, TInterval>().get_region(domain))
    {
    }

    template<std::size_t dim, class TInterval, class T, std::size_t size>
    Bc<dim, TInterval, T, size>::Bc(const Bc& bc)
    : p_bcvalue(bc.p_bcvalue->clone())
    , m_domain(bc.m_domain)
    , m_region(bc.m_region)
    {
    }

    template<std::size_t dim, class TInterval, class T, std::size_t size>
    Bc<dim, TInterval, T, size>& Bc<dim, TInterval, T, size>::operator=(const Bc& bc)
    {
        bcvalue_impl bcvalue = bc.p_bcvalue->clone();
        std::swap(p_bcvalue, bcvalue);
        m_domain = bc.m_domain;
        m_region = bc.m_region;
        return *this;
    }

    template<std::size_t dim, class TInterval, class T, std::size_t size>
    template<class Region>
    inline auto Bc<dim, TInterval, T, size>::on(const Region& region)
    {
        m_region = make_region<dim, TInterval>(region).get_region(m_domain);
        return this;
    }

    template<std::size_t dim, class TInterval, class T, std::size_t size>
    inline auto Bc<dim, TInterval, T, size>::get_region() const
    {
        return m_region;
    }

    template<std::size_t dim, class TInterval, class T, std::size_t size>
    template<class Direction>
    void Bc<dim, TInterval, T, size>::update_values(const Direction& dir, std::size_t level, const TInterval& i, const xt::xtensor_fixed<typename TInterval::value_t, xt::xshape<dim-1>> index)
    {
        if (p_bcvalue.get()->type() == BCVType::function)
        {
            coords_t coords;
            double dx = 1./(1<<level);

            coords[0] = dx*i.start + 0.5*(1 + dir[0])*dx;
            for(std::size_t d = 1; d < dim; ++d)
            {
                coords[d] = dx*index[d-1] + 0.5*( 1 + dir[d])*dx;
            }

            if constexpr (size == 1)
            {
                m_value.resize({i.size()});
            }
            else
            {
                m_value.resize({i.size(), size});
            }

            for(std::size_t ii = 0; ii < i.size(); ++ii)
            {
                coords[0] += dx;
                xt::view(m_value, ii) = p_bcvalue->get_value(coords);
            }
        }

    }

    template<std::size_t dim, class TInterval, class T, std::size_t size>
    inline auto Bc<dim, TInterval, T, size>::constant_value() -> value_t
    {
        return p_bcvalue.get()->get_value({});
    }

    template<std::size_t dim, class TInterval, class T, std::size_t size>
    inline const auto& Bc<dim, TInterval, T, size>::value() const
    {
        return m_value;
    }

    template<std::size_t dim, class TInterval, class T, std::size_t size>
    inline BCVType Bc<dim, TInterval, T, size>::get_value_type() const
    {
        return p_bcvalue.get()->type();
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

    template<class Direction, class TValue, class TIndex>
    auto coordinates(const Direction& d, std::size_t level, const Interval<TValue, TIndex>& i, TValue j)
    {
        double dx = 1./(1<<level);
        std::array<std::size_t, 2> shape{i.size(), 2};
        std::vector<xt::xtensor_fixed<double, xt::xshape<2>>> coords(i.size());
        for(std::size_t ii=0; ii < i.size(); ++ii)
        {
            coords[ii] = {dx*(i.start + static_cast<TValue>(ii)) + 0.5*(1 + d[0])*dx,  dx*j + 0.5*( 1 + d[1])*dx};
        }
        return coords;
    }

    template<std::size_t dim, class TInterval, class T, std::size_t size, class Field>
    void apply_bc_impl(Dirichlet<dim, TInterval, T, size>& bc, std::size_t level, Field& field)
    {
        constexpr int ghost_width = std::max(static_cast<int>(Field::mesh_t::config::max_stencil_width),
                                             static_cast<int>(Field::mesh_t::config::prediction_order));

        using value_t = typename Field::value_type;
        using bcvalue_t = typename Dirichlet<dim, TInterval, T, size>::value_t;
        constexpr std::size_t field_size = Field::size;

        using mesh_id_t = typename Field::mesh_t::mesh_id_t;
        auto& mesh = field.mesh()[mesh_id_t::reference];

        auto region = bc.get_region();
        auto& direction = region.first;
        auto& lca = region.second;
        for (std::size_t d=0; d < direction.size(); ++d)
        {
            auto set = intersection(mesh[level], lca[d]).on(level);
            set([&](const auto& i, const auto& index)
            {
                if (bc.get_value_type() == BCVType::constant)
                {
                    for (int ig=0; ig < ghost_width; ++ig)
                    {
                        if constexpr (dim == 1)
                        {
                            field(level, i - ig*direction[d][0]) = 2*bc.constant_value() - field(level, i + (ig + 1)*direction[d][0]);
                        }
                        else if constexpr (dim == 2)
                        {
                            auto j = index[0];
                            field(level, i - ig*direction[d][0], j - ig*direction[d][1]) = 2*bc.constant_value() - field(level, i + (ig + 1)*direction[d][0], j + (ig + 1)*direction[d][1]);
                        }
                        else if constexpr (dim == 3)
                        {
                            auto j = index[0];
                            auto k = index[1];
                            field(level, i - ig*direction[d][0], j - ig*direction[d][1], k - ig*direction[d][2]) = 2*bc.constant_value() - field(level, i + (ig + 1)*direction[d][0], j + (ig + 1)*direction[d][1], k + (ig + 1)*direction[d][2]);
                        }
                    }
                }
                else if (bc.get_value_type() == BCVType::function)
                {
                    bc.update_values(direction[d], level, i, index);

                    for (int ig=0; ig < ghost_width; ++ig)
                    {
                        if constexpr (dim == 1)
                        {
                            field(level, i - ig*direction[d][0]) = 2*bc.value() - field(level, i + (ig + 1)*direction[d][0]);
                        }
                        else if constexpr (dim == 2)
                        {
                            auto j = index[0];
                            field(level, i - ig*direction[d][0], j - ig*direction[d][1]) = 2*bc.value() - field(level, i + (ig + 1)*direction[d][0], j + (ig + 1)*direction[d][1]);
                        }
                        else if constexpr (dim == 3)
                        {
                            auto j = index[0];
                            auto k = index[1];
                            field(level, i - ig*direction[d][0], j - ig*direction[d][1], k - ig*direction[d][2]) = 2*bc.value() - field(level, i + (ig + 1)*direction[d][0], j + (ig + 1)*direction[d][1], k + (ig + 1)*direction[d][2]);
                        }
                    }
                }
            });
        }
    }

    template<std::size_t dim, class TInterval, class T, std::size_t size, class Field>
    void apply_bc_impl(Neumann<dim, TInterval, T, size>& bc, std::size_t level, Field& field)
    {
        constexpr int ghost_width = std::max(static_cast<int>(Field::mesh_t::config::max_stencil_width),
                                             static_cast<int>(Field::mesh_t::config::prediction_order));

        using value_t = typename Field::value_type;
        using bcvalue_t = typename Neumann<dim, TInterval, T, size>::value_t;
        constexpr std::size_t field_size = Field::size;

        using mesh_id_t = typename Field::mesh_t::mesh_id_t;
        auto& mesh = field.mesh()[mesh_id_t::reference];

        auto region = bc.get_region();
        auto& direction = region.first;
        auto& lca = region.second;
        for (std::size_t d=0; d < direction.size(); ++d)
        {
            auto set = intersection(mesh[level], lca[d]).on(level);
            set([&](const auto& i, const auto& index)
            {
                double dx = 1./(1<<level);
                if (bc.get_value_type() == BCVType::constant)
                {
                    for (int ig=0; ig < ghost_width; ++ig)
                    {
                        if constexpr (dim == 1)
                        {
                            field(level, i - ig*direction[d][0]) = dx*bc.constant_value() + field(level, i + (ig + 1)*direction[d][0]);
                        }
                        else if constexpr (dim == 2)
                        {
                            auto j = index[0];
                            field(level, i - ig*direction[d][0], j - ig*direction[d][1]) = dx*bc.constant_value() + field(level, i + (ig + 1)*direction[d][0], j + (ig + 1)*direction[d][1]);
                        }
                        else if constexpr (dim == 3)
                        {
                            auto j = index[0];
                            auto k = index[1];
                            field(level, i - ig*direction[d][0], j - ig*direction[d][1], k - ig*direction[d][2]) = dx*bc.constant_value() + field(level, i + (ig + 1)*direction[d][0], j + (ig + 1)*direction[d][1], k + (ig + 1)*direction[d][2]);
                        }
                    }
                }
                else if (bc.get_value_type() == BCVType::function)
                {
                    bc.update_values(direction[d], level, i, index);

                    for (int ig=0; ig < ghost_width; ++ig)
                    {
                        if constexpr (dim == 1)
                        {
                            field(level, i - ig*direction[d][0]) = dx*bc.value() + field(level, i + (ig + 1)*direction[d][0]);
                        }
                        else if constexpr (dim == 2)
                        {
                            auto j = index[0];
                            field(level, i - ig*direction[d][0], j - ig*direction[d][1]) = dx*bc.value() + field(level, i + (ig + 1)*direction[d][0], j + (ig + 1)*direction[d][1]);
                        }
                        else if constexpr (dim == 3)
                        {
                            auto j = index[0];
                            auto k = index[1];
                            field(level, i - ig*direction[d][0], j - ig*direction[d][1], k - ig*direction[d][2]) = dx*bc.value() + field(level, i + (ig + 1)*direction[d][0], j + (ig + 1)*direction[d][1], k + (ig + 1)*direction[d][2]);
                        }
                    }
                }
            });
        }
    }

    template<class BCType, class Field>
    void apply_bc_impl(BCType& bc, Field& field)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;
        auto& mesh = field.mesh()[mesh_id_t::reference];

        for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
        {
            apply_bc_impl(bc, level, field);
        }
    }

    template<std::size_t dim, class TInterval, class T, std::size_t size, class Field>
    void apply_bc(std::unique_ptr<Bc<dim, TInterval, T, size>>& bc, std::size_t level, Field& field)
    {
        if (dynamic_cast<Dirichlet<dim, TInterval, T, size>*>(bc.get()))
        {
            apply_bc_impl(*dynamic_cast<Dirichlet<dim, TInterval, T, size>*>(bc.get()), level, field);
        }
        else if (dynamic_cast<Neumann<dim, TInterval, T, size>*>(bc.get()))
        {
            apply_bc_impl(*dynamic_cast<Neumann<dim, TInterval, T, size>*>(bc.get()), level, field);
        }
    }

    template<class Field>
    void update_bc(std::size_t level, Field& field)
    {
        for(auto& bc: field.get_bc())
        {
            apply_bc(bc, level, field);
        }
    }

    template<std::size_t dim, class TInterval, class T, std::size_t size, class Field>
    void apply_bc(std::unique_ptr<Bc<dim, TInterval, T, size>>& bc, Field& field)
    {
        if (dynamic_cast<Dirichlet<dim, TInterval, T, size>*>(bc.get()))
        {
            apply_bc_impl(*dynamic_cast<Dirichlet<dim, TInterval, T, size>*>(bc.get()), field);
        }
        else if (dynamic_cast<Neumann<dim, TInterval, T, size>*>(bc.get()))
        {
            apply_bc_impl(*dynamic_cast<Neumann<dim, TInterval, T, size>*>(bc.get()), field);
        }
    }

    template<class Field>
    void update_bc(Field& field)
    {
        for(auto& bc: field.get_bc())
        {
            apply_bc(bc, field);
        }
    }
    // template<std::size_t dim, class TInterval, class T, std::size_t size>
    // struct Robin: public Bc<dim, TInterval, T, size>
    // {
    //     using Bc<dim, TInterval, T, size>::Bc;
    // };

} // namespace samurai