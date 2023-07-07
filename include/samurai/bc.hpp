// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <xtensor/xfixed.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor/xview.hpp>

#include "dispatch.hpp"
#include "samurai_config.hpp"
#include "static_algorithm.hpp"
#include "stencil.hpp"

namespace samurai
{
    enum class BCVType
    {
        constant = 0,
        function = 1,
    };

    template <class F, class... CT>
    class subset_operator;

    template <std::size_t dim, class TInterval>
    class LevelCellArray;

    template <std::size_t dim, class TInterval>
    class LevelCellList;

    template <class D, class Config>
    class Mesh_base;

    template <class Config>
    class UniformMesh;

    template <class mesh_t, class value_t, std::size_t size, bool SOA>
    class Field;

    namespace detail
    {
        template <class T, std::size_t size>
        struct return_type
        {
            using type                       = xt::xtensor_fixed<T, xt::xshape<size>>;
            static constexpr std::size_t dim = 2;
        };

        template <class T>
        struct return_type<T, 1>
        {
            using type                       = T;
            static constexpr std::size_t dim = 1;
        };

        template <class T, std::size_t size>
        using return_type_t = typename return_type<T, size>::type;

        template <class T, std::size_t size>
        void fill(xt::xtensor_fixed<T, xt::xshape<size>>& data, T value)
        {
            data.fill(value);
        }

        template <class T>
        void fill(T& data, T value)
        {
            data = value;
        }
    }

    ////////////////////////
    // BcValue definition //
    ////////////////////////
    template <std::size_t dim, class T, std::size_t size>
    struct BcValue
    {
        using value_t  = detail::return_type_t<T, size>;
        using coords_t = xt::xtensor_fixed<T, xt::xshape<dim>>;

        virtual ~BcValue()                 = default;
        BcValue(const BcValue&)            = delete;
        BcValue& operator=(const BcValue&) = delete;
        BcValue(BcValue&&)                 = delete;
        BcValue& operator=(BcValue&&)      = delete;

        virtual value_t get_value(const coords_t&) const = 0;
        virtual std::unique_ptr<BcValue> clone() const   = 0;
        virtual BCVType type() const                     = 0;

      protected:

        BcValue() = default;
    };

    template <std::size_t dim, class T, std::size_t size>
    class ConstantBc : public BcValue<dim, T, size>
    {
      public:

        using base_t   = BcValue<dim, T, size>;
        using value_t  = typename base_t::value_t;
        using coords_t = typename base_t::coords_t;

        template <class... CT>
        ConstantBc(const CT... v);

        ConstantBc();

        value_t get_value(const coords_t&) const override;
        std::unique_ptr<base_t> clone() const override;
        BCVType type() const override;

      private:

        value_t m_v;
    };

    template <std::size_t dim, class T, std::size_t size>
    class FunctionBc : public BcValue<dim, T, size>
    {
      public:

        using base_t     = BcValue<dim, T, size>;
        using value_t    = typename base_t::value_t;
        using coords_t   = typename base_t::coords_t;
        using function_t = std::function<value_t(const coords_t&)>;

        FunctionBc(const function_t& f);

        value_t get_value(const coords_t& coords) const override;
        std::unique_ptr<base_t> clone() const override;
        BCVType type() const override;

      private:

        function_t m_func;
    };

    ////////////////////////////
    // BcValue implementation //
    ////////////////////////////

    template <std::size_t dim, class T, std::size_t size>
    template <class... CT>
    ConstantBc<dim, T, size>::ConstantBc(const CT... v)
        : m_v{v...}
    {
    }

    template <std::size_t dim, class T, std::size_t size>
    ConstantBc<dim, T, size>::ConstantBc()
    {
        detail::fill(m_v, T{0});
    }

    template <std::size_t dim, class T, std::size_t size>
    inline auto ConstantBc<dim, T, size>::get_value(const coords_t&) const -> value_t
    {
        return m_v;
    }

    template <std::size_t dim, class T, std::size_t size>
    auto ConstantBc<dim, T, size>::clone() const -> std::unique_ptr<base_t>
    {
        return std::make_unique<ConstantBc>(m_v);
    }

    template <std::size_t dim, class T, std::size_t size>
    inline BCVType ConstantBc<dim, T, size>::type() const
    {
        return BCVType::constant;
    }

    template <std::size_t dim, class T, std::size_t size>
    FunctionBc<dim, T, size>::FunctionBc(const function_t& f)
        : m_func(f)
    {
    }

    template <std::size_t dim, class T, std::size_t size>
    inline auto FunctionBc<dim, T, size>::get_value(const coords_t& coords) const -> value_t
    {
        return m_func(coords);
    }

    template <std::size_t dim, class T, std::size_t size>
    auto FunctionBc<dim, T, size>::clone() const -> std::unique_ptr<base_t>
    {
        return std::make_unique<FunctionBc>(m_func);
    }

    template <std::size_t dim, class T, std::size_t size>
    inline BCVType FunctionBc<dim, T, size>::type() const
    {
        return BCVType::function;
    }

    /////////////////////////
    // BcRegion definition //
    /////////////////////////
    template <std::size_t dim, class TInterval>
    struct BcRegion
    {
        using direction_t = xt::xtensor_fixed<int, xt::xshape<dim>>;
        using lca_t       = LevelCellArray<dim, TInterval>;
        using region_t    = std::pair<std::vector<direction_t>, std::vector<lca_t>>;

        virtual ~BcRegion()                  = default;
        BcRegion(const BcRegion&)            = delete;
        BcRegion& operator=(const BcRegion&) = delete;
        BcRegion(BcRegion&&)                 = delete;
        BcRegion& operator=(BcRegion&&)      = delete;

        virtual region_t get_region(const lca_t&) const = 0;
        virtual std::unique_ptr<BcRegion> clone() const = 0;

      protected:

        BcRegion() = default;
    };

    template <std::size_t dim, class TInterval>
    struct Everywhere : public BcRegion<dim, TInterval>
    {
        using base_t      = BcRegion<dim, TInterval>;
        using direction_t = typename base_t::direction_t;
        using lca_t       = typename base_t::lca_t;
        using region_t    = typename base_t::region_t;

        Everywhere() = default;

        region_t get_region(const lca_t& domain) const override;
        std::unique_ptr<base_t> clone() const override;
    };

    template <std::size_t dim, class TInterval, std::size_t nd>
    class OnDirection : public BcRegion<dim, TInterval>
    {
      public:

        using base_t      = BcRegion<dim, TInterval>;
        using direction_t = typename base_t::direction_t;
        using lca_t       = typename base_t::lca_t;
        using region_t    = typename base_t::region_t;

        OnDirection(const std::array<direction_t, nd>& d);

        region_t get_region(const lca_t& domain) const override;
        std::unique_ptr<base_t> clone() const override;

      private:

        std::array<direction_t, nd> m_d;
    };

    template <std::size_t dim, class TInterval>
    class CoordsRegion : public BcRegion<dim, TInterval>
    {
      public:

        using base_t      = BcRegion<dim, TInterval>;
        using direction_t = typename base_t::direction_t;
        using lca_t       = typename base_t::lca_t;
        using region_t    = typename base_t::region_t;
        using function_t  = std::function<bool(const xt::xtensor_fixed<double, xt::xshape<dim>>&)>;

        CoordsRegion(const function_t& f);

        std::unique_ptr<base_t> clone() const override;
        region_t get_region(const lca_t& domain) const override;

      private:

        function_t m_func;
    };

    template <std::size_t dim, class TInterval, class Set>
    class SetRegion : public BcRegion<dim, TInterval>
    {
      public:

        using base_t      = BcRegion<dim, TInterval>;
        using direction_t = typename base_t::direction_t;
        using lca_t       = typename base_t::lca_t;
        using region_t    = typename base_t::region_t;

        SetRegion(const Set& set);

        std::unique_ptr<base_t> clone() const override;
        region_t get_region(const lca_t& domain) const override;

      private:

        Set m_set;
    };

    /////////////////////////////
    // BcRegion implementation //
    /////////////////////////////

    // Everywhere
    template <std::size_t dim, class TInterval>
    inline auto Everywhere<dim, TInterval>::get_region(const lca_t& domain) const -> region_t
    {
        std::vector<direction_t> dir;
        std::vector<lca_t> lca;

        static_nested_loop<dim, -1, 2>(
            [&](auto& stencil)
            {
                int number_of_one = xt::sum(xt::abs(stencil))[0];
                if (number_of_one > 0)
                {
                    dir.emplace_back(stencil);
                }

                if (number_of_one == 1)
                {
                    lca.emplace_back(difference(domain, translate(domain, -stencil)));
                }
                else if (number_of_one > 1)
                {
                    if constexpr (dim == 2)
                    {
                        lca.emplace_back(difference(
                            domain,
                            union_(translate(domain, direction_t{-stencil[0], 0}), translate(domain, direction_t{0, -stencil[1]}))));
                    }
                    else if constexpr (dim == 3)
                    {
                        lca.emplace_back(difference(domain,
                                                    union_(translate(domain, direction_t{-stencil[0], 0, 0}),
                                                           translate(domain, direction_t{0, -stencil[1], 0}),
                                                           translate(domain, direction_t{0, 0, -stencil[2]}))));
                    }
                }
            });

        return std::make_pair(dir, lca);
    }

    template <std::size_t dim, class TInterval>
    auto Everywhere<dim, TInterval>::clone() const -> std::unique_ptr<base_t>
    {
        return std::make_unique<Everywhere>();
    }

    // OnDirection
    template <std::size_t dim, class TInterval, std::size_t nd>
    OnDirection<dim, TInterval, nd>::OnDirection(const std::array<direction_t, nd>& d)
        : m_d(d)
    {
    }

    template <std::size_t dim, class TInterval, std::size_t nd>
    inline auto OnDirection<dim, TInterval, nd>::get_region(const lca_t& domain) const -> region_t
    {
        std::vector<direction_t> dir;
        std::vector<lca_t> lca;

        for (auto& stencil : m_d)
        {
            int number_of_one = xt::sum(xt::abs(stencil))[0];
            if (number_of_one > 0)
            {
                dir.emplace_back(stencil);
            }

            if (number_of_one == 1)
            {
                lca.emplace_back(difference(domain, translate(domain, -stencil)));
            }
            else if (number_of_one > 1)
            {
                if constexpr (dim == 2)
                {
                    lca.emplace_back(
                        difference(domain,
                                   union_(translate(domain, direction_t{-stencil[0], 0}), translate(domain, direction_t{0, -stencil[1]}))));
                }
                else if constexpr (dim == 3)
                {
                    lca.emplace_back(difference(domain,
                                                union_(translate(domain, direction_t{-stencil[0], 0, 0}),
                                                       translate(domain, direction_t{0, -stencil[1], 0}),
                                                       translate(domain, direction_t{0, 0, -stencil[2]}))));
                }
            }
        }

        return std::make_pair(dir, lca);
    }

    template <std::size_t dim, class TInterval, std::size_t nd>
    auto OnDirection<dim, TInterval, nd>::clone() const -> std::unique_ptr<base_t>
    {
        return std::make_unique<OnDirection>(m_d);
    }

    // CoordsRegion
    template <std::size_t dim, class TInterval>
    CoordsRegion<dim, TInterval>::CoordsRegion(const function_t& f)
        : m_func(f)
    {
    }

    template <std::size_t dim, class TInterval>
    auto CoordsRegion<dim, TInterval>::clone() const -> std::unique_ptr<base_t>
    {
        return std::make_unique<CoordsRegion>(m_func);
    }

    // TODO: must be implemented
    template <std::size_t dim, class TInterval>
    inline auto CoordsRegion<dim, TInterval>::get_region(const lca_t&) const -> region_t
    {
        std::cerr << "CoordsRegion::get_region() not implemented" << std::endl;
        assert(false && "To be implemented");
        return std::make_pair(std::vector<direction_t>(), std::vector<lca_t>());
    }

    // SetRegion
    template <std::size_t dim, class TInterval, class Set>
    SetRegion<dim, TInterval, Set>::SetRegion(const Set& set)
        : m_set(set)
    {
    }

    template <std::size_t dim, class TInterval, class Set>
    auto SetRegion<dim, TInterval, Set>::clone() const -> std::unique_ptr<base_t>
    {
        return std::make_unique<SetRegion>(m_set);
    }

    template <std::size_t dim, class TInterval, class Set>
    inline auto SetRegion<dim, TInterval, Set>::get_region(const lca_t& domain) const -> region_t
    {
        std::vector<direction_t> dir;
        std::vector<lca_t> lca;
        for (std::size_t d = 0; d < 2 * dim; ++d)
        {
            DirectionVector<dim> stencil = xt::view(cartesian_directions<dim>(), d);
            lca_t lca_temp               = intersection(m_set, difference(translate(domain, d), domain));
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
    template <std::size_t dim, class TInterval, class F, class... CT>
    auto make_region(subset_operator<F, CT...> region)
    {
        return SetRegion<dim, TInterval, subset_operator<F, CT...>>(region);
    }

    template <std::size_t dim, class TInterval, class Func>
    auto make_region(Func&& func)
    {
        return CoordsRegion<dim, TInterval>(std::forward<Func>(func));
    }

    template <std::size_t dim, class TInterval>
    auto make_region(Everywhere<dim, TInterval>)
    {
        return Everywhere<dim, TInterval>();
    }

    template <std::size_t dim, class TInterval, std::size_t nd>
    auto make_region(const std::array<xt::xtensor_fixed<int, xt::xshape<dim>>, nd>& d)
    {
        return OnDirection<dim, TInterval, nd>(d);
    }

    template <std::size_t dim, class TInterval, class... dir_t>
    auto make_region(const dir_t&... d)
    {
        constexpr std::size_t nd = sizeof...(dir_t);
        using final_type         = OnDirection<dim, TInterval, nd>;
        using direction_t        = typename final_type::direction_t;
        return final_type(std::array<direction_t, nd>{d...});
    }

    template <std::size_t dim, class TInterval>
    auto make_region(const xt::xtensor_fixed<int, xt::xshape<dim>>& d)
    {
        return OnDirection<dim, TInterval, 1>({d});
    }

    ///////////////////
    // Bc definition //
    ///////////////////
    template <class Field>
    class Bc
    {
      public:

        static constexpr std::size_t dim  = Field::dim;
        static constexpr std::size_t size = Field::size;
        using interval_t                  = typename Field::interval_t;

        using bcvalue_t    = BcValue<dim, typename Field::value_type, size>;
        using bcvalue_impl = std::unique_ptr<bcvalue_t>;
        using value_t      = typename bcvalue_t::value_t;
        using coords_t     = typename bcvalue_t::coords_t;

        using bcregion_t = BcRegion<dim, interval_t>;
        using lca_t      = typename bcregion_t::lca_t;
        using region_t   = typename bcregion_t::region_t;

        virtual ~Bc() = default;

        Bc(const lca_t& domain, const bcvalue_t& bcv);
        Bc(const lca_t& domain, const bcvalue_t& bcv, const bcregion_t& bcr);

        Bc(const Bc& bc);
        Bc& operator=(const Bc& bc);

        Bc(Bc&& bc) noexcept            = default;
        Bc& operator=(Bc&& bc) noexcept = default;

        virtual std::unique_ptr<Bc> clone() const = 0;

        template <class Region>
        auto on(const Region& region);

        template <class... Regions>
        auto on(const Regions&... regions);

        auto get_region() const;

        template <class Direction>
        void update_values(const Direction& d,
                           std::size_t level,
                           const interval_t& i,
                           xt::xtensor_fixed<typename interval_t::value_t, xt::xshape<dim - 1>> index);

        value_t constant_value();
        value_t value(const coords_t& coords) const;
        const auto& value() const;
        BCVType get_value_type() const;

      private:

        bcvalue_impl p_bcvalue;
        const lca_t& m_domain; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
        region_t m_region;
        xt::xtensor<typename Field::value_type, detail::return_type<typename Field::value_type, size>::dim> m_value;
    };

    ///////////////////
    // Bc definition //
    ///////////////////
    template <class Field>
    Bc<Field>::Bc(const lca_t& domain, const bcvalue_t& bcv, const bcregion_t& bcr)
        : p_bcvalue(bcv.clone())
        , m_domain(domain)
        , m_region(bcr.get_region(domain))
    {
    }

    template <class Field>
    Bc<Field>::Bc(const lca_t& domain, const bcvalue_t& bcv)
        : p_bcvalue(bcv.clone())
        , m_domain(domain)
        , m_region(Everywhere<dim, interval_t>().get_region(domain))
    {
    }

    template <class Field>
    Bc<Field>::Bc(const Bc& bc)
        : p_bcvalue(bc.p_bcvalue->clone())
        , m_domain(bc.m_domain)
        , m_region(bc.m_region)
    {
    }

    template <class Field>
    Bc<Field>& Bc<Field>::operator=(const Bc& bc)
    {
        if (this == &bc)
        {
            return *this;
        }
        bcvalue_impl bcvalue = bc.p_bcvalue->clone();
        std::swap(p_bcvalue, bcvalue);
        m_domain = bc.m_domain;
        m_region = bc.m_region;
        return *this;
    }

    template <class Field>
    template <class Region>
    inline auto Bc<Field>::on(const Region& region)
    {
        m_region = make_region<dim, interval_t>(region).get_region(m_domain);
        return this;
    }

    template <class Field>
    template <class... Regions>
    inline auto Bc<Field>::on(const Regions&... regions)
    {
        m_region = make_region<dim, interval_t>(regions...).get_region(m_domain);
        return this;
    }

    template <class Field>
    inline auto Bc<Field>::get_region() const
    {
        return m_region;
    }

    template <class Field>
    template <class Direction>
    void Bc<Field>::update_values(const Direction& dir,
                                  std::size_t level,
                                  const interval_t& i,
                                  xt::xtensor_fixed<typename interval_t::value_t, xt::xshape<dim - 1>> index)
    {
        if (p_bcvalue->type() == BCVType::function)
        {
            coords_t coords;
            const double dx = 1. / (1 << level);

            coords[0] = dx * i.start + 0.5 * (1 + dir[0]) * dx;
            for (std::size_t d = 1; d < dim; ++d)
            {
                coords[d] = dx * index[d - 1] + 0.5 * (1 + dir[d]) * dx;
            }

            if constexpr (size == 1)
            {
                m_value.resize({i.size()});
            }
            else
            {
                m_value.resize({i.size(), size});
            }

            for (std::size_t ii = 0; ii < i.size(); ++ii)
            {
                xt::view(m_value, ii) = p_bcvalue->get_value(coords);
                coords[0] += dx;
            }
        }
    }

    template <class Field>
    inline auto Bc<Field>::constant_value() -> value_t
    {
        return p_bcvalue->get_value({});
    }

    template <class Field>
    inline const auto& Bc<Field>::value() const
    {
        return m_value;
    }

    template <class Field>
    inline auto Bc<Field>::value(const coords_t& coords) const -> value_t
    {
        return p_bcvalue->get_value(coords);
    }

    template <class Field>
    inline BCVType Bc<Field>::get_value_type() const
    {
        return p_bcvalue->type();
    }

    /////////////////////////
    // Bc helper functions //
    /////////////////////////
    namespace detail
    {
        template <std::size_t dim, class TInterval>
        decltype(auto) get_mesh(const LevelCellArray<dim, TInterval>& mesh)
        {
            return mesh;
        }

        template <class D, class Config>
        decltype(auto) get_mesh(const Mesh_base<D, Config>& mesh)
        {
            return mesh.domain();
        }

        template <class Config>
        decltype(auto) get_mesh(const UniformMesh<Config>& mesh)
        {
            using mesh_id_t = typename Config::mesh_id_t;
            return mesh[mesh_id_t::cells];
        }
    }

    template <template <class> class bc_type, class Field>
    auto make_bc(Field& field,
                 const std::function<detail::return_type_t<typename Field::value_type, Field::size>(
                     const xt::xtensor_fixed<typename Field::mesh_t::interval_t::value_t, xt::xshape<Field::dim>>&)>& func)
    {
        using value_t              = typename Field::value_type;
        constexpr std::size_t dim  = Field::dim;
        constexpr std::size_t size = Field::size;

        auto& mesh = detail::get_mesh(field.mesh());
        return field.attach_bc(bc_type<Field>(mesh, FunctionBc<dim, value_t, size>(func)));
    }

    template <template <class> class bc_type, class Field>
    auto make_bc(Field& field)
    {
        using value_t              = typename Field::value_type;
        constexpr std::size_t dim  = Field::dim;
        constexpr std::size_t size = Field::size;

        auto& mesh = detail::get_mesh(field.mesh());
        return field.attach_bc(bc_type<Field>(mesh, ConstantBc<dim, value_t, size>()));
    }

    template <template <class> class bc_type, class Field, class... T>
    auto make_bc(Field& field, typename Field::value_type v1, T... v)
    {
        using value_t              = typename Field::value_type;
        constexpr std::size_t dim  = Field::dim;
        constexpr std::size_t size = Field::size;

        static_assert(std::is_same_v<typename Field::value_type, std::common_type_t<typename Field::value_type, T...>>,
                      "The constant value type must be the same as the field value_type");
        static_assert(Field::size == sizeof...(T) + 1,
                      "The number of constant values should be equal to the "
                      "number of element in the field");

        auto& mesh = detail::get_mesh(field.mesh());
        return field.attach_bc(bc_type<Field>(mesh, ConstantBc<dim, value_t, size>(v1, v...)));
    }

    //////////////
    // BC Types //
    //////////////
    template <class Field>
    struct Dirichlet : public Bc<Field>
    {
        using base_t = Bc<Field>;
        using Bc<Field>::Bc;

        std::unique_ptr<base_t> clone() const override
        {
            return std::make_unique<Dirichlet>(*this);
        }
    };

    template <class Field>
    struct Neumann : public Bc<Field>
    {
        using base_t = Bc<Field>;
        using Bc<Field>::Bc;

        std::unique_ptr<base_t> clone() const override
        {
            return std::make_unique<Neumann>(*this);
        }
    };

    template <class Field>
    void apply_bc_impl(Dirichlet<Field>& bc, std::size_t level, Field& field)
    {
        static constexpr std::size_t dim = Field::dim;
        constexpr int ghost_width        = std::max(static_cast<int>(Field::mesh_t::config::max_stencil_width),
                                             static_cast<int>(Field::mesh_t::config::prediction_order));

        using mesh_id_t = typename Field::mesh_t::mesh_id_t;
        auto& mesh      = field.mesh()[mesh_id_t::reference];

        auto region     = bc.get_region();
        auto& direction = region.first;
        auto& lca       = region.second;
        for (std::size_t d = 0; d < direction.size(); ++d)
        {
            bool is_periodic = false;
            for (std::size_t i = 0; i < dim; ++i)
            {
                if (direction[d](i) != 0 && field.mesh().is_periodic(i))
                {
                    is_periodic = true;
                    break;
                }
            }
            if (!is_periodic)
            {
                std::size_t delta_l = lca[d].level() - level;
                for (int ig = 0; ig < ghost_width; ++ig)
                {
                    auto first_layer_ghosts = intersection(intersection(mesh[level], translate(lca[d], (ig + 1) * (direction[d] << delta_l))),
                                                           translate(mesh[level], (2 * ig + 1) * direction[d]))
                                                  .on(level);
                    first_layer_ghosts(
                        [&](const auto& i, const auto& index)
                        {
                            if (bc.get_value_type() == BCVType::constant)
                            {
                                if constexpr (dim == 1)
                                {
                                    field(level, i) = 2 * bc.constant_value() - field(level, i - (2 * ig + 1) * direction[d][0]);
                                }
                                else if constexpr (dim == 2)
                                {
                                    auto j             = index[0];
                                    field(level, i, j) = 2 * bc.constant_value()
                                                       - field(level, i - (2 * ig + 1) * direction[d][0], j - (2 * ig + 1) * direction[d][1]);
                                }
                                else if constexpr (dim == 3)
                                {
                                    auto j                = index[0];
                                    auto k                = index[1];
                                    field(level, i, j, k) = 2 * bc.constant_value()
                                                          - field(level,
                                                                  i - (2 * ig + 1) * direction[d][0],
                                                                  j - (2 * ig + 1) * direction[d][1],
                                                                  k - (2 * ig + 1) * direction[d][2]);
                                }
                            }
                            else if (bc.get_value_type() == BCVType::function)
                            {
                                bc.update_values(direction[d], level, i, index);

                                if constexpr (dim == 1)
                                {
                                    field(level, i) = 2 * bc.value() - field(level, i - (2 * ig + 1) * direction[d][0]);
                                }
                                else if constexpr (dim == 2)
                                {
                                    auto j             = index[0];
                                    field(level, i, j) = 2 * bc.value()
                                                       - field(level, i - (2 * ig + 1) * direction[d][0], j - (2 * ig + 1) * direction[d][1]);
                                }
                                else if constexpr (dim == 3)
                                {
                                    auto j                = index[0];
                                    auto k                = index[1];
                                    field(level, i, j, k) = 2 * bc.value()
                                                          - field(level,
                                                                  i - (2 * ig + 1) * direction[d][0],
                                                                  j - (2 * ig + 1) * direction[d][1],
                                                                  k - (2 * ig + 1) * direction[d][2]);
                                }
                            }
                        });
                }
            }
        }
    }

    template <class Field>
    void apply_bc_impl(Neumann<Field>& bc, std::size_t level, Field& field)
    {
        static constexpr std::size_t dim = Field::dim;
        constexpr int ghost_width        = std::max(static_cast<int>(Field::mesh_t::config::max_stencil_width),
                                             static_cast<int>(Field::mesh_t::config::prediction_order));

        using mesh_id_t = typename Field::mesh_t::mesh_id_t;
        auto& mesh      = field.mesh()[mesh_id_t::reference];

        auto region     = bc.get_region();
        auto& direction = region.first;
        auto& lca       = region.second;
        for (std::size_t d = 0; d < direction.size(); ++d)
        {
            bool is_periodic = false;
            for (std::size_t i = 0; i < dim; ++i)
            {
                if (direction[d](i) != 0 && field.mesh().is_periodic(i))
                {
                    is_periodic = true;
                    break;
                }
            }
            if (!is_periodic)
            {
                std::size_t delta_l = lca[d].level() - level;
                for (int ig = 0; ig < ghost_width; ++ig)
                {
                    auto first_layer_ghosts = intersection(intersection(mesh[level], translate(lca[d], (ig + 1) * (direction[d] << delta_l))),
                                                           translate(mesh[level], (2 * ig + 1) * direction[d]))
                                                  .on(level);
                    first_layer_ghosts(
                        [&](const auto& i, const auto& index)
                        {
                            const double dx = 1. / (1 << level);
                            if (bc.get_value_type() == BCVType::constant)
                            {
                                if constexpr (dim == 1)
                                {
                                    field(level, i) = dx * bc.constant_value() + field(level, i - (2 * ig + 1) * direction[d][0]);
                                }
                                else if constexpr (dim == 2)
                                {
                                    auto j             = index[0];
                                    field(level, i, j) = dx * bc.constant_value()
                                                       + field(level, i - (ig + 1) * direction[d][0], j - (2 * ig + 1) * direction[d][1]);
                                }
                                else if constexpr (dim == 3)
                                {
                                    auto j                = index[0];
                                    auto k                = index[1];
                                    field(level, i, j, k) = dx * bc.constant_value()
                                                          + field(level,
                                                                  i - (2 * ig + 1) * direction[d][0],
                                                                  j - (2 * ig + 1) * direction[d][1],
                                                                  k - (2 * ig + 1) * direction[d][2]);
                                }
                            }
                            else if (bc.get_value_type() == BCVType::function)
                            {
                                bc.update_values(direction[d], level, i, index);

                                if constexpr (dim == 1)
                                {
                                    field(level, i) = dx * bc.value() + field(level, i - (2 * ig + 1) * direction[d][0]);
                                }
                                else if constexpr (dim == 2)
                                {
                                    auto j             = index[0];
                                    field(level, i, j) = dx * bc.value()
                                                       + field(level, i - (2 * ig + 1) * direction[d][0], j - (2 * ig + 1) * direction[d][1]);
                                }
                                else if constexpr (dim == 3)
                                {
                                    auto j                = index[0];
                                    auto k                = index[1];
                                    field(level, i, j, k) = dx * bc.value()
                                                          + field(level,
                                                                  i - (2 * ig + 1) * direction[d][0],
                                                                  j - (2 * ig + 1) * direction[d][1],
                                                                  k - (2 * ig + 1) * direction[d][2]);
                                }
                            }
                        });
                }
            }
        }
    }

    struct select_bc_functor
    {
        template <class BC, class Field>
        void run(BC& bc, std::size_t level, Field& field) const
        {
            return apply_bc_impl(bc, level, field);
        }

        template <class BC, class Field>
        void run(BC& bc, Field& field) const
        {
            return apply_bc_impl(bc, field);
        }

        template <class Field>
        void on_error(Bc<Field>&, std::size_t, Field&) const
        {
            std::cerr << "BC not known" << std::endl;
        }

        template <class Field>
        void on_error(Bc<Field>&, Field&) const
        {
            std::cerr << "BC not known" << std::endl;
        }
    };

    template <class Field>
    using select_bc_dispatcher = unit_static_dispatcher<select_bc_functor, Bc<Field>, BC_TYPES::types<Field>>;

    template <class BCType, class Field>
    void apply_bc_impl(BCType& bc, Field& field)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;
        auto& mesh      = field.mesh()[mesh_id_t::reference];

        for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
        {
            apply_bc_impl(bc, level, field);
        }
    }

    template <class Field>
    void update_bc(std::size_t level, Field& field)
    {
        for (auto& bc : field.get_bc())
        {
            select_bc_dispatcher<Field>::dispatch(*bc.get(), level, field);
        }
    }

    template <class Field, class... Fields>
    void update_bc(std::size_t level, Field& field, Fields&... fields)
    {
        update_bc(level, field);
        update_bc(level, fields...);
    }

    template <class Field>
    void update_bc(Field& field)
    {
        for (auto& bc : field.get_bc())
        {
            select_bc_dispatcher<Field>::dispatch(*bc.get(), field);
        }
    }

    template <class Field, class... Fields>
    void update_bc(Field& field, Fields&... fields)
    {
        update_bc(field);
        update_bc(fields...);
    }

} // namespace samurai
