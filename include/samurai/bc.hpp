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

#include "boundary.hpp"
#include "samurai/cell.hpp"
#include "samurai_config.hpp"
#include "static_algorithm.hpp"
#include "stencil.hpp"

#define INIT_BC(NAME, STENCIL_SIZE)                                                            \
    using base_t  = samurai::Bc<Field>;                                                        \
    using cell_t  = typename base_t::cell_t;                                                   \
    using value_t = typename base_t::value_t;                                                  \
    using base_t::Bc;                                                                          \
    using base_t::dim;                                                                         \
                                                                                               \
    using stencil_t               = samurai::Stencil<STENCIL_SIZE, dim>;                       \
    using constant_stencil_size_t = std::integral_constant<std::size_t, STENCIL_SIZE>;         \
    using stencil_cells_t         = std::array<cell_t, STENCIL_SIZE>;                          \
                                                                                               \
    static_assert(STENCIL_SIZE <= base_t::max_stencil_size, "The stencil size is too large."); \
                                                                                               \
    std::unique_ptr<base_t> clone() const override                                             \
    {                                                                                          \
        return std::make_unique<NAME>(*this);                                                  \
    }                                                                                          \
                                                                                               \
    std::size_t stencil_size() const override                                                  \
    {                                                                                          \
        return STENCIL_SIZE;                                                                   \
    }

#define APPLY_AND_STENCIL_FUNCTIONS(STENCIL_SIZE)                                                       \
    virtual void apply(Field&, const std::array<cell_t, STENCIL_SIZE>&, const value_t&) const           \
    {                                                                                                   \
        assert(false);                                                                                  \
    }                                                                                                   \
                                                                                                        \
    virtual Stencil<STENCIL_SIZE, dim> stencil(std::integral_constant<std::size_t, STENCIL_SIZE>) const \
    {                                                                                                   \
        return line_stencil<dim, 0, STENCIL_SIZE>();                                                    \
    }

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
    template <class Field>
    struct BcValue
    {
        static constexpr std::size_t dim = Field::dim;
        using value_t                    = detail::return_type_t<typename Field::value_type, Field::size>;
        using coords_t                   = xt::xtensor_fixed<double, xt::xshape<dim>>;
        using direction_t                = xt::xtensor_fixed<int, xt::xshape<dim>>;
        using cell_t                     = typename Field::cell_t;

        virtual ~BcValue()                 = default;
        BcValue(const BcValue&)            = delete;
        BcValue& operator=(const BcValue&) = delete;
        BcValue(BcValue&&)                 = delete;
        BcValue& operator=(BcValue&&)      = delete;

        virtual value_t get_value(const direction_t& d, const cell_t&, const coords_t&) const = 0;
        virtual std::unique_ptr<BcValue> clone() const                                        = 0;
        virtual BCVType type() const                                                          = 0;

      protected:

        BcValue() = default;
    };

    template <class Field>
    class ConstantBc : public BcValue<Field>
    {
      public:

        using base_t      = BcValue<Field>;
        using value_t     = typename base_t::value_t;
        using coords_t    = typename base_t::coords_t;
        using direction_t = typename base_t::direction_t;
        using cell_t      = typename base_t::cell_t;

        template <class... CT>
        ConstantBc(const CT... v);

        ConstantBc();

        value_t get_value(const direction_t&, const cell_t&, const coords_t&) const override;
        std::unique_ptr<base_t> clone() const override;
        BCVType type() const override;

      private:

        value_t m_v;
    };

    template <class Field>
    class FunctionBc : public BcValue<Field>
    {
      public:

        using base_t      = BcValue<Field>;
        using value_t     = typename base_t::value_t;
        using coords_t    = typename base_t::coords_t;
        using direction_t = typename base_t::direction_t;
        using cell_t      = typename base_t::cell_t;
        using function_t  = std::function<value_t(const direction_t&, const cell_t&, const coords_t&)>;

        FunctionBc(const function_t& f);

        value_t get_value(const direction_t& d, const cell_t& cell_in, const coords_t& coords) const override;
        std::unique_ptr<base_t> clone() const override;
        BCVType type() const override;

      private:

        function_t m_func;
    };

    ////////////////////////////
    // BcValue implementation //
    ////////////////////////////

    template <class Field>
    template <class... CT>
    ConstantBc<Field>::ConstantBc(const CT... v)
        : m_v{v...}
    {
    }

    template <class Field>
    ConstantBc<Field>::ConstantBc()
    {
        detail::fill(m_v, typename Field::value_type{0});
    }

    template <class Field>
    inline auto ConstantBc<Field>::get_value(const direction_t&, const cell_t&, const coords_t&) const -> value_t
    {
        return m_v;
    }

    template <class Field>
    auto ConstantBc<Field>::clone() const -> std::unique_ptr<base_t>
    {
        return std::make_unique<ConstantBc>(m_v);
    }

    template <class Field>
    inline BCVType ConstantBc<Field>::type() const
    {
        return BCVType::constant;
    }

    template <class Field>
    FunctionBc<Field>::FunctionBc(const function_t& f)
        : m_func(f)
    {
    }

    template <class Field>
    inline auto FunctionBc<Field>::get_value(const direction_t& d, const cell_t& cell_in, const coords_t& coords) const -> value_t
    {
        return m_func(d, cell_in, coords);
    }

    template <class Field>
    auto FunctionBc<Field>::clone() const -> std::unique_ptr<base_t>
    {
        return std::make_unique<FunctionBc>(m_func);
    }

    template <class Field>
    inline BCVType FunctionBc<Field>::type() const
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
        using mesh_t                      = typename Field::mesh_t;
        using interval_t                  = typename Field::interval_t;

        using bcvalue_t    = BcValue<Field>;
        using bcvalue_impl = std::unique_ptr<bcvalue_t>;
        using direction_t  = typename bcvalue_t::direction_t;
        using value_t      = typename bcvalue_t::value_t;
        using coords_t     = typename bcvalue_t::coords_t;
        using cell_t       = typename bcvalue_t::cell_t;

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
        virtual std::size_t stencil_size() const  = 0;

        static constexpr std::size_t max_stencil_size = 10;
        APPLY_AND_STENCIL_FUNCTIONS(1)
        APPLY_AND_STENCIL_FUNCTIONS(2)
        APPLY_AND_STENCIL_FUNCTIONS(3)
        APPLY_AND_STENCIL_FUNCTIONS(4)
        APPLY_AND_STENCIL_FUNCTIONS(5)
        APPLY_AND_STENCIL_FUNCTIONS(6)
        APPLY_AND_STENCIL_FUNCTIONS(7)
        APPLY_AND_STENCIL_FUNCTIONS(8)
        APPLY_AND_STENCIL_FUNCTIONS(9)
        APPLY_AND_STENCIL_FUNCTIONS(10)

        template <class Region>
        auto on(const Region& region);

        template <class... Regions>
        auto on(const Regions&... regions);

        auto get_region() const;

        // void update_values(const mesh_t& mesh,
        //                    const direction_t& d,
        //                    std::size_t level,
        //                    const interval_t& i,
        //                    xt::xtensor_fixed<typename interval_t::value_t, xt::xshape<dim - 1>> index);

        value_t constant_value();
        value_t value(const direction_t& d, const cell_t& cell_in, const coords_t& coords) const;
        // const auto& value() const;
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

    // template <class Field>
    // void Bc<Field>::update_values(const mesh_t& mesh,
    //                               const direction_t& dir,
    //                               std::size_t level,
    //                               const interval_t& i,
    //                               xt::xtensor_fixed<typename interval_t::value_t, xt::xshape<dim - 1>> index)
    // {
    //     if (p_bcvalue->type() == BCVType::function)
    //     {
    //         coords_t coords;
    //         cell_t cell_in;

    //         const double dx = 1. / (1 << level);

    //         cell_in.level = level;
    //         auto shift    = dir[0] < 0 ? -dir[0] : -dir[0] + 1;
    //         coords[0]     = dx * (i.start + shift);
    //         for (std::size_t d = 1; d < dim; ++d)
    //         {
    //             shift              = dir[d] < 0 ? -dir[d] : -dir[d] + 1;
    //             coords[d]          = dx * (index[d - 1] + shift);
    //             cell_in.indices[d] = index[d - 1] - dir[d];
    //         }

    //         if constexpr (size == 1)
    //         {
    //             m_value.resize({i.size()});
    //         }
    //         else
    //         {
    //             m_value.resize({i.size(), size});
    //         }

    //         cell_in.indices[0] = i.start - dir[0];
    //         cell_in.index      = mesh.get_index(level, cell_in.indices);
    //         cell_in.length     = cell_length(level);

    //         for (std::size_t ii = 0; ii < i.size(); ++ii)
    //         {
    //             xt::view(m_value, ii) = p_bcvalue->get_value(dir, cell_in, coords);
    //             coords[0] += dx;
    //             ++cell_in.indices[0];
    //             ++cell_in.index;
    //         }
    //     }
    // }

    template <class Field>
    inline auto Bc<Field>::constant_value() -> value_t
    {
        return p_bcvalue->get_value({}, {}, {});
    }

    // template <class Field>
    // inline const auto& Bc<Field>::value() const
    // {
    //     return m_value;
    // }

    template <class Field>
    inline auto Bc<Field>::value(const direction_t& d, const cell_t& cell_in, const coords_t& coords) const -> value_t
    {
        return p_bcvalue->get_value(d, cell_in, coords);
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
    auto make_bc(Field& field, typename FunctionBc<Field>::function_t func)
    {
        auto& mesh = detail::get_mesh(field.mesh());
        return field.attach_bc(bc_type<Field>(mesh, FunctionBc<Field>(func)));
    }

    template <template <class> class bc_type, class Field>
    auto make_bc(Field& field)
    {
        auto& mesh = detail::get_mesh(field.mesh());
        return field.attach_bc(bc_type<Field>(mesh, ConstantBc<Field>()));
    }

    template <template <class> class bc_type, class Field, class... T>
    auto make_bc(Field& field, typename Field::value_type v1, T... v)
    {
        static_assert(std::is_same_v<typename Field::value_type, std::common_type_t<typename Field::value_type, T...>>,
                      "The constant value type must be the same as the field value_type");
        static_assert(Field::size == sizeof...(T) + 1,
                      "The number of constant values should be equal to the "
                      "number of element in the field");

        auto& mesh = detail::get_mesh(field.mesh());
        return field.attach_bc(bc_type<Field>(mesh, ConstantBc<Field>(v1, v...)));
    }

    //////////////
    // BC Types //
    //////////////
    template <class Field, std::size_t stencil_size>
    void apply_bc_impl(Bc<Field>& bc, std::size_t level, Field& field)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;

        static constexpr std::size_t dim = Field::dim;

        auto& mesh = field.mesh();

        auto region           = bc.get_region();
        auto& direction       = region.first;
        auto& lca             = region.second;
        auto stencil_0        = bc.stencil(std::integral_constant<std::size_t, stencil_size>());
        bool is_line_stencil_ = is_line_stencil(stencil_0);

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
                bool is_cartesian_direction = is_cartesian(direction[d]);

                if (is_cartesian_direction || is_line_stencil_) // Treat diagonals only if it's a line stencil
                {
                    auto stencil = convert_for_direction(stencil_0, direction[d]);

                    // 1. Inner cells in the boundary region
                    auto bdry_cells = intersection(mesh[mesh_id_t::cells][level], lca[d]).on(level);

                    // 2. Inner ghosts in the boundary region that have a neigbouring ghost outside the domain
                    auto translated_outer_nghbr        = translate(mesh[mesh_id_t::reference][level], -direction[d]);
                    auto inner_cells_and_ghosts        = intersection(translated_outer_nghbr, lca[d]);
                    auto inner_ghosts_with_outer_nghbr = difference(inner_cells_and_ghosts, bdry_cells).on(level);

                    if (bc.get_value_type() == BCVType::constant)
                    {
                        auto value = bc.constant_value();
                        for_each_stencil(mesh,
                                         bdry_cells,
                                         stencil,
                                         [&, value](auto& cells)
                                         {
                                             bc.apply(field, cells, value);
                                         });
                        if (is_cartesian_direction)
                        {
                            for_each_stencil(mesh,
                                             inner_ghosts_with_outer_nghbr,
                                             stencil,
                                             [&, value](auto& cells)
                                             {
                                                 bc.apply(field, cells, value);
                                             });
                        }
                    }
                    else if (bc.get_value_type() == BCVType::function)
                    {
                        int origin_index = find_stencil_origin(stencil);
                        assert(origin_index >= 0);
                        for_each_stencil(mesh,
                                         bdry_cells,
                                         stencil,
                                         [&, origin_index](auto& cells)
                                         {
                                             auto& cell_in    = cells[static_cast<std::size_t>(origin_index)];
                                             auto face_coords = cell_in.face_center(direction[d]);
                                             auto value       = bc.value(direction[d], cell_in, face_coords);
                                             bc.apply(field, cells, value);
                                         });
                        if (is_cartesian_direction)
                        {
                            for_each_stencil(mesh,
                                             inner_ghosts_with_outer_nghbr,
                                             stencil,
                                             [&, origin_index](auto& cells)
                                             {
                                                 auto& cell_in    = cells[static_cast<std::size_t>(origin_index)];
                                                 auto face_coords = cell_in.face_center(direction[d]);
                                                 auto value       = bc.value(direction[d], cell_in, face_coords);
                                                 bc.apply(field, cells, value);
                                             });
                        }
                    }
                    else
                    {
                        std::cerr << "Unknown BC type" << std::endl;
                        exit(EXIT_FAILURE);
                    }
                }
            }
        }
    }

    template <class Field>
    struct Dirichlet : public Bc<Field>
    {
        INIT_BC(Dirichlet, 2)

        stencil_t stencil(constant_stencil_size_t) const override
        {
            return line_stencil<dim, 0>(0, 1);
        }

        void apply(Field& f, const stencil_cells_t& cells, const value_t& value) const override
        {
            static constexpr std::size_t in  = 0;
            static constexpr std::size_t out = 1;

            f[cells[out]] = 2 * value - f[cells[in]];
        }
    };

    template <class Field>
    struct Neumann : public Bc<Field>
    {
        INIT_BC(Neumann, 2)

        stencil_t stencil(constant_stencil_size_t) const override
        {
            return line_stencil<dim, 0>(0, 1);
        }

        void apply(Field& f, const stencil_cells_t& cells, const value_t& value) const override
        {
            static constexpr std::size_t in  = 0;
            static constexpr std::size_t out = 1;

            double dx     = cell_length(cells[out].level);
            f[cells[out]] = dx * value + f[cells[in]];
        }
    };

    template <class Field>
    void update_bc(std::size_t level, Field& field)
    {
        static constexpr std::size_t max_stencil_size = Bc<Field>::max_stencil_size;

        for (auto& bc : field.get_bc())
        {
            static_for<1, max_stencil_size + 1>::apply( // for (int i=1; i<=max_stencil_size; i++)
                [&](auto integral_constant_i)
                {
                    static constexpr int i = decltype(integral_constant_i)::value;

                    if (bc->stencil_size() == i)
                    {
                        apply_bc_impl<Field, i>(*bc.get(), level, field);
                    }
                });
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
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;
        auto& mesh      = field.mesh()[mesh_id_t::reference];

        for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
        {
            update_bc(level, field);
        }
    }

    template <class Field, class... Fields>
    void update_bc(Field& field, Fields&... fields)
    {
        update_bc(field);
        update_bc(fields...);
    }

} // namespace samurai
