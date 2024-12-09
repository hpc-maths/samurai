// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "samurai/cell.hpp"
#include "samurai_config.hpp"
#include "static_algorithm.hpp"
#include "stencil.hpp"
#include "storage/containers.hpp"

#define APPLY_AND_STENCIL_FUNCTIONS(STENCIL_SIZE)                                                                                         \
    using apply_function_##STENCIL_SIZE = std::function<void(Field&, const std::array<cell_t, STENCIL_SIZE>&, const value_t&)>;           \
    virtual apply_function_##STENCIL_SIZE get_apply_function(std::integral_constant<std::size_t, STENCIL_SIZE>, const direction_t&) const \
    {                                                                                                                                     \
        return [](Field&, const std::array<cell_t, STENCIL_SIZE>&, const value_t&)                                                        \
        {                                                                                                                                 \
            assert(false);                                                                                                                \
        };                                                                                                                                \
    }                                                                                                                                     \
                                                                                                                                          \
    virtual Stencil<STENCIL_SIZE, dim> get_stencil(std::integral_constant<std::size_t, STENCIL_SIZE>) const                               \
    {                                                                                                                                     \
        return line_stencil<dim, 0, STENCIL_SIZE>();                                                                                      \
    }

#define INIT_BC(NAME, STENCIL_SIZE)                                                                                                \
    using base_t      = samurai::Bc<Field>;                                                                                        \
    using cell_t      = typename base_t::cell_t;                                                                                   \
    using value_t     = typename base_t::value_t;                                                                                  \
    using direction_t = typename base_t::direction_t;                                                                              \
    using base_t::base_t;                                                                                                          \
    using base_t::dim;                                                                                                             \
    using base_t::get_apply_function;                                                                                              \
    using base_t::get_stencil;                                                                                                     \
                                                                                                                                   \
    using stencil_t               = samurai::Stencil<STENCIL_SIZE, dim>;                                                           \
    using constant_stencil_size_t = std::integral_constant<std::size_t, STENCIL_SIZE>;                                             \
    using stencil_cells_t         = std::array<cell_t, STENCIL_SIZE>;                                                              \
    using apply_function_t        = std::function<void(Field&, const std::array<cell_t, STENCIL_SIZE>&, const value_t&)>;          \
                                                                                                                                   \
    static_assert(STENCIL_SIZE <= base_t::max_stencil_size_implemented, "The stencil size is too large.");                         \
    static_assert(Field::mesh_t::config::ghost_width >= STENCIL_SIZE / 2, "Not enough ghost layers for this boundary condition."); \
                                                                                                                                   \
    std::unique_ptr<base_t> clone() const override                                                                                 \
    {                                                                                                                              \
        return std::make_unique<NAME>(*this);                                                                                      \
    }                                                                                                                              \
                                                                                                                                   \
    std::size_t stencil_size() const override                                                                                      \
    {                                                                                                                              \
        return STENCIL_SIZE;                                                                                                       \
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

    ////////////////////////
    // BcValue definition //
    ////////////////////////
    template <class Field>
    struct BcValue
    {
        static constexpr std::size_t dim = Field::dim;
        using value_t                    = CollapsArray<typename Field::value_type, Field::size, Field::is_soa>;
        using coords_t                   = xt::xtensor_fixed<double, xt::xshape<dim>>;
        using direction_t                = DirectionVector<dim>;
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
        fill(m_v, typename Field::value_type{0});
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
        using direction_t = DirectionVector<dim>;
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
        using namespace math;
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

    template <std::size_t dim, class TInterval>
    inline auto CoordsRegion<dim, TInterval>::get_region(const lca_t& domain) const -> region_t
    {
        using lcl_t = LevelCellList<dim, TInterval>;

        std::vector<direction_t> dir;
        std::vector<lca_t> lca;

        static_nested_loop<dim, -1, 2>(
            [&](auto& dir_vector)
            {
                int number_of_one = xt::sum(xt::abs(dir_vector))[0];

                if (number_of_one == 1)
                {
                    auto bdry_dir = difference(domain, translate(domain, -dir_vector));

                    lcl_t cell_list(domain.level());

                    for_each_cell(domain,
                                  bdry_dir,
                                  [&](auto& cell)
                                  {
                                      if (m_func(cell.face_center(dir_vector)))
                                      {
                                          cell_list.add_cell(cell);
                                      }
                                  });

                    if (!cell_list.empty())
                    {
                        dir.emplace_back(dir_vector);
                        lca.emplace_back(cell_list);
                    }
                }
            });
        return std::make_pair(dir, lca);
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
    auto make_bc_region(subset_operator<F, CT...> region)
    {
        return SetRegion<dim, TInterval, subset_operator<F, CT...>>(region);
    }

    template <std::size_t dim, class TInterval>
    auto make_bc_region(const typename CoordsRegion<dim, TInterval>::function_t& func)
    {
        return CoordsRegion<dim, TInterval>(func);
    }

    template <class Mesh>
    auto make_bc_region(const Mesh&, const typename CoordsRegion<Mesh::dim, typename Mesh::interval_t>::function_t& func)
    {
        return CoordsRegion<Mesh::dim, typename Mesh::interval_t>(func);
    }

    template <std::size_t dim, class TInterval>
    auto make_bc_region(Everywhere<dim, TInterval>)
    {
        return Everywhere<dim, TInterval>();
    }

    template <std::size_t dim, class TInterval, std::size_t nd>
    auto make_bc_region(const std::array<xt::xtensor_fixed<int, xt::xshape<dim>>, nd>& d)
    {
        return OnDirection<dim, TInterval, nd>(d);
    }

    template <std::size_t dim, class TInterval, class... dir_t>
    auto make_bc_region(const dir_t&... d)
    {
        constexpr std::size_t nd = sizeof...(dir_t);
        using final_type         = OnDirection<dim, TInterval, nd>;
        using direction_t        = typename final_type::direction_t;
        return final_type(std::array<direction_t, nd>{d...});
    }

    template <std::size_t dim, class TInterval>
    auto make_bc_region(const xt::xtensor_fixed<int, xt::xshape<dim>>& d)
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
        Bc(const lca_t& domain, const bcvalue_t& bcv, bool dummy);

        Bc(const Bc& bc);
        Bc& operator=(const Bc& bc);

        Bc(Bc&& bc) noexcept            = default;
        Bc& operator=(Bc&& bc) noexcept = default;

        virtual std::unique_ptr<Bc> clone() const = 0;
        virtual std::size_t stencil_size() const  = 0;

        static constexpr std::size_t max_stencil_size_implemented = 10;
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

        const region_t& get_region() const;

        value_t constant_value();
        value_t value(const direction_t& d, const cell_t& cell_in, const coords_t& coords) const;
        BCVType get_value_type() const;

      private:

        bcvalue_impl p_bcvalue;
        const lca_t& m_domain; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
        region_t m_region;
        // xt::xtensor<typename Field::value_type, detail::return_type<typename Field::value_type, size>::dim> m_value;
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
    Bc<Field>::Bc(const lca_t& domain, const bcvalue_t& bcv, bool)
        : p_bcvalue(bcv.clone())
        , m_domain(domain)
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
        if constexpr (std::is_base_of_v<BcRegion<dim, interval_t>, Region>)
        {
            m_region = region.get_region(m_domain);
        }
        else
        {
            m_region = make_bc_region<dim, interval_t>(region).get_region(m_domain);
        }
        return this;
    }

    template <class Field>
    template <class... Regions>
    inline auto Bc<Field>::on(const Regions&... regions)
    {
        m_region = make_bc_region<dim, interval_t>(regions...).get_region(m_domain);
        return this;
    }

    template <class Field>
    inline auto Bc<Field>::get_region() const -> const region_t&
    {
        return m_region;
    }

    template <class Field>
    inline auto Bc<Field>::constant_value() -> value_t
    {
        return p_bcvalue->get_value({}, {}, {});
    }

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

    /**
     * Boundary condition as a function
     */
    template <template <class> class bc_type, class Field>
    auto make_bc(Field& field, typename FunctionBc<Field>::function_t func)
    {
        auto& mesh = detail::get_mesh(field.mesh());
        return field.attach_bc(bc_type<Field>(mesh, FunctionBc<Field>(func)));
    }

    template <class bc_type, class Field>
    auto make_bc(Field& field, typename FunctionBc<Field>::function_t func)
    {
        using bc_impl = typename bc_type::template impl_t<Field>;

        auto& mesh = detail::get_mesh(field.mesh());
        return field.attach_bc(bc_impl(mesh, FunctionBc<Field>(func)));
    }

    /**
     * Boundary condition as a default constant
     */
    template <template <class> class bc_type, class Field>
    auto make_bc(Field& field)
    {
        auto& mesh = detail::get_mesh(field.mesh());
        return field.attach_bc(bc_type<Field>(mesh, ConstantBc<Field>()));
    }

    template <class bc_type, class Field>
    auto make_bc(Field& field)
    {
        using bc_impl = typename bc_type::template impl_t<Field>;

        auto& mesh = detail::get_mesh(field.mesh());
        return field.attach_bc(bc_impl(mesh, ConstantBc<Field>()));
    }

    /**
     * Boundary condition as a constant
     */
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

    template <class bc_type, class Field, class... T>
    auto make_bc(Field& field, typename Field::value_type v1, T... v)
    {
        static_assert(std::is_same_v<typename Field::value_type, std::common_type_t<typename Field::value_type, T...>>,
                      "The constant value type must be the same as the field value_type");
        static_assert(Field::size == sizeof...(T) + 1,
                      "The number of constant values should be equal to the "
                      "number of element in the field");

        using bc_impl = typename bc_type::template impl_t<Field>;

        auto& mesh = detail::get_mesh(field.mesh());
        return field.attach_bc(bc_impl(mesh, ConstantBc<Field>(v1, v...)));
    }

    //////////////
    // BC Types //
    //////////////
    template <class Field, class Subset, std::size_t stencil_size, class Vector>
    void
    __apply_bc_on_subset(Bc<Field>& bc, Field& field, Subset& subset, const Stencil<stencil_size, Field::dim>& stencil, const Vector& direction)
    {
        auto apply_bc = bc.get_apply_function(std::integral_constant<std::size_t, stencil_size>(), direction);
        if (bc.get_value_type() == BCVType::constant)
        {
            auto value = bc.constant_value();
            for_each_stencil(field.mesh(),
                             subset,
                             stencil,
                             [&, value](auto& cells)
                             {
                                 apply_bc(field, cells, value);
                             });
        }
        else if (bc.get_value_type() == BCVType::function)
        {
            int origin_index = find_stencil_origin(stencil);
            assert(origin_index >= 0);
            for_each_stencil(field.mesh(),
                             subset,
                             stencil,
                             [&, origin_index](auto& cells)
                             {
                                 auto& cell_in    = cells[static_cast<std::size_t>(origin_index)];
                                 auto face_coords = cell_in.face_center(direction);
                                 auto value       = bc.value(direction, cell_in, face_coords);
                                 apply_bc(field, cells, value);
                             });
        }
        else
        {
            std::cerr << "Unknown BC type" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    template <class Field, std::size_t stencil_size>
    void apply_bc_impl(Bc<Field>& bc, std::size_t level, Field& field)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;

        static constexpr std::size_t dim = Field::dim;

        auto& mesh = field.mesh();

        auto& region    = bc.get_region();
        auto& direction = region.first;
        auto& lca       = region.second;
        auto stencil_0  = bc.get_stencil(std::integral_constant<std::size_t, stencil_size>());

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

                if (is_cartesian_direction)
                {
                    auto stencil = convert_for_direction(stencil_0, direction[d]);

                    // 1. Inner cells in the boundary region
                    auto bdry_cells = intersection(mesh[mesh_id_t::cells][level], lca[d]).on(level);

                    __apply_bc_on_subset(bc, field, bdry_cells, stencil, direction[d]);

                    // 2. Inner ghosts in the boundary region that have a neighbouring ghost outside the domain
                    if (mesh.min_level() != mesh.max_level())
                    {
                        auto translated_outer_nghbr = translate(mesh[mesh_id_t::reference][level], -(stencil_size / 2) * direction[d]);
                        auto inner_cells_and_ghosts = intersection(translated_outer_nghbr, lca[d]);
                        auto inner_ghosts_with_outer_nghbr = difference(inner_cells_and_ghosts, bdry_cells).on(level);

                        __apply_bc_on_subset(bc, field, inner_ghosts_with_outer_nghbr, stencil, direction[d]);
                    }
                }
            }
        }
    }

    template <std::size_t stencil_size, class Field, class Subset>
    void __apply_extrapolation_bc_on_subset(Bc<Field>& bc,
                                            std::size_t level,
                                            Field& field,
                                            const DirectionVector<Field::dim>& direction,
                                            Subset& subset,
                                            bool only_fill_ghost_neighbours)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;

        auto& mesh = field.mesh();

        auto stencil_0 = bc.get_stencil(std::integral_constant<std::size_t, stencil_size>());
        auto stencil   = convert_for_direction(stencil_0, direction);

        // 1. Inner cells in the boundary region
        if (!only_fill_ghost_neighbours)
        {
            auto bdry_cells = intersection(mesh[mesh_id_t::cells][level], subset);
            // We need to check that the furthest ghost exists. It's not always the case for large stencils!
            auto translated_outer_nghbr = translate(mesh[mesh_id_t::reference][level], -(stencil_size / 2) * direction);
            auto cells                  = intersection(translated_outer_nghbr, bdry_cells).on(level);

            __apply_bc_on_subset(bc, field, cells, stencil, direction);
        }

        // 2. Inner ghosts in the boundary region that have a neighbouring ghost outside the domain
        {
            auto bdry_cells                    = intersection(mesh[mesh_id_t::cells][level], subset);
            auto translated_outer_nghbr        = translate(mesh[mesh_id_t::reference][level], -(stencil_size / 2) * direction);
            auto inner_cells_and_ghosts        = intersection(translated_outer_nghbr, subset).on(level);
            auto inner_ghosts_with_outer_nghbr = difference(inner_cells_and_ghosts, bdry_cells).on(level);

            __apply_bc_on_subset(bc, field, inner_ghosts_with_outer_nghbr, stencil, direction);
        }
    }

    template <class Field, std::size_t stencil_size>
    void apply_extrapolation_bc_impl(Bc<Field>& bc, std::size_t level, Field& field, bool diagonals_only, bool only_fill_ghost_neighbours)
    {
        static constexpr std::size_t dim = Field::dim;

        using direction_t = DirectionVector<dim>;

        auto& domain      = field.mesh().domain();
        auto one_interval = 1 << (domain.level() - level);

        static_nested_loop<dim, -1, 2>(
            [&](auto& dir)
            {
                int number_of_one = xt::sum(xt::abs(dir))[0];

                if (number_of_one > 0)
                {
                    bool is_periodic = false;
                    for (std::size_t i = 0; i < dim; ++i)
                    {
                        if (dir(i) != 0 && field.mesh().is_periodic(i))
                        {
                            is_periodic = true;
                            break;
                        }
                    }
                    if (!is_periodic)
                    {
                        bool is_cartesian_direction = is_cartesian(dir);

                        if (!diagonals_only || !is_cartesian_direction)
                        {
                            if (is_cartesian_direction)
                            {
                                auto subset = difference(domain, translate(domain, -one_interval * dir));
                                __apply_extrapolation_bc_on_subset<stencil_size>(bc, level, field, dir, subset, only_fill_ghost_neighbours);
                            }
                            else
                            {
                                if constexpr (dim == 2)
                                {
                                    auto subset = difference(domain,
                                                             union_(translate(domain, one_interval * direction_t{-dir[0], 0}),
                                                                    translate(domain, one_interval * direction_t{0, -dir[1]})));
                                    __apply_extrapolation_bc_on_subset<stencil_size>(bc, level, field, dir, subset, only_fill_ghost_neighbours);
                                }
                                else if constexpr (dim == 3)
                                {
                                    auto subset = difference(domain,
                                                             union_(translate(domain, one_interval * direction_t{-dir[0], 0, 0}),
                                                                    translate(domain, one_interval * direction_t{0, -dir[1], 0}),
                                                                    translate(domain, one_interval * direction_t{0, 0, -dir[2]})));
                                    __apply_extrapolation_bc_on_subset<stencil_size>(bc, level, field, dir, subset, only_fill_ghost_neighbours);
                                }
                            }
                        }
                    }
                }
            });
    }

    template <std::size_t order, class Field>
    struct DirichletImpl : public Bc<Field>
    {
        INIT_BC(DirichletImpl, 2 * order) // stencil_size = 2*order

        stencil_t get_stencil(constant_stencil_size_t) const override
        {
            return line_stencil<dim, 0, 2 * order>();
        }

        apply_function_t get_apply_function(constant_stencil_size_t, const direction_t&) const override
        {
            return [](Field& u, const stencil_cells_t& cells, const value_t& dirichlet_value)
            {
                if constexpr (order == 1)
                {
                    //      [0]   [1]
                    //    |_____|.....|
                    //     cell  ghost

                    u[cells[1]] = 2 * dirichlet_value - u[cells[0]];
                }
                else if constexpr (order == 2)
                {
                    //     [0]   [1]   [2]   [3]
                    //   |_____|_____|.....|.....|
                    //       cells      ghosts

                    // We define a polynomial of degree 2 that passes by 3 points (the 2 cells and the boundary value):
                    //                       p(x) = a*x^2 + b*x + c.
                    // The coefficients a, b, c are found by inverting the Vandermonde matrix obtained by inserting the 3 points into
                    // the polynomial. If we set the abscissa 0 at the center of cells[0], this system reads
                    //                       p( 0 ) = u[cells[0]]
                    //                       p( 1 ) = u[cells[1]]
                    //                       p(3/2) = dirichlet_value.
                    // Then, we want that the ghost values be also located on this polynomial, i.e.
                    //                       u[cells[2]] = p( 2 )
                    //                       u[cells[3]] = p( 3 ).

                    u[cells[2]] = 8. / 3. * dirichlet_value + 1. / 3. * u[cells[0]] - 2. * u[cells[1]];
                    u[cells[3]] = 8. * dirichlet_value + 2. * u[cells[0]] - 9. * u[cells[1]];
                }
                else if constexpr (order == 3)
                {
                    //     [0]   [1]   [2]   [3]   [4]   [5]
                    //   |_____|_____|_____|.....|.....|.....|
                    //          cells             ghosts

                    // We define a polynomial of degree 3 that passes by 4 points (the 3 cells and the boundary value):
                    //                       p(x) = a*x^3 + b*x^2 + c*x + d.
                    // The coefficients a, b, c, d are found by inverting the Vandermonde matrix obtained by inserting the 4 points into
                    // the polynomial. If we set the abscissa 0 at the center of cells[0], this system reads
                    //                       p( 0 ) = u[cells[0]]
                    //                       p( 1 ) = u[cells[1]]
                    //                       p( 2 ) = u[cells[2]]
                    //                       p(5/2) = dirichlet_value.
                    // Then, we want that the ghost values be also located on this polynomial, i.e.
                    //                       u[cells[3]] = p( 3 )
                    //                       u[cells[4]] = p( 4 )
                    //                       u[cells[5]] = p( 5 ).

                    u[cells[3]] = 16. / 5. * dirichlet_value - 1. / 5. * u[cells[0]] + u[cells[1]] - 3. * u[cells[2]];
                    u[cells[4]] = 64. / 5. * dirichlet_value - 9. / 5. * u[cells[0]] + 8. * u[cells[1]] - 18. * u[cells[2]];
                    u[cells[5]] = 32. * dirichlet_value - 6. * u[cells[0]] + 25. * u[cells[1]] - 50. * u[cells[2]];
                }
                else if constexpr (order == 4)
                {
                    u[cells[4]] = 128. / 35 * dirichlet_value + 1. / 7 * u[cells[0]] - 4. / 5 * u[cells[1]] + 2 * u[cells[2]]
                                - 4. * u[cells[3]];
                    u[cells[5]] = 128. / 7. * dirichlet_value + 12. / 7. * u[cells[0]] - 9 * u[cells[1]] + 20 * u[cells[2]]
                                - 30 * u[cells[3]];
                    u[cells[6]] = 384. / 7. * dirichlet_value + 50. / 7. * u[cells[0]] - 36 * u[cells[1]] + 75 * u[cells[2]]
                                - 100 * u[cells[3]];
                    u[cells[7]] = 128 * dirichlet_value + 20 * u[cells[0]] - 98 * u[cells[1]] + 196 * u[cells[2]] - 245 * u[cells[3]];
                }
                else
                {
                    static_assert(order <= 4, "The Dirichlet boundary conditions are only implemented up to order 4.");
                }
            };
        }
    };

    template <std::size_t order = 1>
    struct Dirichlet
    {
        template <class Field>
        using impl_t = DirichletImpl<order, Field>;
    };

    template <std::size_t order, class Field>
    struct NeumannImpl : public Bc<Field>
    {
        INIT_BC(NeumannImpl, 2 * order) // stencil_size = 2*order

        stencil_t get_stencil(constant_stencil_size_t) const override
        {
            return line_stencil<dim, 0, 2 * order>();
        }

        apply_function_t get_apply_function(constant_stencil_size_t, const direction_t&) const override
        {
            return [](Field& f, const stencil_cells_t& cells, const value_t& value)
            {
                if constexpr (order == 1)
                {
                    static constexpr std::size_t in  = 0;
                    static constexpr std::size_t out = 1;

                    double dx     = f.mesh().cell_length(cells[out].level);
                    f[cells[out]] = dx * value + f[cells[in]];
                }
                else
                {
                    static_assert(order <= 1, "The Neumann boundary conditions are only implemented at the first order.");
                }
            };
        }
    };

    template <std::size_t order = 1>
    struct Neumann
    {
        template <class Field>
        using impl_t = NeumannImpl<order, Field>;
    };

    template <class Field, std::size_t stencil_size_>
    struct PolynomialExtrapolation : public Bc<Field>
    {
        INIT_BC(PolynomialExtrapolation, stencil_size_)

        static constexpr std::size_t max_stencil_size_implemented_PE = 6;

        static_assert(stencil_size_ % 2 == 0, "stencil_size must be even.");
        static_assert(stencil_size_ >= 2 && stencil_size_ <= max_stencil_size_implemented_PE);

        apply_function_t get_apply_function(constant_stencil_size_t, const direction_t&) const override
        {
            return [](Field& u, const stencil_cells_t& cells, const value_t&)
            {
                /*
                                u[0]  u[1]  u[2]   ?
                              |_____|_____│_____|_____|
                    cell index   0     1     2     3     (the ghost to fill is always at the last index in 'cell')
                           x =  -3    -2    -1     0     (we arbitrarily set the coordinate x for the extrapolation
                                                          such that the ghost is at x=0)

                    We search the coefficients c[i] of the polynomial P
                          P(x) = c[0]x^2 + c[1]x + c[2]
                    that passes by all the known u[i]. (Note that deg(P) = stencil_size_ - 2)

                    We inverse the Vandermonde system
                        │ (-3)^2  -3  1 │ │c[0]│   │u[0]│
                        │ (-2)^2  -2  1 │ │c[1]│ = │u[1]│.
                        │ (-1)^2  -1  1 │ │c[2]│   │u[2]│
                    This step is done using a symbolic calculus tool.

                    To get the value at x=0, we actually just need c[2]:
                          P(x=0) = c[2].
                */

                const auto& ghost = cells[stencil_size_ - 1];

#ifdef SAMURAI_CHECK_NAN
                for (std::size_t field_i = 0; field_i < Field::size; field_i++)
                {
                    for (std::size_t c = 0; c < stencil_size_ - 1; ++c)
                    {
                        if (std::isnan(field_value(u, cells[c], field_i)))
                        {
                            std::cerr << "NaN detected when applying polynomial extrapolation on the outer ghosts: " << cells[c] << std::endl;
                            assert(false);
                        }
                    }
                }
#endif

                // Last coefficient of the polynomial
                if constexpr (stencil_size_ == 2)
                {
                    u[ghost] = u[cells[0]];
                }
                else if constexpr (stencil_size_ == 4)
                {
                    u[ghost] = u[cells[0]] - u[cells[1]] * 3.0 + u[cells[2]] * 3.0;
                }
                else if constexpr (stencil_size_ == 6)
                {
                    u[ghost] = u[cells[0]] - u[cells[1]] * 5.0 + u[cells[2]] * 1.0E+1 - u[cells[3]] * 1.0E+1 + u[cells[4]] * 5.0;
                }
            };
        }
    };

    template <class Field>
    void update_bc(std::size_t level, Field& field)
    {
        static constexpr std::size_t ghost_width                     = Field::mesh_t::config::ghost_width;
        static constexpr std::size_t prediction_order                = Field::mesh_t::config::prediction_order;
        static constexpr std::size_t max_stencil_size_implemented_BC = Bc<Field>::max_stencil_size_implemented;
        static constexpr std::size_t max_stencil_size_implemented_PE = PolynomialExtrapolation<Field, 2>::max_stencil_size_implemented_PE;

        auto& mesh = field.mesh();

        // Step 0:
        // One level below the boundary cells, there are outer ghosts used for prediction (for the computation of details).
        // Those ghosts are supposed (for now) to be filled by the boundary conditions.
        // However, if the B.C. stencil is larger than the prediction stencil, there are not enough ghosts to apply the B.C.
        // So, in order to make sure that the outer ghosts linked to the detail computation are filled with values,
        // we start by filling them with polynomial extrapolation.
        // If the B.C. can, in fact, be applied, they will overwrite the polynomial extrapolation.
        if constexpr (Field::mesh_t::config::max_stencil_width > prediction_order)
        {
            if (mesh.min_level() != mesh.max_level())
            {
                // We populate the ghosts sequentially from the closest to the farthest.
                for (std::size_t ghost_layer = 1; ghost_layer <= prediction_order; ++ghost_layer)
                {
                    std::size_t stencil_s = 2 * ghost_layer;
                    static_for<2, std::min(max_stencil_size_implemented_PE, 2 * prediction_order) + 1>::apply(
                        [&](auto integral_constant_i)
                        {
                            static constexpr std::size_t i = decltype(integral_constant_i)::value;

                            if constexpr (i % 2 == 0) // (because PolynomialExtrapolation is only implemented for even stencil_size)
                            {
                                if (stencil_s == i)
                                {
                                    auto& domain = detail::get_mesh(field.mesh());
                                    PolynomialExtrapolation<Field, i> bc(domain, ConstantBc<Field>(), true);

                                    bool only_fill_corners          = false;
                                    bool only_fill_ghost_neighbours = true; // because cell neighbours will be filled by the B.C.
                                    apply_extrapolation_bc_impl<Field, i>(bc, level, field, only_fill_corners, only_fill_ghost_neighbours);
                                }
                            }
                        });
                }
            }
        }

        // Step 1:
        // Apply the B.C. attached to the field by the user
        std::size_t real_max_stencil_size = 0;

        for (auto& bc : field.get_bc())
        {
            static_for<1, max_stencil_size_implemented_BC + 1>::apply( // for (int i=1; i<=max_stencil_size_implemented; i++)
                [&](auto integral_constant_i)
                {
                    static constexpr std::size_t i = decltype(integral_constant_i)::value;

                    if (bc->stencil_size() == i)
                    {
                        apply_bc_impl<Field, i>(*bc.get(), level, field);
                    }
                });

            real_max_stencil_size = std::max(real_max_stencil_size, bc->stencil_size());
        }

        // Step 2:
        // Polynomial extrapolation to populate corners and ghosts layers that are not filled by the B.C. (i.e. outside of the B.C.'s
        // stencil)

        // if (mesh.min_level() != mesh.max_level()) // We comment this test because some schemes have a box stencil and need values in
        //                                              external corners. However, the associated B.C. doesn't fill them, so we do it.
        //                                              Since on a uniform mesh, the MR process doesn't need corners, it would be better
        //                                              that the corners be filled by the user as part of the BCs.
        //                                              But so far the users don't have an easy way to do that...
        {
            // We populate the ghosts sequentially from the closest to the farthest.
            for (std::size_t ghost_layer = 1; ghost_layer <= ghost_width; ++ghost_layer)
            {
                std::size_t stencil_s = 2 * ghost_layer;
                static_for<2, std::min(max_stencil_size_implemented_PE, 2 * ghost_width) + 1>::apply(
                    [&](auto integral_constant_i)
                    {
                        static constexpr std::size_t i = decltype(integral_constant_i)::value;

                        if constexpr (i % 2 == 0) // (because PolynomialExtrapolation is only implemented for even stencil_size)
                        {
                            if (stencil_s == i)
                            {
                                auto& domain = detail::get_mesh(field.mesh());
                                PolynomialExtrapolation<Field, i> bc(domain, ConstantBc<Field>(), true);

                                // If the ghost layer is managed by the B.C., we only populate the corners.
                                // Otherwise, we populate the Cartesian directions as well, by polynomial extrapolation.
                                bool only_fill_corners          = ghost_layer <= real_max_stencil_size / 2;
                                bool only_fill_ghost_neighbours = false;
                                apply_extrapolation_bc_impl<Field, i>(bc, level, field, only_fill_corners, only_fill_ghost_neighbours);
                            }
                        }
                    });
            }
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
