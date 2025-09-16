// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "../cell.hpp"
#include "../samurai_config.hpp"
#include "../static_algorithm.hpp"
#include "../stencil.hpp"
#include "../storage/containers.hpp"
#include "../utils.hpp"

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

#define INIT_BC(NAME, STENCIL_SIZE)                                                                                       \
    using base_t      = samurai::Bc<Field>;                                                                               \
    using cell_t      = typename base_t::cell_t;                                                                          \
    using value_t     = typename base_t::value_t;                                                                         \
    using direction_t = typename base_t::direction_t;                                                                     \
    using base_t::base_t;                                                                                                 \
    using base_t::dim;                                                                                                    \
    using base_t::get_apply_function;                                                                                     \
    using base_t::get_stencil;                                                                                            \
                                                                                                                          \
    using stencil_t               = samurai::Stencil<STENCIL_SIZE, dim>;                                                  \
    using constant_stencil_size_t = std::integral_constant<std::size_t, STENCIL_SIZE>;                                    \
    using stencil_cells_t         = std::array<cell_t, STENCIL_SIZE>;                                                     \
    using apply_function_t        = std::function<void(Field&, const std::array<cell_t, STENCIL_SIZE>&, const value_t&)>; \
                                                                                                                          \
    static_assert(STENCIL_SIZE <= base_t::max_stencil_size_implemented, "The stencil size is too large.");                \
                                                                                                                          \
    std::unique_ptr<base_t> clone() const override                                                                        \
    {                                                                                                                     \
        return std::make_unique<NAME>(*this);                                                                             \
    }                                                                                                                     \
                                                                                                                          \
    int stencil_size() const override                                                                                     \
    {                                                                                                                     \
        return STENCIL_SIZE;                                                                                              \
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
    class VectorField;

    template <class mesh_t, class value_t>
    class ScalarField;

    ////////////////////////
    // BcValue definition //
    ////////////////////////
    template <class Field>
    struct BcValue
    {
        static constexpr std::size_t dim = Field::dim;
        using value_t     = CollapsArray<typename Field::value_type, Field::n_comp, detail::is_soa_v<Field>, Field::is_scalar>;
        using coords_t    = xt::xtensor_fixed<double, xt::xshape<dim>>;
        using direction_t = DirectionVector<dim>;
        using cell_t      = typename Field::cell_t;

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

        explicit FunctionBc(const function_t& f);

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

        explicit OnDirection(const std::array<direction_t, nd>& d);

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

        explicit CoordsRegion(const function_t& f);

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

        explicit SetRegion(const Set& set);

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

        static constexpr std::size_t dim = Field::dim;
        using mesh_t                     = typename Field::mesh_t;
        using interval_t                 = typename Field::interval_t;

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
        virtual int stencil_size() const          = 0;

        static constexpr int max_stencil_size_implemented = 10; // cppcheck-suppress unusedStructMember
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
        // xt::xtensor<typename Field::value_type, detail::return_type<typename Field::value_type, n_comp>::dim> m_value;
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
        static_assert(Field::n_comp == sizeof...(T) + 1,
                      "The number of constant values should be equal to the "
                      "number of components in the field");

        auto& mesh = detail::get_mesh(field.mesh());
        return field.attach_bc(bc_type<Field>(mesh, ConstantBc<Field>(v1, v...)));
    }

    template <class bc_type, class Field, class... T>
    auto make_bc(Field& field, typename Field::value_type v1, T... v)
    {
        static_assert(std::is_same_v<typename Field::value_type, std::common_type_t<typename Field::value_type, T...>>,
                      "The constant value type must be the same as the field value_type");
        static_assert(Field::n_comp == sizeof...(T) + 1,
                      "The number of constant values should be equal to the "
                      "number of components in the field");

        using bc_impl = typename bc_type::template impl_t<Field>;

        auto& mesh = detail::get_mesh(field.mesh());
        return field.attach_bc(bc_impl(mesh, ConstantBc<Field>(v1, v...)));
    }

} // namespace samurai
