// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_base.hpp"
#include "traversers/contraction_traverser.hpp"

namespace samurai
{

    template <class Set>
    class Contraction;

    template <class Set>
    struct SetTraits<Contraction<Set>>
    {
        static_assert(IsSet<Set>::value);

        template <std::size_t d>
        using traverser_t = ContractionTraverser<typename Set::template traverser_t<d>>;

        struct Workspace
        {
            typename Set::Workspace child_workspace;
        };

        static constexpr std::size_t dim()
        {
            return Set::dim;
        }
    };

    template <class Set>
    class Contraction : public SetBase<Contraction<Set>>
    {
        using Self = Contraction<Set>;

      public:

        SAMURAI_SET_TYPEDEFS

        using contraction_t    = std::array<value_t, Base::dim>;
        using do_contraction_t = std::array<bool, Base::dim>;

        Contraction(const Set& set, const contraction_t& contraction)
            : m_set(set)
            , m_contraction(contraction)
        {
            assert(std::all_of(m_contraction.cbegin(),
                               m_contraction.cend(),
                               [](const contraction_t& c)
                               {
                                   return c >= 0;
                               }));
        }

        Contraction(const Set& set, const value_t contraction)
            : m_set(set)
        {
            assert(contraction >= 0);
            std::fill(m_contraction.begin(), m_contraction.end(), contraction);
        }

        Contraction(const Set& set, const value_t contraction, const do_contraction_t& do_contraction)
            : m_set(set)
        {
            for (std::size_t i = 0; i != m_contraction.size(); ++i)
            {
                m_contraction[i] = contraction * do_contraction[i];
            }
        }

        SAMURAI_INLINE std::size_t level_impl() const
        {
            return m_set.level();
        }

        SAMURAI_INLINE bool exist_impl() const
        {
            return m_set.exist();
        }

        SAMURAI_INLINE bool empty_impl() const
        {
            return Base::empty_default_impl();
        }

        template <std::size_t d>
        SAMURAI_INLINE void
        init_workspace_impl(const std::size_t n_traversers, std::integral_constant<std::size_t, d> d_ic, Workspace& workspace) const
        {
            m_set.init_workspace(n_traversers, d_ic, workspace.child_workspace);
        }

        template <std::size_t d>
        SAMURAI_INLINE traverser_t<d>
        get_traverser_impl(const yz_index_t& index, std::integral_constant<std::size_t, d> d_ic, Workspace& workspace) const
        {
            return traverser_t<d>(m_set.get_traverser(index, d_ic, workspace.child_workspace), m_contraction[d]);
        }

        template <std::size_t d>
        SAMURAI_INLINE traverser_t<d>
        get_traverser_unordered_impl(const yz_index_t& index, std::integral_constant<std::size_t, d> d_ic, Workspace& workspace) const
        {
            return traverser_t<d>(m_set.get_traverser_unordered(index, d_ic, workspace.child_workspace), m_contraction[d]);
        }

      private:

        Set m_set;
        contraction_t m_contraction;
    };

    //----------------------------------------------------------------//
    //                        Contract                                //
    //----------------------------------------------------------------//

    template <typename T>
    concept IsLCA = std::same_as<LevelCellArray<T::dim, typename T::interval_t>, T>;

    template <std::size_t direction_index, class SubsetOrLCA>
    auto contract_rec(const SubsetOrLCA& set, int width, const std::array<bool, SubsetOrLCA::dim>& contract_directions)
    {
        static constexpr std::size_t dim = SubsetOrLCA::dim;
        if constexpr (direction_index < dim)
        {
            using direction_t = xt::xtensor_fixed<int, xt::xshape<dim>>;

            auto contracted_in_other_dirs = contract_rec<direction_index + 1>(set, width, contract_directions);
            direction_t dir;
            dir.fill(0);
            dir[direction_index] = contract_directions[direction_index] ? width : 0;

            return intersection(contracted_in_other_dirs, translate(set, dir), translate(set, -dir));
        }
        else
        {
            if constexpr (IsLCA<SubsetOrLCA>)
            {
                return self(set);
            }
            else
            {
                return set;
            }
        }
    }

    /**
     * @brief Contract a set in the specified directions.
     *
     * @tparam SubsetOrLCA The type of the set to contract.
     * @param set The set or LevelCellArray to contract.
     * @param width The contraction width.
     * @param contract_directions An array indicating which directions to contract (true for contraction, false for no contraction).
     * @return A new set that is contracted in the specified directions.
     */
    template <class SubsetOrLCA>
    auto contract(const SubsetOrLCA& set, std::size_t width, const std::array<bool, SubsetOrLCA::dim>& contract_directions)
    {
        return contract_rec<0>(set, static_cast<int>(width), contract_directions);
    }

    /**
     * @brief Contract a set in all directions.
     *
     * This function is a convenience wrapper that contracts the set in all dimensions.
     *
     * @tparam SubsetOrLCA The type of the set to contract.
     * @param set The set or LevelCellArray to contract.
     * @param width The contraction width.
     * @return A new set that is contracted in all directions.
     */
    template <class SubsetOrLCA>
    auto contract(const SubsetOrLCA& set, std::size_t width)
    {
        std::array<bool, SubsetOrLCA::dim> contract_directions;
        std::fill(contract_directions.begin(), contract_directions.end(), true);
        return contract(set, width, contract_directions);
    }

} // namespace samurai
