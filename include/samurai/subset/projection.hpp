// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../samurai_config.hpp"
#include "../static_algorithm.hpp"
#include "set_base.hpp"
#include "traversers/last_dim_projection_traverser.hpp"
#include "traversers/projection_traverser.hpp"
#include "utils.hpp"

#include <fmt/ranges.h>

namespace samurai
{

    namespace detail
    {
        template <class Set, typename Seq>
        struct ProjectionWork;

        template <class Set, std::size_t... ds>
        struct ProjectionWork<Set, std::index_sequence<ds...>>
        {
            template <std::size_t d>
            using child_traverser_t = typename Set::template traverser_t<d>;

            template <std::size_t d>
            using work_t = ListOfIntervals<typename child_traverser_t<d>::value_t>;

            using Type = std::tuple<work_t<ds>...>;
        };
    } // namespace detail

    template <class Set>
    class Expansion;

    template <class Set>
    class Projection;

    template <class Set>
    struct SetTraits<Projection<Set>>
    {
        static_assert(IsSet<Set>::value);

        template <std::size_t d>
        using child_traverser_t = typename Set::template traverser_t<d>;

        template <std::size_t d>
        using traverser_t = std::
            conditional_t<d == Set::dim - 1, LastDimProjectionTraverser<child_traverser_t<d>>, ProjectionTraverser<child_traverser_t<d>>>;

        struct Workspace
        {
            typename detail::ProjectionWork<Set, std::make_index_sequence<Set::dim>>::Type projection_workspace;
            typename Set::Workspace child_workspace;
            typename Set::Workspace tmp_child_workspace;
        };

        static constexpr std::size_t dim()
        {
            return Set::dim;
        }
    };

    template <class Set>
    class Projection : public SetBase<Projection<Set>>
    {
        using Self            = Projection<Set>;
        using ListOfIntervals = typename detail::ProjectionWork<Set, std::make_index_sequence<Set::dim>>::Type;

      public:

        friend class Expansion<Self>;

        SAMURAI_SET_TYPEDEFS

        Projection(const Set& set, const std::size_t level)
            : m_set(set)
            , m_level(level)
        {
            if (m_level < m_set.level())
            {
                m_projectionType = ProjectionType::COARSEN;
                m_shift          = m_set.level() - m_level;
            }
            else
            {
                m_projectionType = ProjectionType::REFINE;
                m_shift          = m_level - m_set.level();
            }
        }

        SAMURAI_INLINE std::size_t level_impl() const
        {
            return m_level;
        }

        SAMURAI_INLINE bool exist_impl() const
        {
            return m_set.exist();
        }

        SAMURAI_INLINE bool empty_impl() const
        {
            return m_set.empty();
        }

        template <std::size_t d>
        SAMURAI_INLINE void
        init_workspace_impl(const std::size_t n_traversers, std::integral_constant<std::size_t, d> d_ic, Workspace& workspace) const
        {
            assert(n_traversers == 1);

            m_set.init_workspace(n_traversers, d_ic, workspace.child_workspace);
        }

        template <std::size_t d>
        SAMURAI_INLINE traverser_t<d>
        get_traverser_impl(const yz_index_t& index, std::integral_constant<std::size_t, d> d_ic, Workspace& workspace) const
        {
            if (m_projectionType == ProjectionType::COARSEN)
            {
                if constexpr (d != Base::dim - 1)
                {
                    const auto projection_func = [shift = m_shift](const auto /* d_cur */, const interval_t& interval) -> interval_t
                    {
                        return interval >> shift;
                    };
                    const auto index_range_func = [&index, shift = m_shift](const auto d_cur) -> interval_t
                    {
                        return interval_t(index[d_cur - 1] << shift, ((index[d_cur - 1] + 1) << shift));
                    };

                    auto& list_of_intervals = std::get<d>(workspace.projection_workspace);

                    subset_utils::transform_to_loi(m_set,
                                                   index_range_func,
                                                   d_ic,
                                                   projection_func,
                                                   workspace.tmp_child_workspace,
                                                   list_of_intervals);

                    return traverser_t<d>(m_set.get_traverser(utils::pow2(index, m_shift), d_ic, workspace.child_workspace),
                                          list_of_intervals.cbegin(),
                                          list_of_intervals.cend());
                }
                else
                {
                    return traverser_t<d>(m_set.get_traverser(utils::pow2(index, m_shift), d_ic, workspace.child_workspace),
                                          m_projectionType,
                                          m_shift);
                }
            }
            else
            {
                return traverser_t<d>(m_set.get_traverser(utils::powMinus2(index, m_shift), d_ic, workspace.child_workspace),
                                      m_projectionType,
                                      m_shift);
            }
        }

        template <std::size_t d>
        SAMURAI_INLINE traverser_t<d>
        get_traverser_unordered_impl(const yz_index_t& index, std::integral_constant<std::size_t, d> d_ic, Workspace& workspace) const
        {
            return get_traverser_impl(index, d_ic, workspace);
        }

      private:

        Set m_set;
        std::size_t m_level;
        ProjectionType m_projectionType;
        std::size_t m_shift;
    };

} // namespace samurai
