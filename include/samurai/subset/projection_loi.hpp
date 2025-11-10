// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../samurai_config.hpp"
#include "../static_algorithm.hpp"
#include "set_base.hpp"
#include "traversers/last_dim_projection_loi_traverser.hpp"
#include "traversers/projection_loi_traverser.hpp"
#include "utils.hpp"

#include <fmt/ranges.h>

namespace samurai
{

    namespace detail
    {
        template <class Set, typename Seq>
        struct ProjectionLOIWork;

        template <class Set, std::size_t... ds>
        struct ProjectionLOIWork<Set, std::index_sequence<ds...>>
        {
            template <std::size_t d>
            using child_traverser_t = typename Set::template traverser_t<d>;

            template <std::size_t d>
            using work_t = ListOfIntervals<typename child_traverser_t<d>::value_t>;

            using Type = std::tuple<work_t<ds>...>;
        };
    } // namespace detail

    template <class Set>
    class ProjectionLOI;

    template <class Set>
    struct SetTraits<ProjectionLOI<Set>>
    {
        static_assert(IsSet<Set>::value);

        template <std::size_t d>
        using child_traverser_t = typename Set::template traverser_t<d>;

        template <std::size_t d>
        using traverser_t = std::conditional_t<d == Set::dim - 1,
                                               LastDimProjectionLOITraverser<child_traverser_t<d>>,
                                               ProjectionLOITraverser<child_traverser_t<d>>>;

        struct Workspace
        {
            typename detail::ProjectionLOIWork<Set, std::make_index_sequence<Set::dim>>::Type projection_workspace;
            typename Set::Workspace child_workspace;
            typename Set::Workspace tmp_child_workspace;
        };

        static constexpr std::size_t dim()
        {
            return Set::dim;
        }
    };

    template <class Set>
    class ProjectionLOI : public SetBase<ProjectionLOI<Set>>
    {
        using Self            = ProjectionLOI<Set>;
        using ListOfIntervals = typename detail::ProjectionLOIWork<Set, std::make_index_sequence<Set::dim>>::Type;

      public:

        SAMURAI_SET_TYPEDEFS

        ProjectionLOI(const Set& set, const std::size_t level)
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

        inline std::size_t level_impl() const
        {
            return m_level;
        }

        inline bool exist_impl() const
        {
            return m_set.exist();
        }

        inline bool empty_impl() const
        {
            return m_set.empty();
        }

        template <std::size_t d>
        inline void
        init_workspace_impl(const std::size_t n_traversers, std::integral_constant<std::size_t, d> d_ic, Workspace& workspace) const
        {
            assert(n_traversers == 1);

            m_set.init_workspace(n_traversers, d_ic, workspace.child_workspace);
        }

        template <std::size_t d>
        inline traverser_t<d>
        get_traverser_impl(const yz_index_t& index, std::integral_constant<std::size_t, d> d_ic, Workspace& workspace) const
        {
            auto& listOfIntervals = std::get<d>(workspace.projection_workspace);
            listOfIntervals.clear();

            if (m_projectionType == ProjectionType::COARSEN)
            {
                if constexpr (d != Base::dim - 1)
                {
                    yz_index_t index_min(utils::pow2(index, m_shift));
                    yz_index_t index_max(utils::sumAndPow2(index, 1, m_shift));

                    yz_index_t index_rec;
                    fill_list_of_interval_rec(index_min,
                                              index_max,
                                              index_rec,
                                              d_ic,
                                              std::integral_constant<std::size_t, Base::dim - 1>{},
                                              workspace);

                    return traverser_t<d>(m_set.get_traverser(utils::pow2(index, m_shift), d_ic, workspace.child_workspace),
                                          listOfIntervals.cbegin(),
                                          listOfIntervals.cend());
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
        inline traverser_t<d>
        get_traverser_unordered_impl(const yz_index_t& index, std::integral_constant<std::size_t, d> d_ic, Workspace& workspace) const
        {
            return get_traverser_impl(index, d_ic, workspace);
        }

      private:

        template <std::size_t d, std::size_t dCur>
        inline void fill_list_of_interval_rec(const yz_index_t& index_min,
                                              const yz_index_t& index_max,
                                              yz_index_t& index,
                                              std::integral_constant<std::size_t, d> d_ic,
                                              std::integral_constant<std::size_t, dCur> dCur_ic,
                                              Workspace& workspace) const
        {
            using child_traverser_t        = typename Set::template traverser_t<dCur>;
            using child_current_interval_t = typename child_traverser_t::current_interval_t;
            using ChildWorkspace           = typename Set::Workspace;

            ChildWorkspace& child_workspace = workspace.tmp_child_workspace;

            m_set.init_workspace(1, dCur_ic, child_workspace);

            for (child_traverser_t traverser = m_set.get_traverser(index, dCur_ic, child_workspace); !traverser.is_empty();
                 traverser.next_interval())
            {
                child_current_interval_t interval = traverser.current_interval();

                if constexpr (dCur == d)
                {
                    std::get<d>(workspace.projection_workspace).add_interval(interval >> m_shift);
                }
                else
                {
                    const auto index_start = std::max(interval.start, index_min[dCur - 1]);
                    const auto index_bound = std::min(interval.end, index_max[dCur - 1]);

                    for (index[dCur - 1] = index_start; index[dCur - 1] < index_bound; ++index[dCur - 1])
                    {
                        fill_list_of_interval_rec(index_min, index_max, index, d_ic, std::integral_constant<std::size_t, dCur - 1>{}, workspace);
                    }
                }
            }
        }

        Set m_set;
        std::size_t m_level;
        ProjectionType m_projectionType;
        std::size_t m_shift;
    };

} // namespace samurai
