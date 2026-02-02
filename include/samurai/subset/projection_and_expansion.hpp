// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "traversers/last_dim_projection_and_expansion_traverser.hpp"

namespace samurai
{

    template <class Set>
    class ProjectionLOI;

    template <class Set>
    class Expansion;

    template <class Set>
    class Expansion<ProjectionLOI<Set>>;

    template <class Set>
    struct SetTraits<Expansion<ProjectionLOI<Set>>>
    {
        static_assert(IsSet<Set>::value);

        template <std::size_t d>
        using child_traverser_t = typename Set::template traverser_t<d>;

        template <std::size_t d>
        using traverser_t = std::conditional_t<d == Set::dim - 1,
                                               LastDimProjectionAndExpansionTraverser<child_traverser_t<d>>,
                                               LOITraverser<typename Set::template traverser_t<d>::value_t>>;

        struct Workspace
        {
            // we are going to use the same workspace as the projection.
            // the only difference is that we are going to apply the expansion right away.
            typename detail::ProjectionLOIWork<Set, std::make_index_sequence<Set::dim>>::Type projection_and_expand_workspace;
            typename Set::Workspace child_workspace;
            typename Set::Workspace tmp_child_workspace;
        };

        static constexpr std::size_t dim()
        {
            return Set::dim;
        }
    };

    template <class Set>
    class Expansion<ProjectionLOI<Set>> : public SetBase<Expansion<ProjectionLOI<Set>>>
    {
        using Self            = Expansion<ProjectionLOI<Set>>;
        using ListOfIntervals = typename detail::ProjectionLOIWork<Set, std::make_index_sequence<Set::dim>>::Type;

      public:

        SAMURAI_SET_TYPEDEFS

        using expansion_t    = std::array<value_t, Base::dim>;
        using do_expansion_t = std::array<bool, Base::dim>;

        explicit Expansion(const ProjectionLOI<Set>& projected_set, const expansion_t& expansions)
            : m_set(projected_set.m_set)
            , m_level(projected_set.m_level)
            , m_projectionType(projected_set.m_projectionType)
            , m_shift(projected_set.m_shift)
            , m_expansions(expansions)
        {
        }

        Expansion(const ProjectionLOI<Set>& projected_set, const value_t expansion)
            : m_set(projected_set.m_set)
            , m_level(projected_set.m_level)
            , m_projectionType(projected_set.m_projectionType)
            , m_shift(projected_set.m_shift)
        {
            m_expansions.fill(expansion);
        }

        Expansion(const ProjectionLOI<Set>& projected_set, const value_t expansion, const do_expansion_t& do_expansion)
            : m_set(projected_set.m_set)
            , m_level(projected_set.m_level)
            , m_projectionType(projected_set.m_projectionType)
            , m_shift(projected_set.m_shift)
        {
            for (std::size_t i = 0; i != m_expansions.size(); ++i)
            {
                m_expansions[i] = expansion * do_expansion[i];
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
            if constexpr (d == Base::dim - 1)
            {
                // in theory, we should apply the inverse transformation to index.
                // However, since this is the last dim, invOp(index) will not be used and therefore it does not need to be computed.
                return traverser_t<d>(m_set.get_traverser(index, d_ic, workspace.child_workspace), m_projectionType, m_shift, m_expansions[d]);
            }
            else if (m_projectionType == ProjectionType::COARSEN)
            {
                const auto projection_func = [shift = m_shift, &expansions = m_expansions](const auto /* d_cur */,
                                                                                           const interval_t& interval) -> interval_t
                {
                    return interval_t(traverser_utils::coarsen_start(interval.start, shift) - expansions[d],
                                      traverser_utils::coarsen_end(interval.end, shift) + expansions[d]);
                };
                const auto index_range_func = [&index, shift = m_shift, &expansions = m_expansions](const auto d_cur) -> interval_t
                {
                    return interval_t((index[d_cur - 1] - expansions[d_cur]) << shift, (index[d_cur - 1] + expansions[d_cur] + 1) << shift);
                };

                auto& list_of_intervals = std::get<d>(workspace.projection_and_expand_workspace);

                subset_utils::transform_to_loi(m_set, index_range_func, d_ic, projection_func, workspace.tmp_child_workspace, list_of_intervals);

                return traverser_t<d>(list_of_intervals.cbegin(), list_of_intervals.cend());
            }
            else
            {
                const auto projection_func = [shift = m_shift, &expansions = m_expansions](const auto /* d_cur */,
                                                                                           const interval_t& interval) -> interval_t
                {
                    return interval_t(traverser_utils::refine_start(interval.start, shift) - expansions[d],
                                      traverser_utils::refine_end(interval.end, shift) + expansions[d]);
                };
                const auto index_range_func = [&index, shift = m_shift, scale = 1. / std::pow(2., m_shift), &expansions = m_expansions](
                                                  const auto d_cur) -> interval_t
                {
                    return interval_t((index[d_cur - 1] - expansions[d_cur]) >> shift, ((index[d_cur - 1] + expansions[d_cur]) >> shift) + 1);
                };

                auto& list_of_intervals = std::get<d>(workspace.projection_and_expand_workspace);

                subset_utils::transform_to_loi(m_set, index_range_func, d_ic, projection_func, workspace.tmp_child_workspace, list_of_intervals);

                return traverser_t<d>(list_of_intervals.cbegin(), list_of_intervals.cend());
            }
        }

      private:

        Set m_set;
        std::size_t m_level;
        ProjectionType m_projectionType;
        std::size_t m_shift;

        expansion_t m_expansions;
    };

} // namespace samurai
