// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../difference_id_traverser.hpp"
#include "../difference_traverser.hpp"
#include "../intersection_traverser.hpp"
#include "../union_traverser.hpp"

namespace samurai
{
    enum class NAryTraverserType
    {
        UNION,
        INTERSECTION,
        DIFFERENCE,
        DIFFERENCE_ID
    };

    template <NAryTraverserType Op, class... SetTraversers>
    struct NAryTraverserTypeTraits;

    template <class... SetTraversers>
    struct NAryTraverserTypeTraits<NAryTraverserType::UNION, SetTraversers...>
    {
        using Type = UnionTraverser<SetTraversers...>;
    };

    template <class... SetTraversers>
    struct NAryTraverserTypeTraits<NAryTraverserType::INTERSECTION, SetTraversers...>
    {
        using Type = IntersectionTraverser<SetTraversers...>;
    };

    template <class... SetTraversers>
    struct NAryTraverserTypeTraits<NAryTraverserType::DIFFERENCE, SetTraversers...>
    {
        using Type = DifferenceTraverser<SetTraversers...>;
    };

    template <class... SetTraversers>
    struct NAryTraverserTypeTraits<NAryTraverserType::DIFFERENCE_ID, SetTraversers...>
    {
        using Type = DifferenceIdTraverser<SetTraversers...>;
    };

} // namespace samurai
