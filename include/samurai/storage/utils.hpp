// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

namespace samurai
{
    template <class T>
    struct range_t
    {
        T start;
        T end;
        T step = 1;
    };
}
