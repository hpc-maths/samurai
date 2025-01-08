// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

namespace samurai
{
    enum class layout_type
    {
        /*! row major layout_type */
        row_major = 0x00,
        /*! column major layout_type */
        column_major = 0x01
    };

#if defined(SAMURAI_CONTAINER_LAYOUT_COL_MAJOR)
#define SAMURAI_DEFAULT_LAYOUT ::samurai::layout_type::column_major
#else
#define SAMURAI_DEFAULT_LAYOUT ::samurai::layout_type::row_major
#endif
}
