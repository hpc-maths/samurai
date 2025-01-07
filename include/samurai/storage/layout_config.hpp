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

#ifndef SAMURAI_DEFAULT_LAYOUT
#define SAMURAI_DEFAULT_LAYOUT ::samurai::layout_type::row_major
    // #define SAMURAI_DEFAULT_LAYOUT ::samurai::layout_type::column_major
#endif
}
