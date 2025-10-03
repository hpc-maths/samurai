#pragma once


#include <iostream>
#include "print.hpp"
#include <fmt/ostream.h>

#define SAMURAI_ASSERT(condition, msg)                                                                                \
    do                                                                                                                \
    {                                                                                                                 \
        if (!(condition))                                                                                             \
        {                                                                                                             \
            samurai::io::eprint("Assertion failed \nin {} \n @line {}: {}\n", __FILE__, __LINE__, fmt::streamed(msg)); \
            std::terminate();                                                                                         \
        }                                                                                                             \
    } while (false)

// #ifdef NDEBUG
#define SAMURAI_LOG(msg)                                \
    do                                                  \
    {                                                   \
        samurai::io::eprint("SMR::Log:: {}\n", fmt::streamed(msg)); \
    } while (0)

#define SAMURAI_TRACE(msg)                                                        \
    do                                                                            \
    {                                                                             \
        samurai::io::eprint("SMR::Trace[line {}] :{}\n", __LINE__, fmt::streamed(msg)); \
    } while (0)
// #else
// #define MGS_LOG( msg )
// #endif
