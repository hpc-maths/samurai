#pragma once 

#include <iostream>

#define SAMURAI_ASSERT(condition, msg) \
do { \
    if (! (condition)) { \
        std::cerr << "Assertion failed \nin " << __FILE__ \
                    << "\n @line " << __LINE__ << ": " << msg << std::endl; \
        std::terminate(); \
    } \
} while (false)

//#ifdef NDEBUG
#define SAMURAI_LOG( msg ) do { std::cerr <<  "SMR::Log:: " \
                                          <<  msg << std::endl; } while (0)

#define SAMURAI_TRACE( msg ) do { std::cerr <<  "SMR::Trace[line " << __LINE__ << "] :" \
                                          <<  msg << std::endl; } while (0)
//#else
//#define MGS_LOG( msg )
//#endif
