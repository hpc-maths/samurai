# FindPTScotch.cmake
# Find the PT-Scotch library for parallel graph partitioning.
#
# This module defines:
#   PTScotch_FOUND        - True if PT-Scotch was found
#   PTScotch_INCLUDE_DIRS - Include directories
#   PTScotch_LIBRARIES    - Libraries to link against
#   PTScotch_VERSION      - Version string (if available)
#
# Hints:
#   PTSCOTCH_ROOT          - Root of the PT-Scotch installation

find_path(PTSCOTCH_INCLUDE_DIR
    NAMES ptscotch.h
    HINTS ${PTSCOTCH_ROOT} $ENV{PTSCOTCH_ROOT} $ENV{PTSCOTCH_ROOT}/include
    PATH_SUFFIXES include include/scotch)

find_library(PTSCOTCH_LIBRARY
    NAMES ptscotch
    HINTS ${PTSCOTCH_ROOT} $ENV{PTSCOTCH_ROOT}
    PATH_SUFFIXES lib lib64)

find_library(SCOTCH_LIBRARY
    NAMES scotch
    HINTS ${PTSCOTCH_ROOT} $ENV{PTSCOTCH_ROOT}
    PATH_SUFFIXES lib lib64)

find_library(PTSCOTCHERR_LIBRARY
    NAMES ptscotcherr
    HINTS ${PTSCOTCH_ROOT} $ENV{PTSCOTCH_ROOT}
    PATH_SUFFIXES lib lib64)

find_library(SCOTCHERR_LIBRARY
    NAMES scotcherr
    HINTS ${PTSCOTCH_ROOT} $ENV{PTSCOTCH_ROOT}
    PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PTScotch
    DEFAULT_MSG
    PTSCOTCH_INCLUDE_DIR
    PTSCOTCH_LIBRARY
    SCOTCH_LIBRARY)

if(PTScotch_FOUND)
    set(PTScotch_INCLUDE_DIRS ${PTSCOTCH_INCLUDE_DIR})
    set(PTScotch_LIBRARIES ${PTSCOTCH_LIBRARY} ${SCOTCH_LIBRARY})

    # Error libraries are optional (may be linked implicitly)
    if(PTSCOTCHERR_LIBRARY)
        list(APPEND PTScotch_LIBRARIES ${PTSCOTCHERR_LIBRARY})
    endif()
    if(SCOTCHERR_LIBRARY)
        list(APPEND PTScotch_LIBRARIES ${SCOTCHERR_LIBRARY})
    endif()

    # Extract version from scotch.h
    if(EXISTS "${PTSCOTCH_INCLUDE_DIR}/scotch.h")
        file(STRINGS "${PTSCOTCH_INCLUDE_DIR}/scotch.h" _scotch_version_line
             REGEX "SCOTCH_VERSION.*SCOTCH_RELEASE.*SCOTCH_PATCHLEVEL")
        if(_scotch_version_line)
            string(REGEX REPLACE ".*SCOTCH_VERSION[ \t]+([0-9]+).*" "\\1" SCOTCH_VERSION_MAJOR "${_scotch_version_line}")
            string(REGEX REPLACE ".*SCOTCH_RELEASE[ \t]+([0-9]+).*" "\\1" SCOTCH_VERSION_MINOR "${_scotch_version_line}")
            string(REGEX REPLACE ".*SCOTCH_PATCHLEVEL[ \t]+([0-9]+).*" "\\1" SCOTCH_VERSION_PATCH "${_scotch_version_line}")
            set(PTScotch_VERSION "${SCOTCH_VERSION_MAJOR}.${SCOTCH_VERSION_MINOR}.${SCOTCH_VERSION_PATCH}")
        endif()
    endif()

    if(NOT TARGET PTScotch::PTScotch)
        add_library(PTScotch::PTScotch UNKNOWN IMPORTED)
        set_target_properties(PTScotch::PTScotch PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${PTScotch_INCLUDE_DIRS}"
            IMPORTED_LOCATION "${PTSCOTCH_LIBRARY}")
    endif()
    if(NOT TARGET PTScotch::Scotch)
        add_library(PTScotch::Scotch UNKNOWN IMPORTED)
        set_target_properties(PTScotch::Scotch PROPERTIES
            IMPORTED_LOCATION "${SCOTCH_LIBRARY}")
    endif()

    mark_as_advanced(PTSCOTCH_INCLUDE_DIR PTSCOTCH_LIBRARY SCOTCH_LIBRARY PTSCOTCHERR_LIBRARY SCOTCHERR_LIBRARY)
endif()
