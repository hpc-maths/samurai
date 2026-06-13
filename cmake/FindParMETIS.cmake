# FindParMETIS.cmake
# Find the ParMETIS library for parallel graph partitioning.
#
# This module defines:
#   ParMETIS_FOUND        - True if ParMETIS was found
#   ParMETIS_INCLUDE_DIRS - Include directories
#   ParMETIS_LIBRARIES    - Libraries to link against
#   ParMETIS_VERSION      - Version string (if available)
#
# Hints:
#   PARMETIS_ROOT          - Root of the ParMETIS installation

find_path(PARMETIS_INCLUDE_DIR
    NAMES parmetis.h
    HINTS ${PARMETIS_ROOT} $ENV{PARMETIS_ROOT}
    PATH_SUFFIXES include)

find_library(PARMETIS_LIBRARY
    NAMES parmetis
    HINTS ${PARMETIS_ROOT} $ENV{PARMETIS_ROOT}
    PATH_SUFFIXES lib lib64)

# Metis is a dependency of ParMETIS
find_library(METIS_LIBRARY
    NAMES metis
    HINTS ${PARMETIS_ROOT} $ENV{PARMETIS_ROOT}
    PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ParMETIS
    DEFAULT_MSG
    PARMETIS_INCLUDE_DIR
    PARMETIS_LIBRARY
    METIS_LIBRARY)

if(ParMETIS_FOUND)
    set(ParMETIS_INCLUDE_DIRS ${PARMETIS_INCLUDE_DIR})
    set(ParMETIS_LIBRARIES ${PARMETIS_LIBRARY} ${METIS_LIBRARY})

    # Extract version from parmetis.h if available
    if(EXISTS "${PARMETIS_INCLUDE_DIR}/parmetis.h")
        file(STRINGS "${PARMETIS_INCLUDE_DIR}/parmetis.h" _parmetis_version_line
             REGEX "PARMETIS_MAJOR_VERSION.*PARMETIS_MINOR_VERSION.*PARMETIS_SUBMINOR_VERSION")
        if(_parmetis_version_line)
            string(REGEX REPLACE ".*PARMETIS_MAJOR_VERSION[ \t]+([0-9]+).*" "\\1" PARMETIS_VERSION_MAJOR "${_parmetis_version_line}")
            string(REGEX REPLACE ".*PARMETIS_MINOR_VERSION[ \t]+([0-9]+).*" "\\1" PARMETIS_VERSION_MINOR "${_parmetis_version_line}")
            string(REGEX REPLACE ".*PARMETIS_SUBMINOR_VERSION[ \t]+([0-9]+).*" "\\1" PARMETIS_VERSION_PATCH "${_parmetis_version_line}")
            set(ParMETIS_VERSION "${PARMETIS_VERSION_MAJOR}.${PARMETIS_VERSION_MINOR}.${PARMETIS_VERSION_PATCH}")
        endif()
    endif()

    if(NOT TARGET ParMETIS::ParMETIS)
        add_library(ParMETIS::ParMETIS UNKNOWN IMPORTED)
        set_target_properties(ParMETIS::ParMETIS PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${ParMETIS_INCLUDE_DIRS}"
            IMPORTED_LOCATION "${PARMETIS_LIBRARY}")
    endif()
    if(NOT TARGET ParMETIS::Metis)
        add_library(ParMETIS::Metis UNKNOWN IMPORTED)
        set_target_properties(ParMETIS::Metis PROPERTIES
            IMPORTED_LOCATION "${METIS_LIBRARY}")
    endif()

    mark_as_advanced(PARMETIS_INCLUDE_DIR PARMETIS_LIBRARY METIS_LIBRARY)
endif()