include(FetchContent)
FetchContent_Declare(highfive
    GIT_REPOSITORY      https://github.com/BlueBrain/HighFive
    GIT_TAG             v2.8.0)
FetchContent_GetProperties(highfive)
if(NOT highfive_POPULATED)
    FetchContent_Populate(highfive)
    set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE BOOL "")
    add_subdirectory(${highfive_SOURCE_DIR} ${highfive_BINARY_DIR} EXCLUDE_FROM_ALL)
    unset(CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
endif()
