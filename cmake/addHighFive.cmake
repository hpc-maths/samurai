include(FetchContent)
FetchContent_Declare(HighFive
    GIT_REPOSITORY      https://github.com/BlueBrain/HighFive
    GIT_TAG             v2.8.0)
FetchContent_GetProperties(highfive)
if(NOT HighFive_POPULATED)
    FetchContent_Populate(HighFive)
    set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE BOOL "")
    add_subdirectory(${HighFive_SOURCE_DIR} ${HighFive_BINARY_DIR} EXCLUDE_FROM_ALL)
    unset(CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
endif()
