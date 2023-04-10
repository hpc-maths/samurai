include(FetchContent)
FetchContent_Declare(googlebench
    GIT_REPOSITORY      https://github.com/google/benchmark.git
    GIT_TAG             v1.5.2)
FetchContent_GetProperties(googlebench)
if(NOT googlebench_POPULATED)
    FetchContent_Populate(googlebench)
    set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE BOOL "")
    add_subdirectory(${googlebench_SOURCE_DIR} ${googlebench_BINARY_DIR} EXCLUDE_FROM_ALL)
    unset(CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
endif()
