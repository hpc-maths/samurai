include(FetchContent)
FetchContent_Declare(rapidcheck
    GIT_REPOSITORY      https://github.com/emil-e/rapidcheck
)
FetchContent_GetProperties(rapidcheck)
if(NOT rapidcheck_POPULATED)
    FetchContent_Populate(rapidcheck)
    set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE BOOL "")
    set(RC_ENABLE_GTEST true CACHE BOOL "")
    add_subdirectory(${rapidcheck_SOURCE_DIR} ${rapidcheck_BINARY_DIR} EXCLUDE_FROM_ALL)
    unset(CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
endif()
