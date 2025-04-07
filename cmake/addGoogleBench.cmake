include(FetchContent)
FetchContent_Declare(googlebench
    GIT_REPOSITORY      https://github.com/google/benchmark.git
    GIT_TAG             v1.9.2)
FetchContent_GetProperties(googlebench)
if(NOT googlebench_POPULATED)
    FetchContent_MakeAvailable(googlebench)
endif()
