include(FetchContent)
FetchContent_Declare(googletest
    GIT_REPOSITORY      https://github.com/google/googletest.git
    GIT_TAG             v1.16.0)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
    FetchContent_MakeAvailable(googletest)
endif()
