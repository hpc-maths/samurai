add_executable(finite-volume-amr-burgers-hat VF_AMR_Burgers_Hat.cpp)
target_link_libraries(finite-volume-amr-burgers-hat samurai)

add_executable(finite-volume-level-set VF_level_set.cpp)
target_link_libraries(finite-volume-level-set samurai)

add_executable(finite-volume-level-set-from-scratch VF_level_set_from_scratch.cpp)
target_link_libraries(finite-volume-level-set-from-scratch samurai)

add_executable(finite-volume-advection-1d VF_advection_1d.cpp)
target_link_libraries(finite-volume-advection-1d samurai)

add_executable(finite-volume-advection-2d VF_advection_2d.cpp)
target_link_libraries(finite-volume-advection-2d samurai)

add_executable(finite-volume-scalar-burgers-2d VF_scalar_burgers_2d.cpp)
target_link_libraries(finite-volume-scalar-burgers-2d samurai)

add_subdirectory(BZ)