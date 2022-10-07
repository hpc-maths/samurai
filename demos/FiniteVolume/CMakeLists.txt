add_executable(finite-volume-amr-burgers-hat AMR_Burgers_Hat.cpp)
target_link_libraries(finite-volume-amr-burgers-hat samurai)

add_executable(finite-volume-level-set level_set.cpp)
target_link_libraries(finite-volume-level-set samurai)

add_executable(finite-volume-level-set-from-scratch level_set_from_scratch.cpp)
target_link_libraries(finite-volume-level-set-from-scratch samurai)

add_executable(finite-volume-advection-1d advection_1d.cpp)
target_link_libraries(finite-volume-advection-1d samurai CLI11::CLI11)

add_executable(finite-volume-advection-2d advection_2d.cpp)
target_link_libraries(finite-volume-advection-2d samurai)

add_executable(finite-volume-scalar-burgers-2d scalar_burgers_2d.cpp)
target_link_libraries(finite-volume-scalar-burgers-2d samurai)

# add_subdirectory(BZ)