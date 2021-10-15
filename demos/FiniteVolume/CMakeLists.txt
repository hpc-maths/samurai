add_executable(finite-volume-amr-burgers-hat VF_AMR_Burgers_Hat.cpp)
target_link_libraries(finite-volume-amr-burgers-hat samurai)

add_executable(finite-volume-level-set VF_level_set.cpp)
target_link_libraries(finite-volume-level-set samurai)

add_executable(finite-volume-level-set-from-scratch VF_level_set_from_scratch.cpp)
target_link_libraries(finite-volume-level-set-from-scratch samurai)
