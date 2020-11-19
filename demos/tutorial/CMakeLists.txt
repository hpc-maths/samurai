add_executable(test interval.cpp)
target_link_libraries(test PRIVATE samurai)

# add_executable(2D_mesh 2D_mesh.cpp)
# target_link_libraries(2D_mesh PRIVATE samurai)

# add_executable(simple_2d simple_2d.cpp)
# target_link_libraries(simple_2d PRIVATE samurai)

# add_executable(test_mesh_base test_mesh_base.cpp)
# target_link_libraries(test_mesh_base PRIVATE samurai)

# add_executable(fix_bug fix_bug.cpp)
# target_link_libraries(fix_bug PRIVATE samurai)

add_executable(graduation_case_3 graduation_case_3.cpp)
target_link_libraries(graduation_case_3 PRIVATE samurai)

# add_executable(set_operator set_operator.cpp)
# target_link_libraries(set_operator PRIVATE samurai)
