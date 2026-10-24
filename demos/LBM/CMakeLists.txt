# New schemes/lbm formalism (see notes/schemes-lbm-design.md)

add_executable(lbm-new-D1Q2-advection new_D1Q2_advection.cpp)
target_link_libraries(lbm-new-D1Q2-advection samurai)

add_executable(lbm-new-D2Q4-advection new_D2Q4_advection.cpp)
target_link_libraries(lbm-new-D2Q4-advection samurai)

add_executable(lbm-new-D1Q3-shallow-waters-dam new_D1Q3_shallow_waters_dam.cpp)
target_link_libraries(lbm-new-D1Q3-shallow-waters-dam samurai)

add_executable(lbm-new-D2Q4diag-advection new_D2Q4diag_advection.cpp)
target_link_libraries(lbm-new-D2Q4diag-advection samurai)

add_executable(lbm-new-D2Q9-taylor-green new_D2Q9_taylor_green.cpp)
target_link_libraries(lbm-new-D2Q9-taylor-green samurai)

add_executable(lbm-new-D1Q5-shallow-waters-dam new_D1Q5_shallow_waters_dam.cpp)
target_link_libraries(lbm-new-D1Q5-shallow-waters-dam samurai)

add_executable(lbm-new-D1Q222-euler-sod new_D1Q222_euler_sod.cpp)
target_link_libraries(lbm-new-D1Q222-euler-sod samurai)

add_executable(lbm-new-D2Q4444-euler-lax-liu new_D2Q4444_euler_lax_liu.cpp)
target_link_libraries(lbm-new-D2Q4444-euler-lax-liu samurai)

add_executable(lbm-new-D2Q4444-euler-implosion new_D2Q4444_euler_implosion.cpp)
target_link_libraries(lbm-new-D2Q4444-euler-implosion samurai)

add_executable(lbm-new-D2Q5444-euler-rayleigh-taylor new_D2Q5444_euler_rayleigh_taylor.cpp)
target_link_libraries(lbm-new-D2Q5444-euler-rayleigh-taylor samurai)

add_executable(lbm-new-D2Q9-von-karman new_D2Q9_von_karman.cpp)
target_link_libraries(lbm-new-D2Q9-von-karman samurai)
