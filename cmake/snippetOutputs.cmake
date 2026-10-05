# Program outputs shown in the documentation.
#
# A snippet whose output a page shows declares it with samurai_snippet_output.
# The target update_snippet_outputs runs every declared snippet and writes its
# output to a <name>_output.txt file next to the snippet source, which the page
# includes with literalinclude. The CI runs it and fails if a file changes.
#
#   samurai_snippet_output(<target>
#       [OUTPUT <file>]             file name next to the snippet source,
#                                   ending with _output.txt
#                                   (default: <target>_output.txt)
#       [ARGS <arg>...]             command-line arguments of the program
#       [NP <n>]                    run with mpiexec on n processes
#       [EXIT_CODE <code>]          expected exit code (default: 0)
#       [STDERR]                    capture the standard error too
#                                   (default: standard output only)
#       [INPUTS <file>...]          files copied into the working directory,
#                                   relative to the snippet source
#       [REPLACE <regex> <text>...] pairs applied to the output, for the
#                                   values that change from run to run
#                                   (Python regular expressions)
#       [HEAD <n>] [TAIL <n>]       keep the first and last lines only,
#                                   with "..." in place of the others
#
# An output declared with NP is generated only in a build with WITH_MPI=ON, an
# output declared without NP only in a build without MPI: the same program can
# print other options or values when it is compiled with MPI.

include_guard(GLOBAL)

find_package(Python3 COMPONENTS Interpreter)

add_custom_target(update_snippet_outputs)
if(NOT Python3_Interpreter_FOUND)
    add_custom_command(TARGET update_snippet_outputs POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "update_snippet_outputs needs a Python 3 interpreter"
        COMMAND ${CMAKE_COMMAND} -E false
        VERBATIM)
endif()

if(WITH_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
endif()

set(SAMURAI_SNIPPET_OUTPUT_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/snippet_output.py")

function(samurai_snippet_output target)
    cmake_parse_arguments(PARSE_ARGV 1 arg
        "STDERR"
        "OUTPUT;NP;EXIT_CODE;HEAD;TAIL"
        "ARGS;INPUTS;REPLACE")

    if(arg_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "samurai_snippet_output(${target}): unknown arguments ${arg_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT TARGET ${target})
        message(FATAL_ERROR "samurai_snippet_output(${target}): no such target")
    endif()
    if(NOT Python3_Interpreter_FOUND)
        return()
    endif()
    if(arg_NP AND NOT WITH_MPI)
        return()
    endif()
    if(NOT arg_NP AND WITH_MPI)
        return()
    endif()

    # The output file sits next to the snippet source
    get_target_property(sources ${target} SOURCES)
    list(GET sources 0 source)
    get_filename_component(source_dir "${source}" DIRECTORY)
    if(NOT IS_ABSOLUTE "${source_dir}")
        get_target_property(target_dir ${target} SOURCE_DIR)
        set(source_dir "${target_dir}/${source_dir}")
    endif()
    if(NOT arg_OUTPUT)
        set(arg_OUTPUT "${target}_output.txt")
    endif()
    if(NOT arg_OUTPUT MATCHES "^[A-Za-z0-9_]+_output\\.txt$")
        message(FATAL_ERROR "samurai_snippet_output(${target}): OUTPUT ${arg_OUTPUT} must be a file name ending with _output.txt")
    endif()
    set(output "${source_dir}/${arg_OUTPUT}")

    get_property(declared GLOBAL PROPERTY SAMURAI_SNIPPET_OUTPUTS)
    if(output IN_LIST declared)
        message(FATAL_ERROR "samurai_snippet_output(${target}): ${arg_OUTPUT} is already declared")
    endif()
    set_property(GLOBAL APPEND PROPERTY SAMURAI_SNIPPET_OUTPUTS "${output}")

    set(options --output "${output}")
    if(DEFINED arg_EXIT_CODE)
        list(APPEND options --exit-code ${arg_EXIT_CODE})
    endif()
    if(arg_STDERR)
        list(APPEND options --stderr)
    endif()
    foreach(input IN LISTS arg_INPUTS)
        list(APPEND options --input "${source_dir}/${input}")
    endforeach()
    list(LENGTH arg_REPLACE n_replace)
    math(EXPR odd "${n_replace} % 2")
    if(odd)
        message(FATAL_ERROR "samurai_snippet_output(${target}): REPLACE takes pairs of a regular expression and a replacement")
    endif()
    while(arg_REPLACE)
        list(POP_FRONT arg_REPLACE pattern replacement)
        list(APPEND options "--replace=${pattern}" "--by=${replacement}")
    endwhile()
    if(arg_HEAD)
        list(APPEND options --head ${arg_HEAD})
    endif()
    if(arg_TAIL)
        list(APPEND options --tail ${arg_TAIL})
    endif()
    if(arg_NP)
        list(APPEND options "--mpiexec=${MPIEXEC_EXECUTABLE}" "--numproc-flag=${MPIEXEC_NUMPROC_FLAG}" "--np=${arg_NP}")
        foreach(flag IN LISTS MPIEXEC_PREFLAGS)
            list(APPEND options "--mpiexec-preflag=${flag}")
        endforeach()
    endif()

    string(REGEX REPLACE "_output\\.txt$" "" name "${arg_OUTPUT}")
    list(JOIN arg_ARGS " " args_text)
    add_custom_target(snippet_output_${name}
        COMMAND ${Python3_EXECUTABLE} ${SAMURAI_SNIPPET_OUTPUT_SCRIPT} ${options} -- $<TARGET_FILE:${target}> ${arg_ARGS}
        COMMENT "Running ${target} ${args_text} for ${arg_OUTPUT}"
        VERBATIM)
    add_dependencies(snippet_output_${name} ${target})
    add_dependencies(update_snippet_outputs snippet_output_${name})
endfunction()
