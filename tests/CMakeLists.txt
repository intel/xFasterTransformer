if (NOT XFT_BUILD_TESTS)
    return()
endif()
message(STATUS "Build xfastertransformer tests...")

add_subdirectory(ut)