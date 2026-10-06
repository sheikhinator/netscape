# Desktop (x86_64) build of the app's own writer_jni.cpp, so CI can run the exact native code
# that ships in the APK against the real writer model.
cmake_minimum_required(VERSION 3.22)
project(writer_host LANGUAGES C CXX)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)  # needed by the tools
set(LLAMA_BUILD_TOOLS ON CACHE BOOL "" FORCE)   # llama-quantize
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(${LLAMA_DIR} llama)
find_package(JNI REQUIRED)
add_library(netscape_writer_host SHARED ${WRITER_JNI})
target_include_directories(netscape_writer_host PRIVATE ${JNI_INCLUDE_DIRS})
target_compile_features(netscape_writer_host PRIVATE cxx_std_17)
target_link_libraries(netscape_writer_host PRIVATE llama)
