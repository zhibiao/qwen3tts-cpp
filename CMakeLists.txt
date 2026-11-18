cmake_minimum_required(VERSION 3.20)

project(qwen3_tts_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 所有可执行文件与 ggml 的共享库统一输出到 <build>/bin/<config>，
# 这样 exe 旁边就是全部 backend DLL，不需要额外拷贝。
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(BUILD_SHARED_LIBS ON CACHE BOOL "build shared libraries")

if(BUILD_SHARED_LIBS)
    set(GGML_BACKEND_DL ON CACHE BOOL "" FORCE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "" FORCE)
else()
    set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
endif()
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)

option(
    QWEN3TTS_ENABLE_CUDA
    "Build Qwen3TTS with the GGML CUDA backend"
    ON
)
set(
    _QWEN3TTS_DEFAULT_CUDA_ARCHITECTURES
    "75-real;86-real;89-real;120-real;120-virtual"
)
set(
    QWEN3TTS_CUDA_ARCHITECTURES
    "${_QWEN3TTS_DEFAULT_CUDA_ARCHITECTURES}"
    CACHE STRING
    "CUDA architectures: -real = SASS, -virtual = PTX, bare = both"
)
# 旧 CMakeCache 里可能留着空值；空值时回落到默认列表。
if(NOT QWEN3TTS_CUDA_ARCHITECTURES)
    set(
        QWEN3TTS_CUDA_ARCHITECTURES
        "${_QWEN3TTS_DEFAULT_CUDA_ARCHITECTURES}"
        CACHE STRING "" FORCE
    )
endif()

if(QWEN3TTS_ENABLE_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    set(GGML_CUDA_NCCL OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE)
    if(QWEN3TTS_CUDA_ARCHITECTURES)
        set(
            CMAKE_CUDA_ARCHITECTURES
            "${QWEN3TTS_CUDA_ARCHITECTURES}"
            CACHE STRING "" FORCE
        )
    endif()
    message(STATUS "Qwen3TTS CUDA backend enabled")
    message(STATUS "Qwen3TTS CUDA architectures: "
        "${CMAKE_CUDA_ARCHITECTURES}")
else()
    set(GGML_CUDA OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA_GRAPHS OFF CACHE BOOL "" FORCE)
    message(STATUS "Qwen3TTS CUDA backend disabled")
endif()

add_subdirectory(vendor/ggml-0.29.0)

function(qwen3_tts_configure_target target)
    target_include_directories(${target} PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/vendor
    )
    target_compile_definitions(${target} PRIVATE
        UNI_ALGO_STATIC_DATA
    )
    if(MSVC)
        target_compile_options(${target} PRIVATE /utf-8)
    endif()
    target_link_libraries(${target} PRIVATE ggml)
endfunction()

set(QWEN3_TTS_CORE_SOURCES
    ${PROJECT_SOURCE_DIR}/src/config.cpp
    ${PROJECT_SOURCE_DIR}/src/gguf_model.cpp
    ${PROJECT_SOURCE_DIR}/src/talker.cpp
    ${PROJECT_SOURCE_DIR}/src/code_predictor.cpp
    ${PROJECT_SOURCE_DIR}/src/sampler.cpp
    ${PROJECT_SOURCE_DIR}/src/text_tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/src/text_tokenizer_internal.cpp
    ${PROJECT_SOURCE_DIR}/src/speech_tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/src/qwen3_tts.cpp
)

add_executable(qwen3tts-cli
    ${PROJECT_SOURCE_DIR}/app/qwen3tts_cli.cpp
    ${QWEN3_TTS_CORE_SOURCES}
    ${PROJECT_SOURCE_DIR}/src/wav_writer.cpp
)
qwen3_tts_configure_target(qwen3tts-cli)
if(WIN32)
    target_link_libraries(qwen3tts-cli PRIVATE shell32)
endif()

include(CTest)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
