cmake_minimum_required(VERSION 3.13)

project(headsail-crt C CXX ASM)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)

# add example code
add_executable(microtvm.bin
    model_c/lib0.c
    model_c/lib1.c
    model_c/devc.c
    model_c/graph_c.json.c
    model_c/params_c.bin.c
    src/main.c
    src/bundle_static.c
    $ENV{TVM_HOME}/build/standalone_crt/src/runtime/crt/common/crt_runtime_api.c
    $ENV{TVM_HOME}/build/standalone_crt/src/runtime/crt/graph_executor/graph_executor.c
    $ENV{TVM_HOME}/build/standalone_crt/src/runtime/crt/memory/page_allocator.c
    $ENV{TVM_HOME}/build/standalone_crt/src/runtime/crt/common/packed_func.c
    $ENV{TVM_HOME}/build/standalone_crt/src/runtime/crt/common/func_registry.c
    $ENV{TVM_HOME}/build/standalone_crt/src/runtime/crt/graph_executor/load_json.c
    $ENV{TVM_HOME}/build/standalone_crt/src/runtime/crt/common/crt_runtime_api.c
    $ENV{TVM_HOME}/build/standalone_crt/src/runtime/crt/common/ndarray.c
    $ENV{TVM_HOME}/build/standalone_crt/src/runtime/crt/common/crt_backend_api.c
)

target_include_directories(microtvm.bin
    PUBLIC $ENV{TVM_HOME}/build/standalone_crt/include
)

target_include_directories(microtvm.bin
    PUBLIC $ENV{TVM_HOME}/build/crt_config
)

target_link_libraries(microtvm.bin)

execute_process(COMMAND python generate_crt.py -o ./build -m ../torch/model.onnx)
