#!/usr/bin/env python3

import os
import sys
import pathlib
import argparse
import tvm
import tvm.micro
import tvm.micro.testing
import onnx
import tvm.contrib.utils
from tvm import runtime as tvm_runtime
from tvm import te
from tvm import relay

def onnx_to_relay(onnx_model_path):
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    mod, params = relay.frontend.from_onnx(onnx_model)
    return mod, params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_dir", default=".")
    parser.add_argument("-m", "--model_path", default=".")

    opts = parser.parse_args()

    mod, params = onnx_to_relay(opts.model_path)
    RUNTIME = tvm.relay.backend.Runtime("crt", {"system-lib" : True})
    TARGET = tvm.micro.testing.get_target("crt")
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        graph, lib, params = relay.build(mod, target=TARGET, runtime=RUNTIME, params=params)
        build_dir = os.path.abspath(opts.out_dir)
        if not os.path.isdir(build_dir):
            os.makedirs(build_dir)
        lib.export_library(os.path.join(build_dir, "model_c.tar"))

        graph_path = os.path.join(build_dir, "{name}_c.{ext}".format(name="graph", ext="json"))
        with open(graph_path,"w") as f_graph_json:
            f_graph_json.write(graph)

        params_path = os.path.join(build_dir, "{name}_c.{ext}".format(name="params", ext="bin"))
        with open(params_path, "wb") as f_params:
            f_params.write(tvm_runtime.save_param_dict(params))

    # Convert graph and weights to hexdumps
    graph_path_rel = os.path.relpath("{name}_c.{ext}".format(name="graph", ext="json"))
    params_path_rel = os.path.relpath("{name}_c.{ext}".format(name="params", ext="bin"))
    os.chdir(build_dir)
    os.system("xxd -i {graph} > {graphc} ".format(graph=graph_path_rel, graphc=(graph_path_rel + ".c")))
    os.system("xxd -i {params} > {paramsc} ".format(params=params_path_rel, paramsc=(params_path_rel + ".c")))
