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
        with open(
            os.path.join(build_dir, "{name}_c.{ext}".format(name="graph", ext="json")), "w"
        ) as f_graph_json:
            f_graph_json.write(graph)
        with open(
            os.path.join(build_dir, "{name}_c.{ext}".format(name="params", ext="bin")), "wb"
        ) as f_params:
            f_params.write(tvm_runtime.save_param_dict(params))

        # Generate header for output TODO: Possibly add header for input too
        extra_tar_file = os.path.join(build_dir, "output.tar")

        with tarfile.open(extra_tar_file, "w:gz") as tf:
            create_header_file(
                "output_data",
                np.zeros(
                    shape=output_shape,
                    dtype=output_dtype,
                ),
                "include/tvm",
                tf,
            )
