# SPDX-License-Identifier: Apache-2.0
"""Exercise the real opbuild CMake dependency graph without CANN/NPU."""

import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class TestOpbuildRegeneration(unittest.TestCase):
    def test_definition_edit_regenerates_aclnn(self):
        if not shutil.which("cmake") or not shutil.which("c++") or not shutil.which("make"):
            self.skipTest("cmake, make and a C++ compiler are required")
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            definition = source / "example_def.cpp"
            definition.write_text("int example_definition = 1;\n")
            generator = source / "mock_opbuild.py"
            generator.write_text(
                f"#!{sys.executable}\n"
                "import pathlib, sys\n"
                "out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)\n"
                "with (out / 'runs.txt').open('a') as f: f.write('generated\\n')\n"
                "(out / 'api.h').write_text('// generated header\\n')\n"
                "(out / 'api.cpp').write_text('// generated source\\n')\n"
            )
            generator.chmod(0o755)
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.16)\nproject(regeneration LANGUAGES CXX)\n"
                "foreach(name intf_pub_cxx17 exe_graph register c_sec)\n"
                "add_library(${name} INTERFACE)\nendforeach()\n"
                f'include("{ROOT}/csrc/cmake/opbuild.cmake")\n'
                f'set(OP_BUILD_TOOL "{generator}")\n'
                "gen_opbuild_target(TARGET generate PREFIX test GENACLNN 1\n"
                'OUT_DIR "${CMAKE_BINARY_DIR}" OUT_SUB_DIR generated\n'
                'IN_SRCS "${CMAKE_SOURCE_DIR}/example_def.cpp"\n'
                'OUT_SRCS "${CMAKE_BINARY_DIR}/generated/api.cpp"\n'
                'OUT_HEADERS "${CMAKE_BINARY_DIR}/generated/api.h")\n'
            )
            build = source / "build"
            self.run_command("cmake", "-S", str(source), "-B", str(build), "-G", "Unix Makefiles")
            self.run_command("cmake", "--build", str(build), "--target", "generate")
            runs = build / "generated/runs.txt"
            self.assertEqual(len(runs.read_text().splitlines()), 1)
            self.run_command("cmake", "--build", str(build), "--target", "generate")
            self.assertEqual(len(runs.read_text().splitlines()), 1, "No-op rebuild must reuse generated APIs")
            time.sleep(1.1)  # Make implementations may compare timestamps at second resolution.
            definition.write_text("int example_definition = 2;\n")
            self.run_command("cmake", "--build", str(build), "--target", "generate")
            self.assertEqual(len(runs.read_text().splitlines()), 2, "Definition edit must regenerate ACLNN APIs")

    def run_command(self, *command):
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
