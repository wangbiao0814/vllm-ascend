# SPDX-License-Identifier: Apache-2.0
"""Check age-only source staging using the real dependency/copy CMake code."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class TestLimAgeSharedHeaders(unittest.TestCase):
    def test_age_source_copy_stages_declared_limiter_headers(self):
        if not shutil.which("cmake") or not shutil.which("c++") or not shutil.which("make"):
            self.skipTest("cmake, make and C++ compiler required")
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            (project / "empty.cpp").write_text("int empty_stub;\n")
            (project / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.16)\nproject(stage LANGUAGES CXX)\n"
                f'include("{ROOT}/csrc/cmake/func.cmake")\n'
                "macro(add_modules_sources)\nendmacro()\n"
                "function(add_ops_compile_options)\nendfunction()\n"
                "set(OPHOST_NAME stub)\nset(BUILD_OPEN_PROJECT OFF)\n"
                "add_library(stub_opapi_obj OBJECT empty.cpp)\n"
                "add_library(stub_tiling_obj OBJECT empty.cpp)\n"
                f'add_subdirectory("{ROOT}/csrc/attention/fused_lightning_indexer_age_manage/op_host" host)\n'
                "if(NOT fused_lightning_indexer_age_manage_depends)\n"
                'message(FATAL_ERROR "Age-only staging requires declared shared inputs")\nendif()\n'
                'set(stage "${CMAKE_BINARY_DIR}/prepared")\n'
                f'add_ops_src_copy(TARGET_NAME age_copy SRC "{ROOT}/csrc/attention/fused_lightning_indexer_age_manage" '
                'DST "${stage}/fused_lightning_indexer_age_manage" COMPUTE_UNIT ascend910b)\n'
                "foreach(dep ${fused_lightning_indexer_age_manage_depends})\n"
                'get_filename_component(name "${dep}" NAME)\n'
                f'add_ops_src_copy(TARGET_NAME shared_copy SRC "{ROOT}/csrc/${{dep}}" '
                'DST "${stage}/${name}" COMPUTE_UNIT ascend910b BE_RELIED age_copy)\n'
                "endforeach()\n"
            )
            build = project / "build"
            for command in (
                ["cmake", "-S", str(project), "-B", str(build), "-G", "Unix Makefiles"],
                ["cmake", "--build", str(build), "--target", "age_copy"],
            ):
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            kernel = build / "prepared/fused_lightning_indexer_age_manage/op_kernel"
            for name in (
                "fused_lightning_indexer_manage_template_tiling_key.h",
                "fused_lightning_indexer_manage_union.h",
            ):
                staged = kernel / "../../fused_lightning_indexer_manage/op_kernel" / name
                self.assertTrue(staged.is_file(), str(staged))
                original = ROOT / "csrc/attention/fused_lightning_indexer_manage/op_kernel" / name
                self.assertEqual(staged.read_bytes(), original.read_bytes())
            self.assertTrue((kernel / "age_hit_blocks.h").is_file())


if __name__ == "__main__":
    unittest.main()
