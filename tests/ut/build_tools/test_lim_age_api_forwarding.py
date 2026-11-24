# SPDX-License-Identifier: Apache-2.0
"""Compile and execute the actual ACLNN forwarding layer with stub ACL types."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
API = ROOT / "csrc/attention/fused_lightning_indexer_age_manage/op_host/op_api"


class TestLimAgeApiForwarding(unittest.TestCase):
    def test_versioned_forwarding_and_stale_generated_header(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("C++ compiler required")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "aclnn").mkdir()
            (root / "aclnn/acl_meta.h").write_text(
                "#pragma once\n#include <cstdint>\nstruct aclTensor {}; struct aclOpExecutor {};\n"
                "using aclnnStatus=int; using aclrtStream=void*;\n"
            )
            (root / "aclnn/aclnn_base.h").write_text('#include "acl_meta.h"\n')
            inputs = ", ".join(f"const aclTensor *in{i}" for i in range(14))
            outputs = ", ".join(f"const aclTensor *out{i}" for i in range(8))
            signature = (
                "aclnnStatus aclnnInnerFusedLightningIndexerAgeManageGetWorkspaceSize("
                f"{inputs}, int64_t probation, int64_t flags, {outputs}, uint64_t *size, aclOpExecutor **executor)"
            )
            generated = root / "aclnnInner_fused_lightning_indexer_age_manage.h"
            generated.write_text(
                '#include "aclnn/acl_meta.h"\nextern "C" {\n' + signature + ";\n"
                "aclnnStatus aclnnInnerFusedLightningIndexerAgeManage("
                "void*, uint64_t, aclOpExecutor*, aclrtStream);\n}\n"
            )
            harness = root / "main.cpp"
            checks = "\n".join(f"assert(in{i} == &items[{i}]);" for i in range(14))
            checks += "\n" + "\n".join(f"assert(out{i} == &items[{14 + i}]);" for i in range(8))
            input_args = ", ".join(f"&items[{i}]" for i in range(14))
            output_args = ", ".join(f"&items[{14 + i}]" for i in range(8))
            harness.write_text(
                '#include "aclnn_fused_lightning_indexer_age_manage.h"\n#include <cassert>\n'
                'aclTensor items[22]; int64_t expectedFlags=3;\nextern "C" '
                + signature
                + " {\n"
                + checks
                + "\nassert(probation==4); assert(flags==expectedFlags); *size=0; return 0; }\n"
                'extern "C" aclnnStatus aclnnInnerFusedLightningIndexerAgeManage('
                "void*, uint64_t, aclOpExecutor*, aclrtStream) { return 0; }\n"
                "int main() { uint64_t size=0; aclOpExecutor *executor=nullptr;\n"
                f"assert(aclnnFusedLightningIndexerAgeManageV2GetWorkspaceSize({input_args}, 4, 3, "
                f"{output_args}, &size, &executor)==0);\n"
                "expectedFlags=0;\n"
                f"assert(aclnnFusedLightningIndexerAgeManageGetWorkspaceSize({input_args}, 4, "
                f"{output_args}, &size, &executor)==0);\n"
                "assert(aclnnFusedLightningIndexerAgeManage(nullptr, 0, nullptr, nullptr)==0);\n"
                "assert(aclnnFusedLightningIndexerAgeManageV2(nullptr, 0, nullptr, nullptr)==0);\n}\n"
            )
            wrapper = API / "aclnn_fused_lightning_indexer_age_manage.cpp"
            binary = root / "forwarding"
            command = [compiler, "-std=c++17", "-I", str(root), "-I", str(API), str(wrapper)]
            good = subprocess.run(command + [str(harness), "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(good.returncode, 0, good.stdout + good.stderr)
            subprocess.run([str(binary)], check=True)
            generated.write_text(generated.read_text().replace("int64_t flags, ", ""))
            stale = subprocess.run(command + ["-fsyntax-only"], capture_output=True, text=True)
            self.assertNotEqual(stale.returncode, 0, "A stale single-attribute header must not compile")


if __name__ == "__main__":
    unittest.main()
