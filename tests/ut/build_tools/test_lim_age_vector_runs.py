# SPDX-License-Identifier: Apache-2.0
"""Execute the actual vector-run source against a CPU instruction model."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class TestVectorRuns(unittest.TestCase):
    def test_vector_pipeline_matches_scalar_metadata(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("C++ compiler required")
        program = r"""
#include "age_hit_runs_vector.h"
#include <algorithm>
#include <random>
#include <vector>
using namespace AscendC;
struct Storage {
    std::vector<uint8_t> bytes;
    Storage(uint32_t words): bytes(words*4U, 0xcd) {}
    template <typename T> LocalTensor<T> view() {return {bytes.data(),0,static_cast<uint32_t>(bytes.size())};}
};
int main() {
    std::mt19937 rng(7);
    for (uint32_t n: {64U,128U,2048U})
    for (uint32_t cap: {1U,128U,512U,8320U,8448U,131200U,2097152U})
    for (uint32_t mode=0;mode<5;++mode) {
        std::vector<int32_t> values(n);
        for (uint32_t i=0;i<n;++i) {
            values[i]=static_cast<int32_t>(rng()%cap);
            if (i%5==0) values[i]=cap-1;
            if (i%7==0) values[i]=cap; // invalid neighbor can share the partial last block
            if (i%11==0) values[i]=-1;
            if (i%17==0) values[i]=INT32_MIN;
            if (mode==1 && i<n/2) values[i]=-1;
            if (mode==2) values[i]=-1;
            if (mode==3) values[i]=0;
            if (mode==4) values[i]=static_cast<int32_t>(i%cap);
        }
        std::sort(values.begin(),values.end(),[](int32_t a,int32_t b){return a>b;});
        Storage src(n), scratch(4*n+LIMAge::HIT_RUN_SCRATCH_EXTRA),
            block(n), start(n), end(n), refBlock(n), refStart(n), refEnd(n);
        auto sources=src.view<int32_t>();
        for (uint32_t i=0;i<n;++i) sources.SetValue(i,values[i]);
        auto b=block.view<int32_t>(), s=start.view<int32_t>(), e=end.view<int32_t>();
        auto rb=refBlock.view<int32_t>(), rs=refStart.view<int32_t>(), re=refEnd.view<int32_t>();
        uint64_t bc=0,sc=0,ec=0;
        LIMAge::BuildVectorHitRunMetadata(sources,scratch.view<float>(),b,s,e,n,cap,bc,sc,ec);
        uint32_t count=LIMAge::BuildSortedHitRuns(sources,n,cap,rb,rs,re);
        assert(bc==count && sc==count && ec==count);
        for (uint32_t i=0;i<count;++i) {
            assert(b.GetValue(i)==rb.GetValue(i));
            assert(s.GetValue(i)==rs.GetValue(i));
            assert(e.GetValue(i)==re.GetValue(i));
        }
    }
}
"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cpp = root / "check.cpp"
            cpp.write_text(program)
            binary = root / "check"
            subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-O2",
                    "-I",
                    str(ROOT / "tests/ut/build_tools/fixtures/vector_model"),
                    "-I",
                    str(ROOT / "csrc/attention/fused_lightning_indexer_age_manage/op_kernel"),
                    str(cpp),
                    "-o",
                    str(binary),
                ],
                check=True,
            )
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
