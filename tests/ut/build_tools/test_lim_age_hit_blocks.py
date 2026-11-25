# SPDX-License-Identifier: Apache-2.0
"""Run the actual kernel bucket/clear helpers against a scalar reference."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class TestLimAgeHitBlocks(unittest.TestCase):
    def test_bucket_clear_matches_reference(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("C++ compiler required")
        program = r"""
#include "age_hit_blocks.h"
#include <algorithm>
#include <cassert>
#include <random>
#include <set>
#include <vector>
struct Tensor {
    std::vector<int32_t> data;
    int32_t GetValue(uint32_t i) const { return data.at(i); }
    void SetValue(uint32_t i, int32_t v) { data.at(i)=v; }
};
void check(uint32_t capacity, uint32_t misses, std::mt19937 &rng) {
    constexpr uint32_t K=2048;
    Tensor sources{std::vector<int32_t>(K)}, next{std::vector<int32_t>(K)},
        heads{std::vector<int32_t>((capacity+511)/512,-1)}, touched{std::vector<int32_t>(K)};
    std::vector<int32_t> ages(capacity), expected(capacity);
    for (uint32_t i=0;i<capacity;++i) ages[i]=expected[i]=17+i%1024;
    std::set<uint32_t> blocks;
    uint32_t count=0;
    for (uint32_t i=0;i<K;++i) {
        int32_t source=static_cast<int32_t>(rng()%capacity);
        if (i%31==0) source=-1;
        if (i%37==0) source=capacity;
        if (i%29==0) source=capacity-1; // partial final block and repeated hits
        sources.SetValue(i, source);
        if (i<misses) continue;
        LIMAge::AppendHitBlock(source,i,capacity,heads,next,touched,count);
        if (source>=0 && static_cast<uint32_t>(source)<capacity) {
            expected[source]=0;
            blocks.insert(static_cast<uint32_t>(source)/512);
        }
    }
    assert(count==blocks.size());
    for (uint32_t i=0;i<count;++i) {
        uint32_t block=touched.GetValue(i), start=block*512;
        uint32_t n=std::min(512U,capacity-start);
        Tensor tile{std::vector<int32_t>(512,-777)};
        std::copy_n(ages.begin()+start,n,tile.data.begin());
        LIMAge::ClearHitBlock(heads.GetValue(block),sources,next,tile);
        assert(std::all_of(tile.data.begin()+n,tile.data.end(),[](int x){return x==-777;}));
        std::copy_n(tile.data.begin(),n,ages.begin()+start);
    }
    assert(ages==expected);
    // Exercise the new production run builder/clear helper on sorted full IDs.
    // Sorting itself is AscendC SortAll on device; std::sort is a CPU surrogate.
    for (uint32_t i=0;i<misses;++i) sources.data[i]=-1;
    std::sort(sources.data.begin(),sources.data.end(),[](int32_t a,int32_t b){return a>b;});
    Tensor runBlocks{std::vector<int32_t>(K)}, starts{std::vector<int32_t>(K)}, ends{std::vector<int32_t>(K)};
    uint32_t runs=LIMAge::BuildSortedHitRuns(sources,K,capacity,runBlocks,starts,ends);
    assert(runs==blocks.size());
    for (uint32_t i=0;i<capacity;++i) ages[i]=17+i%1024;
    for (uint32_t i=0;i<runs;++i) {
        uint32_t block=runBlocks.GetValue(i), start=block*512;
        uint32_t first=starts.GetValue(i), last=ends.GetValue(i);
        assert(first<last && last<=K);
        if (i>0) assert(runBlocks.GetValue(i-1)>static_cast<int32_t>(block));
        for (uint32_t j=first;j<last;++j) assert(static_cast<uint32_t>(sources.GetValue(j))/512==block);
        uint32_t n=std::min(512U,capacity-start);
        Tensor tile{std::vector<int32_t>(512,-777)};
        std::copy_n(ages.begin()+start,n,tile.data.begin());
        LIMAge::ClearSortedHitRun(first,last,sources,tile);
        assert(std::all_of(tile.data.begin()+n,tile.data.end(),[](int x){return x==-777;}));
        std::copy_n(tile.data.begin(),n,ages.begin()+start);
    }
    assert(ages==expected);
}
int main() {
    std::mt19937 rng(7);
    for (uint32_t capacity: {128U,512U,8320U,8448U,131200U,2097152U})
        for (uint32_t misses: {0U,1U,1024U,2047U,2048U}) check(capacity,misses,rng);
}
"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "check.cpp"
            source.write_text(program)
            binary = root / "check"
            include = ROOT / "csrc/attention/fused_lightning_indexer_age_manage/op_kernel"
            subprocess.run(
                [compiler, "-std=c++17", "-O2", "-I", str(include), str(source), "-o", str(binary)], check=True
            )
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
