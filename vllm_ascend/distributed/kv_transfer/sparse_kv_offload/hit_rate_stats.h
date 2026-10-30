// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <unordered_map>

// Owned by one manager, never process-global. Counts unique valid selections
// per query row so duplicate indices do not distort either policy's hit rate.
class SparseKVHitRateStats {
 public:
  SparseKVHitRateStats(std::vector<std::string> layers, std::string policy, std::string worker,
                       int64_t max_token, int64_t interval, int64_t warmup, bool on_request_end = false)
      : layers_(std::move(layers)), policy_(std::move(policy)), worker_(std::move(worker)),
        interval_(interval), warmup_(warmup), on_request_end_(on_request_end),
        marks_(max_token, 0), counters_(layers_.size()), last_batches_(layers_.size()) {
    if (max_token <= 0 || interval < 0 || warmup < 0) {
      throw std::invalid_argument("Invalid sparse KV hit-rate statistics configuration");
    }
  }

  void record(int64_t layer, uintptr_t selected_ptr, uintptr_t current_ptr, uintptr_t count_ptr,
              uintptr_t misses_ptr, uintptr_t req_ids_ptr, int64_t rows, int64_t topk) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto& counter = counters_.at(layer);
    counter.last_hits = counter.last_misses = 0;
    auto& batch = last_batches_.at(layer);
    batch.clear();
    const auto* selected = reinterpret_cast<const int32_t*>(selected_ptr);
    const auto* current = reinterpret_cast<const int32_t*>(current_ptr);
    const auto* counts = reinterpret_cast<const int32_t*>(count_ptr);
    const auto* misses = reinterpret_cast<const int32_t*>(misses_ptr);
    const auto* req_ids = reinterpret_cast<const int64_t*>(req_ids_ptr);
    for (int64_t row = 0; row < rows; ++row) {
      if (req_ids[row] == 0) continue;  // Graph padding rows carry the zero request ID.
      if (epoch_ >= std::numeric_limits<uint64_t>::max() - 2) {
        std::fill(marks_.begin(), marks_.end(), 0);
        epoch_ = 0;
      }
      epoch_ += 2;
      uint64_t row_hits = 0, row_misses = 0;
      for (int64_t i = 0; i < counts[row]; ++i) {
        const int32_t token = misses[row * topk + i];
        if (valid(token)) marks_[token] = epoch_;
      }
      for (int64_t pos = row * topk; pos < (row + 1) * topk; ++pos) {
        const int32_t token = selected[pos];
        if (!valid(token) || marks_[token] == epoch_ + 1) continue;
        if (marks_[token] == epoch_ || current[pos] < 0) ++row_misses;
        else ++row_hits;
        marks_[token] = epoch_ + 1;
      }
      counter.last_hits += row_hits;
      counter.last_misses += row_misses;
      if (on_request_end_) {
        batch[req_ids[row]].first += row_hits;
        batch[req_ids[row]].second += row_misses;
      }
    }
    commit_requests(layer);
    commit(layer);
  }

  // skip_topk layers copy the source's current batch counts, including during
  // warmup. Stream ordering guarantees the source callback has finished first.
  void record_reused(int64_t layer, int64_t source) {
    std::lock_guard<std::mutex> guard(mutex_);
    counters_.at(layer).last_hits = counters_.at(source).last_hits;
    counters_.at(layer).last_misses = counters_.at(source).last_misses;
    if (on_request_end_) last_batches_.at(layer) = last_batches_.at(source);
    commit_requests(layer);
    commit(layer);
  }

  // Called on the compute stream after the last decode callback for this request.
  // Erase after output so completed requests do not grow memory without bound.
  void finish_request(int64_t id, const std::string& name) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto found = requests_.find(id);
    if (found == requests_.end()) return;
    uint64_t hits = 0, misses = 0;
    for (size_t layer = 0; layer < layers_.size(); ++layer) {
      const auto& c = found->second[layer];
      hits += c.hits;
      misses += c.misses;
      std::cout << "[SparseKVRequestHitRate] " << worker_ << " policy=" << policy_
                << " request_id=" << name << " layer=" << layers_[layer]
                << " updates=" << c.updates << " hits=" << c.hits << " misses=" << c.misses
                << " hit_rate=" << rate(c.hits, c.misses) << "%" << std::endl;
    }
    std::cout << "[SparseKVRequestHitRate] " << worker_ << " policy=" << policy_
              << " request_id=" << name << " layer=ALL hits=" << hits << " misses=" << misses
              << " hit_rate=" << rate(hits, misses) << "%" << std::endl;
    requests_.erase(found);
    for (auto& batch : last_batches_) batch.erase(id);
  }

  std::vector<std::array<uint64_t, 3>> snapshot() {
    std::lock_guard<std::mutex> guard(mutex_);
    std::vector<std::array<uint64_t, 3>> result;
    for (const auto& c : counters_) result.push_back({c.updates, c.hits, c.misses});
    return result;
  }

  void reset() {
    std::lock_guard<std::mutex> guard(mutex_);
    std::fill(counters_.begin(), counters_.end(), Counter{});
    requests_.clear();
    for (auto& batch : last_batches_) batch.clear();
  }

 private:
  struct Counter {
    uint64_t updates = 0, hits = 0, misses = 0, last_hits = 0, last_misses = 0;
  };
  bool valid(int32_t token) const { return token >= 0 && static_cast<size_t>(token) < marks_.size(); }

  static double rate(uint64_t hits, uint64_t misses) {
    return hits + misses ? 100.0 * hits / (hits + misses) : 0.0;
  }

  void commit_requests(int64_t layer) {
    if (!on_request_end_) return;
    for (const auto& entry : last_batches_[layer]) {
      auto found = requests_.try_emplace(entry.first, layers_.size()).first;
      auto& counter = found->second[layer];
      ++counter.updates;
      if (counter.updates > static_cast<uint64_t>(warmup_)) {
        counter.hits += entry.second.first;
        counter.misses += entry.second.second;
      }
    }
  }

  void commit(int64_t layer) {
    auto& c = counters_.at(layer);
    ++c.updates;
    if (c.updates <= static_cast<uint64_t>(warmup_)) return;
    c.hits += c.last_hits;
    c.misses += c.last_misses;
    if (interval_ == 0 || (c.updates - warmup_) % interval_ != 0) return;
    uint64_t hits = 0, misses = 0;
    for (const auto& entry : counters_) {
      hits += entry.hits;
      misses += entry.misses;
    }
    // Overall is access-weighted, not an arithmetic mean of layer percentages.
    std::cout << "[SparseKVHitRate] " << worker_ << " policy=" << policy_ << " layer=" << layers_[layer]
              << " updates=" << c.updates << " hits=" << c.hits << " misses=" << c.misses
              << " hit_rate=" << rate(c.hits, c.misses) << "% overall_hits=" << hits
              << " overall_misses=" << misses << " overall_hit_rate=" << rate(hits, misses) << "%" << std::endl;
  }

  std::vector<std::string> layers_;
  std::string policy_, worker_;
  int64_t interval_, warmup_;
  bool on_request_end_;
  uint64_t epoch_ = 0;
  std::vector<uint64_t> marks_;
  std::vector<Counter> counters_;
  using Batch = std::unordered_map<int64_t, std::pair<uint64_t, uint64_t>>;
  std::vector<Batch> last_batches_;
  std::unordered_map<int64_t, std::vector<Counter>> requests_;
  std::mutex mutex_;
};
