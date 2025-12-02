#pragma once

#include "merlin_hashtable.cuh"
#include <memory>
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>

namespace hkv_wrapper {

using TableOptions = nv::merlin::HashTableOptions;
using EvictStrategy = nv::merlin::EvictStrategy;

/**
 * CUDA Stream Pool for reusing streams across operations.
 * Avoids the overhead of creating/destroying streams per operation.
 */
class CudaStreamPool {
public:
    CudaStreamPool(size_t pool_size = 4) : pool_size_(pool_size), current_idx_(0) {
        streams_.resize(pool_size);
        for (size_t i = 0; i < pool_size; ++i) {
            cudaStreamCreate(&streams_[i]);
        }
    }
    
    ~CudaStreamPool() {
        for (auto& stream : streams_) {
            cudaStreamDestroy(stream);
        }
    }
    
    cudaStream_t get() {
        cudaStream_t stream = streams_[current_idx_];
        current_idx_ = (current_idx_ + 1) % pool_size_;
        return stream;
    }
    
    void synchronize_all() {
        for (auto& stream : streams_) {
            cudaStreamSynchronize(stream);
        }
    }

private:
    std::vector<cudaStream_t> streams_;
    size_t pool_size_;
    size_t current_idx_;
};

/**
 * GPU Memory Pool for reusing GPU memory allocations.
 * Reduces cudaMalloc/cudaFree overhead for repeated operations.
 */
template<typename T>
class GpuMemoryPool {
public:
    GpuMemoryPool(size_t max_batch_size, size_t dim) 
        : max_batch_size_(max_batch_size), dim_(dim), allocated_(false) {}
    
    ~GpuMemoryPool() {
        free();
    }
    
    void ensure_allocated() {
        if (!allocated_) {
            allocate();
        }
    }
    
    T* get_buffer(size_t size) {
        if (size > max_batch_size_ * dim_) {
            throw std::runtime_error("Requested size exceeds pool capacity");
        }
        ensure_allocated();
        return buffer_;
    }
    
    void free() {
        if (allocated_) {
            cudaFree(buffer_);
            allocated_ = false;
        }
    }

private:
    void allocate() {
        cudaMalloc(&buffer_, max_batch_size_ * dim_ * sizeof(T));
        allocated_ = true;
    }
    
    T* buffer_;
    size_t max_batch_size_;
    size_t dim_;
    bool allocated_;
};

template<typename K = uint64_t, typename V = float, typename S = uint64_t>
class HashTableWrapper {
public:
    using HKVTable = nv::merlin::HashTable<K, V, S, EvictStrategy::kLru>;
    
    HashTableWrapper(size_t init_capacity, size_t max_capacity, 
                    size_t embedding_dim, size_t max_hbm_gb = 16);
    
    ~HashTableWrapper() = default;
    
    // Core interface
    std::pair<std::vector<V>, std::vector<bool>> 
    find_or_insert(const std::vector<K>& keys);
    
    void insert_or_assign(const std::vector<K>& keys, 
                         const std::vector<V>& values);
    
    std::pair<std::vector<V>, std::vector<bool>> 
    find(const std::vector<K>& keys);
    
    // Batch operation interface (raw pointers)
    void batch_insert_or_assign(const K* keys, const V* values, size_t batch_size);
    std::pair<std::vector<V>, std::vector<bool>> 
    batch_find_or_insert(const K* keys, size_t batch_size);
    
    // Gradient-aware operations (for optimizer integration)
    void apply_gradients(const std::vector<K>& keys, 
                        const std::vector<V>& gradients,
                        float learning_rate);
    
    // Bulk export
    std::pair<std::vector<K>, std::vector<V>> export_all(size_t max_count = 0);

    
    std::vector<K> HashTableWrapper<K, V, S>::export_keys();
    
    // Statistics
    size_t size() const;
    size_t capacity() const;
    double load_factor() const;
    size_t embedding_dim() const { return embedding_dim_; }
    
    // Memory management
    void reserve(size_t capacity);
    void clear();
    
    // Stream management
    void synchronize() {
        if (stream_pool_) {
            stream_pool_->synchronize_all();
        }
    }
    
private:
    std::unique_ptr<HKVTable> table_;
    size_t embedding_dim_;
    bool initialized_;
    
    // Optimized CUDA resource management
    std::unique_ptr<CudaStreamPool> stream_pool_;
    
    void ensure_initialized() const {
        if (!initialized_) {
            throw std::runtime_error("HashTable not initialized");
        }
    }
};

// Type aliases - only HKV supported types
using DefaultHashTable = HashTableWrapper<uint64_t, float, uint64_t>;
using Int64HashTable = HashTableWrapper<int64_t, float, uint64_t>;

} // namespace hkv_wrapper

// Explicit template instantiation declarations
extern template class hkv_wrapper::HashTableWrapper<uint64_t, float, uint64_t>;
extern template class hkv_wrapper::HashTableWrapper<int64_t, float, uint64_t>;
