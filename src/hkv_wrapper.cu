#include "hkv_wrapper/hkv_wrapper.hpp"
#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>

namespace hkv_wrapper {

template<typename K, typename V, typename S>
HashTableWrapper<K, V, S>::HashTableWrapper(
    size_t init_capacity, size_t max_capacity, 
    size_t embedding_dim, size_t max_hbm_gb) 
    : embedding_dim_(embedding_dim), initialized_(false) {
    
    try {
        // Check CUDA device
        int device_count;
        cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
        if (cuda_status != cudaSuccess || device_count == 0) {
            throw std::runtime_error("No CUDA devices found");
        }
        
        // Initialize stream pool for reusing CUDA streams
        stream_pool_ = std::make_unique<CudaStreamPool>(4);
        
        table_ = std::make_unique<HKVTable>();
        
        TableOptions options;
        options.init_capacity = init_capacity;
        options.max_capacity = max_capacity;
        options.dim = embedding_dim;
        options.max_hbm_for_vectors = nv::merlin::GB(max_hbm_gb);
        
        table_->init(options);
        initialized_ = true;
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to initialize HashTable: " + std::string(e.what()));
    }
}

template<typename K, typename V, typename S>
std::pair<std::vector<V>, std::vector<bool>> 
HashTableWrapper<K, V, S>::find_or_insert(const std::vector<K>& keys) {
    ensure_initialized();
    
    size_t batch_size = keys.size();
    if (batch_size == 0) {
        return {std::vector<V>(), std::vector<bool>()};
    }
    
    std::vector<V> embeddings(batch_size * embedding_dim_);
    
    try {
        // Get stream from pool (reuse streams)
        cudaStream_t stream = stream_pool_->get();
        
        // Allocate GPU memory
        K* d_keys;
        V* d_vectors;
        S* d_scores = nullptr;
        
        cudaMalloc(&d_keys, batch_size * sizeof(K));
        cudaMalloc(&d_vectors, batch_size * embedding_dim_ * sizeof(V));
        
        // Copy keys to GPU
        cudaMemcpyAsync(d_keys, keys.data(), batch_size * sizeof(K), 
                       cudaMemcpyHostToDevice, stream);
        
        // Call HKV's find_or_insert
        table_->find_or_insert(batch_size, d_keys, d_vectors, d_scores, stream);
        
        // Copy results back to CPU
        cudaMemcpyAsync(embeddings.data(), d_vectors, 
                       batch_size * embedding_dim_ * sizeof(V), 
                       cudaMemcpyDeviceToHost, stream);
        
        // Synchronize
        cudaStreamSynchronize(stream);
        
        // Cleanup GPU memory
        cudaFree(d_keys);
        cudaFree(d_vectors);
        
        // find_or_insert always succeeds
        std::vector<bool> found_flags(batch_size, true);
        
        return {embeddings, found_flags};
        
    } catch (const std::exception& e) {
        throw std::runtime_error("find_or_insert failed: " + std::string(e.what()));
    }
}

template<typename K, typename V, typename S>
void HashTableWrapper<K, V, S>::insert_or_assign(
    const std::vector<K>& keys, const std::vector<V>& values) {
    
    ensure_initialized();
    
    size_t batch_size = keys.size();
    if (batch_size == 0) return;
    
    if (values.size() != batch_size * embedding_dim_) {
        throw std::invalid_argument("Values size mismatch");
    }
    
    try {
        cudaStream_t stream = stream_pool_->get();
        
        K* d_keys;
        V* d_vectors;
        S* d_scores = nullptr;
        
        cudaMalloc(&d_keys, batch_size * sizeof(K));
        cudaMalloc(&d_vectors, batch_size * embedding_dim_ * sizeof(V));
        
        // Async copies
        cudaMemcpyAsync(d_keys, keys.data(), batch_size * sizeof(K), 
                       cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_vectors, values.data(), 
                       batch_size * embedding_dim_ * sizeof(V), 
                       cudaMemcpyHostToDevice, stream);
        
        table_->insert_or_assign(batch_size, d_keys, d_vectors, d_scores, stream);
        
        cudaStreamSynchronize(stream);
        
        cudaFree(d_keys);
        cudaFree(d_vectors);
        
    } catch (const std::exception& e) {
        throw std::runtime_error("insert_or_assign failed: " + std::string(e.what()));
    }
}

template<typename K, typename V, typename S>
std::pair<std::vector<V>, std::vector<bool>> 
HashTableWrapper<K, V, S>::find(const std::vector<K>& keys) {
    ensure_initialized();
    
    size_t batch_size = keys.size();
    if (batch_size == 0) {
        return {std::vector<V>(), std::vector<bool>()};
    }
    
    std::vector<V> embeddings(batch_size * embedding_dim_);
    std::vector<bool> found_flags(batch_size);
    
    try {
        cudaStream_t stream = stream_pool_->get();
        
        K* d_keys;
        V* d_vectors;
        bool* d_found;
        S* d_scores = nullptr;
        
        cudaMalloc(&d_keys, batch_size * sizeof(K));
        cudaMalloc(&d_vectors, batch_size * embedding_dim_ * sizeof(V));
        cudaMalloc(&d_found, batch_size * sizeof(bool));
        
        cudaMemcpyAsync(d_keys, keys.data(), batch_size * sizeof(K), 
                       cudaMemcpyHostToDevice, stream);
        
        table_->find(batch_size, d_keys, d_vectors, d_found, d_scores, stream);
        
        cudaMemcpyAsync(embeddings.data(), d_vectors, 
                       batch_size * embedding_dim_ * sizeof(V), 
                       cudaMemcpyDeviceToHost, stream);
        
        // Handle bool array
        std::vector<char> temp_found(batch_size);
        cudaMemcpyAsync(temp_found.data(), d_found, 
                       batch_size * sizeof(bool), 
                       cudaMemcpyDeviceToHost, stream);
        
        cudaStreamSynchronize(stream);
        
        // Convert to std::vector<bool>
        for (size_t i = 0; i < batch_size; ++i) {
            found_flags[i] = static_cast<bool>(temp_found[i]);
        }
        
        cudaFree(d_keys);
        cudaFree(d_vectors);
        cudaFree(d_found);
        
        return {embeddings, found_flags};
        
    } catch (const std::exception& e) {
        throw std::runtime_error("find failed: " + std::string(e.what()));
    }
}

template<typename K, typename V, typename S>
void HashTableWrapper<K, V, S>::batch_insert_or_assign(
    const K* keys, const V* values, size_t batch_size) {
    
    ensure_initialized();
    
    if (batch_size == 0) return;
    
    try {
        cudaStream_t stream = stream_pool_->get();
        
        K* d_keys;
        V* d_vectors;
        S* d_scores = nullptr;
        
        cudaMalloc(&d_keys, batch_size * sizeof(K));
        cudaMalloc(&d_vectors, batch_size * embedding_dim_ * sizeof(V));
        
        cudaMemcpyAsync(d_keys, keys, batch_size * sizeof(K), 
                       cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_vectors, values, 
                       batch_size * embedding_dim_ * sizeof(V), 
                       cudaMemcpyHostToDevice, stream);
        
        table_->insert_or_assign(batch_size, d_keys, d_vectors, d_scores, stream);
        
        cudaStreamSynchronize(stream);
        
        cudaFree(d_keys);
        cudaFree(d_vectors);
        
    } catch (const std::exception& e) {
        throw std::runtime_error("batch_insert_or_assign failed: " + std::string(e.what()));
    }
}

template<typename K, typename V, typename S>
std::pair<std::vector<V>, std::vector<bool>> 
HashTableWrapper<K, V, S>::batch_find_or_insert(const K* keys, size_t batch_size) {
    ensure_initialized();
    
    if (batch_size == 0) {
        return {std::vector<V>(), std::vector<bool>()};
    }
    
    std::vector<V> embeddings(batch_size * embedding_dim_);
    
    try {
        cudaStream_t stream = stream_pool_->get();
        
        K* d_keys;
        V* d_vectors;
        S* d_scores = nullptr;
        
        cudaMalloc(&d_keys, batch_size * sizeof(K));
        cudaMalloc(&d_vectors, batch_size * embedding_dim_ * sizeof(V));
        
        cudaMemcpyAsync(d_keys, keys, batch_size * sizeof(K), 
                       cudaMemcpyHostToDevice, stream);
        
        table_->find_or_insert(batch_size, d_keys, d_vectors, d_scores, stream);
        
        cudaMemcpyAsync(embeddings.data(), d_vectors, 
                       batch_size * embedding_dim_ * sizeof(V), 
                       cudaMemcpyDeviceToHost, stream);
        
        cudaStreamSynchronize(stream);
        
        cudaFree(d_keys);
        cudaFree(d_vectors);
        
        std::vector<bool> found_flags(batch_size, true);
        
        return {embeddings, found_flags};
        
    } catch (const std::exception& e) {
        throw std::runtime_error("batch_find_or_insert failed: " + std::string(e.what()));
    }
}

template<typename K, typename V, typename S>
void HashTableWrapper<K, V, S>::apply_gradients(
    const std::vector<K>& keys, 
    const std::vector<V>& gradients,
    float learning_rate) {
    
    ensure_initialized();
    
    size_t batch_size = keys.size();
    if (batch_size == 0) return;
    
    if (gradients.size() != batch_size * embedding_dim_) {
        throw std::invalid_argument("Gradients size mismatch");
    }
    
    try {
        cudaStream_t stream = stream_pool_->get();
        
        // First, find current embeddings
        K* d_keys;
        V* d_vectors;
        V* d_grads;
        S* d_scores = nullptr;
        
        cudaMalloc(&d_keys, batch_size * sizeof(K));
        cudaMalloc(&d_vectors, batch_size * embedding_dim_ * sizeof(V));
        cudaMalloc(&d_grads, batch_size * embedding_dim_ * sizeof(V));
        
        cudaMemcpyAsync(d_keys, keys.data(), batch_size * sizeof(K), 
                       cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_grads, gradients.data(), 
                       batch_size * embedding_dim_ * sizeof(V), 
                       cudaMemcpyHostToDevice, stream);
        
        // Get current embeddings
        table_->find_or_insert(batch_size, d_keys, d_vectors, d_scores, stream);
        
        // Apply gradient update on GPU (could use a custom kernel, but using CPU for simplicity)
        cudaStreamSynchronize(stream);
        
        std::vector<V> current_embeddings(batch_size * embedding_dim_);
        cudaMemcpy(current_embeddings.data(), d_vectors, 
                  batch_size * embedding_dim_ * sizeof(V), cudaMemcpyDeviceToHost);
        
        // Update: embedding = embedding - lr * gradient
        for (size_t i = 0; i < current_embeddings.size(); ++i) {
            current_embeddings[i] -= learning_rate * gradients[i];
        }
        
        // Write back
        cudaMemcpyAsync(d_vectors, current_embeddings.data(), 
                       batch_size * embedding_dim_ * sizeof(V), 
                       cudaMemcpyHostToDevice, stream);
        
        table_->insert_or_assign(batch_size, d_keys, d_vectors, d_scores, stream);
        
        cudaStreamSynchronize(stream);
        
        cudaFree(d_keys);
        cudaFree(d_vectors);
        cudaFree(d_grads);
        
    } catch (const std::exception& e) {
        throw std::runtime_error("apply_gradients failed: " + std::string(e.what()));
    }
}

template<typename K, typename V, typename S>
std::pair<std::vector<K>, std::vector<V>> 
HashTableWrapper<K, V, S>::export_all(size_t max_count) {
    ensure_initialized();
    
    cudaStream_t stream = stream_pool_->get();
    
    size_t table_size = table_->size(stream);
    cudaStreamSynchronize(stream);
    
    if (table_size == 0) {
        return {std::vector<K>(), std::vector<V>()};
    }
    
    size_t export_count = (max_count > 0 && max_count < table_size) ? max_count : table_size;
    
    std::vector<K> keys(export_count);
    std::vector<V> values(export_count * embedding_dim_);
    
    try {
        K* d_keys;
        V* d_vectors;
        S* d_scores = nullptr;
        
        cudaMalloc(&d_keys, export_count * sizeof(K));
        cudaMalloc(&d_vectors, export_count * embedding_dim_ * sizeof(V));
        
        // Export using HKV's export_batch
        size_t exported = 0;
        table_->export_batch(table_->capacity(), 0, d_keys, d_vectors, d_scores, stream);
        
        cudaMemcpyAsync(keys.data(), d_keys, export_count * sizeof(K), 
                       cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(values.data(), d_vectors, 
                       export_count * embedding_dim_ * sizeof(V), 
                       cudaMemcpyDeviceToHost, stream);
        
        cudaStreamSynchronize(stream);
        
        cudaFree(d_keys);
        cudaFree(d_vectors);
        
        return {keys, values};
        
    } catch (const std::exception& e) {
        throw std::runtime_error("export_all failed: " + std::string(e.what()));
    }
}

template<typename K, typename V, typename S>
std::vector<K> HashTableWrapper<K, V, S>::export_keys() {
    ensure_initialized();
    
    try {
        // Get table size
        size_t table_size = table_->size();
        if (table_size == 0) {
            return std::vector<K>();
        }
        
        // Allocate buffers
        std::vector<K> keys(table_size);
        K* d_keys;
        cudaMalloc(&d_keys, table_size * sizeof(K));
        
        cudaStream_t stream = stream_pool_->get();
        
        // Export keys from HKV table
        size_t actual_size = 0;
        table_->export_batch(table_size, 0, d_keys, nullptr, nullptr, &actual_size, stream);
        
        // Copy back to host
        cudaMemcpyAsync(keys.data(), d_keys, actual_size * sizeof(K), 
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        cudaFree(d_keys);
        
        keys.resize(actual_size);
        return keys;
        
    } catch (const std::exception& e) {
        throw std::runtime_error("export_keys failed: " + std::string(e.what()));
    }
}


template<typename K, typename V, typename S>
size_t HashTableWrapper<K, V, S>::size() const {
    ensure_initialized();
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t result = table_->size(stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    
    return result;
}

template<typename K, typename V, typename S>
size_t HashTableWrapper<K, V, S>::capacity() const {
    ensure_initialized();
    return table_->capacity();
}

template<typename K, typename V, typename S>
double HashTableWrapper<K, V, S>::load_factor() const {
    ensure_initialized();
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    double result = table_->load_factor(stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    
    return result;
}

template<typename K, typename V, typename S>
void HashTableWrapper<K, V, S>::reserve(size_t capacity) {
    ensure_initialized();
    
    cudaStream_t stream = stream_pool_->get();
    table_->reserve(capacity, stream);
    cudaStreamSynchronize(stream);
}

template<typename K, typename V, typename S>
void HashTableWrapper<K, V, S>::clear() {
    ensure_initialized();
    
    cudaStream_t stream = stream_pool_->get();
    table_->clear(stream);
    cudaStreamSynchronize(stream);
}

// Explicit template instantiation
template class HashTableWrapper<uint64_t, float, uint64_t>;
template class HashTableWrapper<int64_t, float, uint64_t>;

} // namespace hkv_wrapper
