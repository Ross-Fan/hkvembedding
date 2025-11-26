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
        // 检查CUDA设备
        int device_count;
        cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
        if (cuda_status != cudaSuccess || device_count == 0) {
            throw std::runtime_error("No CUDA devices found");
        }
        
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
        // 创建CUDA流
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        // 分配GPU内存
        K* d_keys;
        V* d_vectors;
        S* d_scores = nullptr;  // 不使用scores
        
        cudaMalloc(&d_keys, batch_size * sizeof(K));
        cudaMalloc(&d_vectors, batch_size * embedding_dim_ * sizeof(V));
        
        // 复制keys到GPU
        cudaMemcpy(d_keys, keys.data(), batch_size * sizeof(K), cudaMemcpyHostToDevice);
        
        // 调用HKV的find_or_insert - 注意：不需要found参数！
        table_->find_or_insert(batch_size, d_keys, d_vectors, d_scores, stream);
        
        // 同步流
        cudaStreamSynchronize(stream);
        
        // 复制结果回CPU
        cudaMemcpy(embeddings.data(), d_vectors, 
                   batch_size * embedding_dim_ * sizeof(V), cudaMemcpyDeviceToHost);
        
        // 清理GPU内存
        cudaFree(d_keys);
        cudaFree(d_vectors);
        cudaStreamDestroy(stream);
        
        // find_or_insert总是成功的，所以所有的found_flags都是true
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
        // 创建CUDA流
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        // 分配GPU内存
        K* d_keys;
        V* d_vectors;
        S* d_scores = nullptr;  // 不使用scores
        
        cudaMalloc(&d_keys, batch_size * sizeof(K));
        cudaMalloc(&d_vectors, batch_size * embedding_dim_ * sizeof(V));
        
        // 复制数据到GPU
        cudaMemcpy(d_keys, keys.data(), batch_size * sizeof(K), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vectors, values.data(), 
                   batch_size * embedding_dim_ * sizeof(V), cudaMemcpyHostToDevice);
        
        // 调用HKV的insert_or_assign
        table_->insert_or_assign(batch_size, d_keys, d_vectors, d_scores, stream);
        
        // 同步流
        cudaStreamSynchronize(stream);
        
        // 清理GPU内存
        cudaFree(d_keys);
        cudaFree(d_vectors);
        cudaStreamDestroy(stream);
        
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
        // 创建CUDA流
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        // 分配GPU内存
        K* d_keys;
        V* d_vectors;
        bool* d_found;
        S* d_scores = nullptr;
        
        cudaMalloc(&d_keys, batch_size * sizeof(K));
        cudaMalloc(&d_vectors, batch_size * embedding_dim_ * sizeof(V));
        cudaMalloc(&d_found, batch_size * sizeof(bool));
        
        // 复制keys到GPU
        cudaMemcpy(d_keys, keys.data(), batch_size * sizeof(K), cudaMemcpyHostToDevice);
        
        // 调用HKV的find
        table_->find(batch_size, d_keys, d_vectors, d_found, d_scores, stream);
        
        // 同步流
        cudaStreamSynchronize(stream);
        
        // 复制embeddings结果回CPU
        cudaMemcpy(embeddings.data(), d_vectors, 
                   batch_size * embedding_dim_ * sizeof(V), cudaMemcpyDeviceToHost);
        
        // 处理bool数组 - 使用临时缓冲区避免std::vector<bool>的问题
        std::vector<char> temp_found(batch_size);
        cudaMemcpy(temp_found.data(), d_found, 
                   batch_size * sizeof(bool), cudaMemcpyDeviceToHost);
        
        // 转换到std::vector<bool>
        for (size_t i = 0; i < batch_size; ++i) {
            found_flags[i] = static_cast<bool>(temp_found[i]);
        }
        
        // 清理GPU内存
        cudaFree(d_keys);
        cudaFree(d_vectors);
        cudaFree(d_found);
        cudaStreamDestroy(stream);
        
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
        // 创建CUDA流
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        // 分配GPU内存
        K* d_keys;
        V* d_vectors;
        S* d_scores = nullptr;
        
        cudaMalloc(&d_keys, batch_size * sizeof(K));
        cudaMalloc(&d_vectors, batch_size * embedding_dim_ * sizeof(V));
        
        // 复制数据到GPU
        cudaMemcpy(d_keys, keys, batch_size * sizeof(K), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vectors, values, 
                   batch_size * embedding_dim_ * sizeof(V), cudaMemcpyHostToDevice);
        
        // 调用HKV的insert_or_assign
        table_->insert_or_assign(batch_size, d_keys, d_vectors, d_scores, stream);
        
        // 同步流
        cudaStreamSynchronize(stream);
        
        // 清理GPU内存
        cudaFree(d_keys);
        cudaFree(d_vectors);
        cudaStreamDestroy(stream);
        
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
        // 创建CUDA流
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        // 分配GPU内存
        K* d_keys;
        V* d_vectors;
        S* d_scores = nullptr;
        
        cudaMalloc(&d_keys, batch_size * sizeof(K));
        cudaMalloc(&d_vectors, batch_size * embedding_dim_ * sizeof(V));
        
        // 复制keys到GPU
        cudaMemcpy(d_keys, keys, batch_size * sizeof(K), cudaMemcpyHostToDevice);
        
        // 调用HKV的find_or_insert
        table_->find_or_insert(batch_size, d_keys, d_vectors, d_scores, stream);
        
        // 同步流
        cudaStreamSynchronize(stream);
        
        // 复制结果回CPU
        cudaMemcpy(embeddings.data(), d_vectors, 
                   batch_size * embedding_dim_ * sizeof(V), cudaMemcpyDeviceToHost);
        
        // 清理GPU内存
        cudaFree(d_keys);
        cudaFree(d_vectors);
        cudaStreamDestroy(stream);
        
        // find_or_insert总是成功的
        std::vector<bool> found_flags(batch_size, true);
        
        return {embeddings, found_flags};
        
    } catch (const std::exception& e) {
        throw std::runtime_error("batch_find_or_insert failed: " + std::string(e.what()));
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
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    table_->reserve(capacity, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

template<typename K, typename V, typename S>
void HashTableWrapper<K, V, S>::clear() {
    ensure_initialized();
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    table_->clear(stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

// 显式实例化 - 只保留HKV支持的类型组合
// Key必须是int64_t或uint64_t，Score必须是uint64_t
template class HashTableWrapper<uint64_t, float, uint64_t>;
template class HashTableWrapper<int64_t, float, uint64_t>;
// 移除不支持的uint32_t组合
// template class HashTableWrapper<uint32_t, float, uint32_t>;  // 这行导致错误

} // namespace hkv_wrapper
