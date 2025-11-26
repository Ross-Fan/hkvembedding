#pragma once

#include "merlin_hashtable.cuh"
#include <memory>
#include <vector>
#include <stdexcept>

namespace hkv_wrapper {

using TableOptions = nv::merlin::HashTableOptions;
using EvictStrategy = nv::merlin::EvictStrategy;

template<typename K = uint64_t, typename V = float, typename S = uint64_t>
class HashTableWrapper {
public:
    using HKVTable = nv::merlin::HashTable<K, V, S, EvictStrategy::kLru>;
    
    HashTableWrapper(size_t init_capacity, size_t max_capacity, 
                    size_t embedding_dim, size_t max_hbm_gb = 16);
    
    ~HashTableWrapper() = default;
    
    // 核心接口
    std::pair<std::vector<V>, std::vector<bool>> 
    find_or_insert(const std::vector<K>& keys);
    
    void insert_or_assign(const std::vector<K>& keys, 
                         const std::vector<V>& values);
    
    std::pair<std::vector<V>, std::vector<bool>> 
    find(const std::vector<K>& keys);
    
    // 批量操作接口
    void batch_insert_or_assign(const K* keys, const V* values, size_t batch_size);
    std::pair<std::vector<V>, std::vector<bool>> 
    batch_find_or_insert(const K* keys, size_t batch_size);
    
    // 统计信息
    size_t size() const;
    size_t capacity() const;
    double load_factor() const;
    size_t embedding_dim() const { return embedding_dim_; }
    
    // 内存管理
    void reserve(size_t capacity);
    void clear();
    
private:
    std::unique_ptr<HKVTable> table_;
    size_t embedding_dim_;
    bool initialized_;
    
    void ensure_initialized() const {
        if (!initialized_) {
            throw std::runtime_error("HashTable not initialized");
        }
    }
};

// 类型别名 - 只使用HKV支持的类型
using DefaultHashTable = HashTableWrapper<uint64_t, float, uint64_t>;
using Int64HashTable = HashTableWrapper<int64_t, float, uint64_t>;

} // namespace hkv_wrapper

// 显式声明模板实例化 - 只保留支持的类型
extern template class hkv_wrapper::HashTableWrapper<uint64_t, float, uint64_t>;
extern template class hkv_wrapper::HashTableWrapper<int64_t, float, uint64_t>;
// 移除不支持的类型声明
// extern template class hkv_wrapper::HashTableWrapper<uint32_t, float, uint32_t>;
