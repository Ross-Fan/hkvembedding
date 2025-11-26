#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include "hkv_wrapper/hkv_wrapper.hpp"

namespace py = pybind11;
using namespace hkv_wrapper;

// 辅助函数：将numpy数组转换为vector
template<typename T>
std::vector<T> numpy_to_vector(py::array_t<T> input) {
    py::buffer_info buf_info = input.request();
    T* ptr = static_cast<T*>(buf_info.ptr);
    return std::vector<T>(ptr, ptr + buf_info.size);
}

// 辅助函数：创建numpy数组的副本（避免内存问题）
template<typename T>
py::array_t<T> vector_to_numpy(const std::vector<T>& vec, const std::vector<ssize_t>& shape) {
    auto result = py::array_t<T>(shape);
    py::buffer_info buf = result.request();
    T* ptr = static_cast<T*>(buf.ptr);
    std::copy(vec.begin(), vec.end(), ptr);
    return result;
}

PYBIND11_MODULE(hkv_core, m) {
    m.doc() = "HierarchicalKV Python bindings for high-performance embedding tables";
    
    // 绑定主要的HashTable类 - 使用uint64_t
    py::class_<HashTableWrapper<uint64_t, float, uint64_t>>(m, "HashTable")
        .def(py::init<size_t, size_t, size_t, size_t>(),
             py::arg("init_capacity"), py::arg("max_capacity"), 
             py::arg("embedding_dim"), py::arg("max_hbm_gb") = 16,
             "Initialize HashTable with specified capacities and embedding dimension")
        
        // 核心查找和插入方法
        .def("find_or_insert", 
             [](HashTableWrapper<uint64_t, float, uint64_t>& self, py::array_t<uint64_t> keys) {
                 auto keys_vec = numpy_to_vector(keys);
                 auto [embeddings, found_flags] = self.find_or_insert(keys_vec);
                 
                 // 转换为numpy数组
                 auto emb_shape = std::vector<ssize_t>{static_cast<ssize_t>(keys_vec.size()), 
                                                      static_cast<ssize_t>(self.embedding_dim())};
                 auto emb_array = vector_to_numpy(embeddings, emb_shape);
                 auto flags_shape = std::vector<ssize_t>{static_cast<ssize_t>(found_flags.size())};
                 auto flags_array = vector_to_numpy(found_flags, flags_shape);
                 
                 return py::make_tuple(emb_array, flags_array);
             },
             py::arg("keys"), "Find or insert embeddings for given keys")
        
        .def("insert_or_assign", 
             [](HashTableWrapper<uint64_t, float, uint64_t>& self, 
                py::array_t<uint64_t> keys, py::array_t<float> values) {
                 auto keys_vec = numpy_to_vector(keys);
                 auto values_vec = numpy_to_vector(values);
                 self.insert_or_assign(keys_vec, values_vec);
             },
             py::arg("keys"), py::arg("values"), "Insert or assign embeddings for given keys")
        
        .def("find", 
             [](HashTableWrapper<uint64_t, float, uint64_t>& self, py::array_t<uint64_t> keys) {
                 auto keys_vec = numpy_to_vector(keys);
                 auto [embeddings, found_flags] = self.find(keys_vec);
                 
                 auto emb_shape = std::vector<ssize_t>{static_cast<ssize_t>(keys_vec.size()), 
                                                      static_cast<ssize_t>(self.embedding_dim())};
                 auto emb_array = vector_to_numpy(embeddings, emb_shape);
                 auto flags_shape = std::vector<ssize_t>{static_cast<ssize_t>(found_flags.size())};
                 auto flags_array = vector_to_numpy(found_flags, flags_shape);
                 
                 return py::make_tuple(emb_array, flags_array);
             },
             py::arg("keys"), "Find embeddings for given keys")
        
        // 统计信息方法
        .def("size", &HashTableWrapper<uint64_t, float, uint64_t>::size,
             "Get current number of entries in the table")
        .def("capacity", &HashTableWrapper<uint64_t, float, uint64_t>::capacity,
             "Get maximum capacity of the table")
        .def("load_factor", &HashTableWrapper<uint64_t, float, uint64_t>::load_factor,
             "Get current load factor (size/capacity)")
        .def("embedding_dim", &HashTableWrapper<uint64_t, float, uint64_t>::embedding_dim,
             "Get embedding dimension")
        
        // 内存管理方法
        .def("reserve", &HashTableWrapper<uint64_t, float, uint64_t>::reserve,
             py::arg("capacity"), "Reserve capacity (if supported)")
        .def("clear", &HashTableWrapper<uint64_t, float, uint64_t>::clear,
             "Clear all entries (if supported)")
        
        // 属性
        .def_property_readonly("embedding_dim", 
                              &HashTableWrapper<uint64_t, float, uint64_t>::embedding_dim)
        
        // 字符串表示
        .def("__repr__", [](const HashTableWrapper<uint64_t, float, uint64_t>& self) {
            return "<HKV HashTable: size=" + std::to_string(self.size()) + 
                   ", capacity=" + std::to_string(self.capacity()) + 
                   ", embedding_dim=" + std::to_string(self.embedding_dim()) + ">";
        });
    
    // 可选：绑定int64_t版本
    py::class_<HashTableWrapper<int64_t, float, uint64_t>>(m, "Int64HashTable")
        .def(py::init<size_t, size_t, size_t, size_t>(),
             py::arg("init_capacity"), py::arg("max_capacity"), 
             py::arg("embedding_dim"), py::arg("max_hbm_gb") = 16,
             "Initialize HashTable with int64_t keys");
    
    // 添加一些实用函数
    m.def("version", []() { return "1.0.0"; }, "Get version string");
    
    // 添加异常处理
    py::register_exception<std::runtime_error>(m, "HKVRuntimeError");
    py::register_exception<std::invalid_argument>(m, "HKVInvalidArgument");
}
