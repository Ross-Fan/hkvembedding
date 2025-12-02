#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include "hkv_wrapper/hkv_wrapper.hpp"

namespace py = pybind11;
using namespace hkv_wrapper;

// Helper: numpy array to vector
template<typename T>
std::vector<T> numpy_to_vector(py::array_t<T> input) {
    py::buffer_info buf_info = input.request();
    T* ptr = static_cast<T*>(buf_info.ptr);
    return std::vector<T>(ptr, ptr + buf_info.size);
}

// Helper: vector to numpy array (copy to avoid memory issues)
template<typename T>
py::array_t<T> vector_to_numpy(const std::vector<T>& vec, const std::vector<ssize_t>& shape) {
    auto result = py::array_t<T>(shape);
    py::buffer_info buf = result.request();
    T* ptr = static_cast<T*>(buf.ptr);
    std::copy(vec.begin(), vec.end(), ptr);
    return result;
}

// Define the HashTable binding for a specific key type
template<typename K>
void bind_hashtable(py::module& m, const char* name) {
    using Wrapper = HashTableWrapper<K, float, uint64_t>;
    
    py::class_<Wrapper>(m, name)
        .def(py::init<size_t, size_t, size_t, size_t>(),
             py::arg("init_capacity"), py::arg("max_capacity"), 
             py::arg("embedding_dim"), py::arg("max_hbm_gb") = 16,
             "Initialize HashTable with specified capacities and embedding dimension")
        
        // Core find and insert methods
        .def("find_or_insert", 
             [](Wrapper& self, py::array_t<K> keys) {
                 auto keys_vec = numpy_to_vector(keys);
                 auto [embeddings, found_flags] = self.find_or_insert(keys_vec);
                 
                 auto emb_shape = std::vector<ssize_t>{
                     static_cast<ssize_t>(keys_vec.size()), 
                     static_cast<ssize_t>(self.embedding_dim())
                 };
                 auto emb_array = vector_to_numpy(embeddings, emb_shape);
                 auto flags_shape = std::vector<ssize_t>{static_cast<ssize_t>(found_flags.size())};
                 auto flags_array = vector_to_numpy(found_flags, flags_shape);
                 
                 return py::make_tuple(emb_array, flags_array);
             },
             py::arg("keys"), 
             "Find or insert embeddings for given keys. Returns (embeddings, found_flags)")
        
        .def("insert_or_assign", 
             [](Wrapper& self, py::array_t<K> keys, py::array_t<float> values) {
                 auto keys_vec = numpy_to_vector(keys);
                 auto values_vec = numpy_to_vector(values);
                 self.insert_or_assign(keys_vec, values_vec);
             },
             py::arg("keys"), py::arg("values"), 
             "Insert or assign embeddings for given keys")
        
        .def("find", 
             [](Wrapper& self, py::array_t<K> keys) {
                 auto keys_vec = numpy_to_vector(keys);
                 auto [embeddings, found_flags] = self.find(keys_vec);
                 
                 auto emb_shape = std::vector<ssize_t>{
                     static_cast<ssize_t>(keys_vec.size()), 
                     static_cast<ssize_t>(self.embedding_dim())
                 };
                 auto emb_array = vector_to_numpy(embeddings, emb_shape);
                 auto flags_shape = std::vector<ssize_t>{static_cast<ssize_t>(found_flags.size())};
                 auto flags_array = vector_to_numpy(found_flags, flags_shape);
                 
                 return py::make_tuple(emb_array, flags_array);
             },
             py::arg("keys"), 
             "Find embeddings for given keys. Returns (embeddings, found_flags)")
        
        // Gradient application (for optimizer integration)
        .def("apply_gradients",
             [](Wrapper& self, py::array_t<K> keys, py::array_t<float> gradients, float lr) {
                 auto keys_vec = numpy_to_vector(keys);
                 auto grads_vec = numpy_to_vector(gradients);
                 self.apply_gradients(keys_vec, grads_vec, lr);
             },
             py::arg("keys"), py::arg("gradients"), py::arg("learning_rate"),
             "Apply gradient updates: embedding = embedding - lr * gradient")
        
        // Export functionality
        .def("export_all",
             [](Wrapper& self, size_t max_count) {
                 auto [keys, values] = self.export_all(max_count);
                 
                if (keys.empty()) {
                    // Construct shapes explicitly to avoid brace-init ambiguity
                    auto keys_arr = py::array_t<K>(0);
                    std::vector<ssize_t> values_shape = {
                        static_cast<ssize_t>(0),
                        static_cast<ssize_t>(self.embedding_dim())
                    };
                    auto values_arr = py::array_t<float>(values_shape);
                    return py::make_tuple(keys_arr, values_arr);
                }
                 
                 auto keys_shape = std::vector<ssize_t>{static_cast<ssize_t>(keys.size())};
                 auto values_shape = std::vector<ssize_t>{
                     static_cast<ssize_t>(keys.size()), 
                     static_cast<ssize_t>(self.embedding_dim())
                 };
                 
                 return py::make_tuple(
                     vector_to_numpy(keys, keys_shape),
                     vector_to_numpy(values, values_shape)
                 );
             },
             py::arg("max_count") = 0,
             "Export all key-value pairs. Returns (keys, embeddings)")
        .def("export_keys",
             [](Wrapper& self) {
                 return self.export_keys();
             },
             "Export all keys from the hash table")
        // Statistics methods
        .def("size", &Wrapper::size,
             "Get current number of entries in the table")
        .def("capacity", &Wrapper::capacity,
             "Get maximum capacity of the table")
        .def("load_factor", &Wrapper::load_factor,
             "Get current load factor (size/capacity)")
        .def("embedding_dim", &Wrapper::embedding_dim,
             "Get embedding dimension")
        
        // Memory management methods
        .def("reserve", &Wrapper::reserve,
             py::arg("capacity"), "Reserve capacity")
        .def("clear", &Wrapper::clear,
             "Clear all entries")
        .def("synchronize", &Wrapper::synchronize,
             "Synchronize all pending CUDA operations")
        
        // Properties
        .def_property_readonly("embedding_dim", &Wrapper::embedding_dim)
        
        // String representation
        .def("__repr__", [](Wrapper& self) {
            return "<HKV HashTable: size=" + std::to_string(self.size()) + 
                   ", capacity=" + std::to_string(self.capacity()) + 
                   ", embedding_dim=" + std::to_string(self.embedding_dim()) + ">";
        })
        
        // Context manager support for resource cleanup
        .def("__enter__", [](Wrapper& self) -> Wrapper& { return self; })
        .def("__exit__", [](Wrapper& self, py::object, py::object, py::object) {
            self.synchronize();
        });
}

PYBIND11_MODULE(hkv_core, m) {
    m.doc() = R"doc(
HierarchicalKV Python bindings for high-performance embedding tables.

This module provides GPU-accelerated hash tables optimized for 
recommendation system embeddings with billions of unique IDs.

Key Features:
- GPU-native storage on HBM (High Bandwidth Memory)
- LRU eviction for capacity management
- Optimized for sparse embedding lookups
- Thread-safe operations

Example:
    >>> import hkv_core
    >>> table = hkv_core.HashTable(1000, 100000, 64, 4)  # 64-dim, 4GB HBM
    >>> embeddings, found = table.find_or_insert(keys)
)doc";
    
    // Bind uint64_t key type (default)
    bind_hashtable<uint64_t>(m, "HashTable");
    
    // Bind int64_t key type
    bind_hashtable<int64_t>(m, "Int64HashTable");
    
    // Version function
    m.def("version", []() { return "1.1.0"; }, "Get version string");
    
    // Utility functions
    m.def("cuda_device_count", []() {
        int count;
        cudaGetDeviceCount(&count);
        return count;
    }, "Get number of available CUDA devices");
    
    m.def("cuda_memory_info", [](int device) {
        cudaSetDevice(device);
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        return py::make_tuple(free_mem, total_mem);
    }, py::arg("device") = 0, "Get (free, total) GPU memory in bytes");
    
    // Exception types
    py::register_exception<std::runtime_error>(m, "HKVRuntimeError");
    py::register_exception<std::invalid_argument>(m, "HKVInvalidArgument");
}
