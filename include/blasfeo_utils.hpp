#pragma once

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <blasfeo/blasfeo_d_aux.h>
#include <blasfeo/blasfeo_d_blas.h>

class BlasfeoMatrixOps {
private:
    void *memory_dmat, *memory_dvec1, *memory_dvec2;
    struct blasfeo_dmat sA;
    struct blasfeo_dvec sx, sy;
    int rows, cols;
    bool initialized;
    bool matrix_packed;  // Track if matrix is already packed

public:
    BlasfeoMatrixOps() : memory_dmat(nullptr), memory_dvec1(nullptr),
                     memory_dvec2(nullptr), rows(0), cols(0),
                     initialized(false), matrix_packed(false) {}

    ~BlasfeoMatrixOps() {
        if (initialized) {
            free(memory_dvec2);
            free(memory_dvec1);
            free(memory_dmat);
        }
    }

    void initialize(int r, int c) {
        // Free previous memory if dimensions change
        if (initialized && (rows != r || cols != c)) {
            free(memory_dvec2);
            free(memory_dvec1);
            free(memory_dmat);
            initialized = false;
            matrix_packed = false;
        }

        if (!initialized) {
            rows = r;
            cols = c;

            // Calculate required memory sizes
            int dmat_size = blasfeo_memsize_dmat(rows, cols);
            int dvec_size = blasfeo_memsize_dvec(std::max(rows, cols));

            // Allocate memory
            memory_dmat = malloc(dmat_size);
            memory_dvec1 = malloc(dvec_size);
            memory_dvec2 = malloc(dvec_size);

            if (!memory_dmat || !memory_dvec1 || !memory_dvec2) {
                throw std::runtime_error("Memory allocation failed");
            }

            // Initialize structures
            blasfeo_create_dmat(rows, cols, &sA, memory_dmat);
            blasfeo_create_dvec(cols, &sx, memory_dvec1);
            blasfeo_create_dvec(rows, &sy, memory_dvec2);

            initialized = true;
        }
    }

    // Standard matrix-vector multiplication
    void matrix_vector_mult(const std::vector<double>& matrix,
                           const std::vector<double>& vector,
                           std::vector<double>& result) {
        if (!initialized) {
            throw std::runtime_error("BlasfeoMatrixOps not initialized");
        }

        // Pack data
        blasfeo_pack_dmat(rows, cols, const_cast<double*>(matrix.data()), cols, &sA, 0, 0);
        blasfeo_pack_dvec(cols, const_cast<double*>(vector.data()), 1, &sx, 0);

        // Initialize result vector to zero
        blasfeo_dvecse(rows, 0.0, &sy, 0);

        // Perform multiplication
        blasfeo_dgemv_n(rows, cols, 1.0, &sA, 0, 0, &sx, 0, 0.0, &sy, 0, &sy, 0);

        // Unpack result
        blasfeo_unpack_dvec(rows, &sy, 0, result.data(), 1);
    }

    // Pack matrix once and reuse for multiple operations
    void pack_matrix(const std::vector<double>& matrix) {
        if (!initialized) {
            throw std::runtime_error("BlasfeoMatrixOps not initialized");
        }

        blasfeo_pack_dmat(rows, cols, const_cast<double*>(matrix.data()), cols, &sA, 0, 0);
        matrix_packed = true;
    }

    // Use pre-packed matrix for multiplication
    void matrix_vector_mult_packed(const std::vector<double>& vector,
                                  std::vector<double>& result) {
        if (!initialized) {
            throw std::runtime_error("BlasfeoMatrixOps not initialized");
        }

        if (!matrix_packed) {
            throw std::runtime_error("Matrix not packed. Call pack_matrix first.");
        }

        // Pack only the vector
        blasfeo_pack_dvec(cols, const_cast<double*>(vector.data()), 1, &sx, 0);

        // Initialize result vector to zero
        blasfeo_dvecse(rows, 0.0, &sy, 0);

        // Perform multiplication
        blasfeo_dgemv_n(rows, cols, 1.0, &sA, 0, 0, &sx, 0, 0.0, &sy, 0, &sy, 0);

        // Unpack result
        blasfeo_unpack_dvec(rows, &sy, 0, result.data(), 1);
    }

    // Raw pointer versions for even better performance
    void matrix_vector_mult(const double* matrix, const double* vector,
                           double* result) {
        if (!initialized) {
            throw std::runtime_error("BlasfeoMatrixOps not initialized");
        }

        // Pack data
        blasfeo_pack_dmat(rows, cols, const_cast<double*>(matrix), cols, &sA, 0, 0);
        blasfeo_pack_dvec(cols, const_cast<double*>(vector), 1, &sx, 0);

        // Initialize result vector to zero
        blasfeo_dvecse(rows, 0.0, &sy, 0);

        // Perform multiplication
        blasfeo_dgemv_n(rows, cols, 1.0, &sA, 0, 0, &sx, 0, 0.0, &sy, 0, &sy, 0);

        // Unpack result
        blasfeo_unpack_dvec(rows, &sy, 0, result, 1);
    }

    void pack_matrix(const double* matrix) {
        if (!initialized) {
            throw std::runtime_error("BlasfeoMatrixOps not initialized");
        }

        blasfeo_pack_dmat(rows, cols, const_cast<double*>(matrix), cols, &sA, 0, 0);
        matrix_packed = true;
    }

    void matrix_vector_mult_packed(const double* vector, double* result) {
        if (!initialized) {
            throw std::runtime_error("BlasfeoMatrixOps not initialized");
        }

        if (!matrix_packed) {
            throw std::runtime_error("Matrix not packed. Call pack_matrix first.");
        }

        // Pack only the vector
        blasfeo_pack_dvec(cols, const_cast<double*>(vector), 1, &sx, 0);

        // Initialize result vector to zero
        blasfeo_dvecse(rows, 0.0, &sy, 0);

        // Perform multiplication
        blasfeo_dgemv_n(rows, cols, 1.0, &sA, 0, 0, &sx, 0, 0.0, &sy, 0, &sy, 0);

        // Unpack result
        blasfeo_unpack_dvec(rows, &sy, 0, result, 1);
    }

    // Vector operations for SGD optimization
    void vector_subtract(const double* a, const double* b, double* result, int size) {
        // Create temporary vectors if needed
        if (!initialized || cols < size) {
            initialize(1, size);
        }

        // Pack vectors
        blasfeo_pack_dvec(size, const_cast<double*>(a), 1, &sx, 0);
        blasfeo_pack_dvec(size, const_cast<double*>(b), 1, &sy, 0);

        // Create a temporary vector for the result
        struct blasfeo_dvec sz;
        void* memory_dvec_temp = malloc(blasfeo_memsize_dvec(size));
        blasfeo_create_dvec(size, &sz, memory_dvec_temp);

        // Copy a to result
        blasfeo_dveccp(size, &sx, 0, &sz, 0);

        // Perform sz = sz - sy using axpy: sz = sz + (-1.0)*sy
        blasfeo_dvecad(size, -1.0, &sy, 0, &sz, 0);

        // Unpack result
        blasfeo_unpack_dvec(size, &sz, 0, result, 1);

        // Free temporary memory
        free(memory_dvec_temp);
    }

    // Specialized SGD update: weights = weights - learning_rate * gradients
    void sgd_update(const double* weights, const double* gradients,
                   double* result, int size, double learning_rate) {
        // Create temporary vectors if needed
        if (!initialized || cols < size) {
            initialize(1, size);
        }

        // Pack weights
        blasfeo_pack_dvec(size, const_cast<double*>(weights), 1, &sx, 0);

        // Pack gradients
        blasfeo_pack_dvec(size, const_cast<double*>(gradients), 1, &sy, 0);

        // Scale gradients by learning rate
        blasfeo_dvecsc(size, learning_rate, &sy, 0);

        // Create a temporary vector for the result
        struct blasfeo_dvec sz;
        void* memory_dvec_temp = malloc(blasfeo_memsize_dvec(size));
        blasfeo_create_dvec(size, &sz, memory_dvec_temp);

        // Copy weights to result
        blasfeo_dveccp(size, &sx, 0, &sz, 0);

        // Perform sz = sz - learning_rate*gradients using axpy: sz = sz + (-1.0)*sy
        blasfeo_dvecad(size, -1.0, &sy, 0, &sz, 0);

        // Unpack result
        blasfeo_unpack_dvec(size, &sz, 0, result, 1);

        // Free temporary memory
        free(memory_dvec_temp);
    }
};

// Global utility functions for common operations
namespace BlasUtils {
    // Convert 2D vector to flat vector
    template<typename T>
    std::vector<double> flatten_matrix(const std::vector<std::vector<T>>& matrix) {
        if (matrix.empty()) return {};

        size_t rows = matrix.size();
        size_t cols = matrix[0].size();
        std::vector<double> flat(rows * cols);

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                flat[i * cols + j] = static_cast<double>(matrix[i][j]);
            }
        }

        return flat;
    }

    // Convert flat vector back to 2D vector
    template<typename T>
    void unflatten_matrix(const std::vector<double>& flat,
                         std::vector<std::vector<T>>& matrix) {
        if (flat.empty() || matrix.empty()) return;

        size_t rows = matrix.size();
        size_t cols = matrix[0].size();

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                matrix[i][j] = static_cast<T>(flat[i * cols + j]);
            }
        }
    }
}
