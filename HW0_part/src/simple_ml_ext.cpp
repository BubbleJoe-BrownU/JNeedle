#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void print_matrix(const float *X, size_t m, size_t n);

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    for(size_t i=0; i<m; i+= batch) {
        size_t end = i+batch > m? m : i+batch;
        // for each example in this batch
        float logits[batch][k];
        for(size_t b=0; b<end-i; b++) {
            // for each output dimension
            for (size_t j=0; j<k; j++) {
                // initialize the logits matrix
                logits[b][j] = 0;
                // for each feature dimension
                    for(size_t v=0; v<n; v++) {
                    logits[b][j] += X[(i+b)*n + v] * theta[v*k + j];
                }
            }
        }
        // normalize the logits by subtracting the max value
        for (size_t b=0; b<end-i; b++) {
            float max = logits[b][0];
            for (size_t j=1; j<k; j++) {
                if (logits[b][j] > max) max = logits[b][j];
            }
            for (size_t j=0; j<k; j++) {
                logits[b][j] -= max;
            }
        }
        // apply softmax to convert logits to probabilities
        for (size_t b=0; b<end-i; b++) {
            float logits_sum_exp = 0;
            // exponential
            for (size_t j=0; j<k; j++) {
                logits[b][j] = exp(logits[b][j]);
                logits_sum_exp += logits[b][j];
            }
            // normalization
            for (size_t j=0; j<k; j++) {
                logits[b][j] /= logits_sum_exp;
            }
            // test if the sum of probabilities is 1
            // test passed :)
        }
        // print_matrix((const float *)logits, end-i, k);
        // get the one_hot label vector
        float labels[end-i][k];
        for (size_t b=0; b<end-i; b++) {
            for (size_t j=0; j<k; j++) {
                labels[b][j] = 0;
                if (j == y[i+b]) labels[b][j] = 1;
            }
        }
        // print_matrix((const float *)labels, end-i, k);
        // calculate the gradient, G = mean(X.T @ (probs - labels))
        float gradients[n][k];
        for (size_t v=0; v<n; ++v) {
            for (size_t j=0; j<k; ++j) {
                // initialize the gradient matrix
                gradients[v][j] = 0;
                for (size_t b=0; b<end-i; b++) {
                    gradients[v][j] += X[(i+b)*n + v] * (logits[b][j] - labels[b][j]);
                }
                gradients[v][j] /= (end - i);
            }
        }

        // perform the gradient descent: update matrix theta
        for (size_t v=0; v<n; ++v) {
            for (size_t j=0; j<k; ++j) {
                theta[v*k + j] -= lr* gradients[v][j];
            }
        }
        // print_matrix(theta, n, k);
        
    }
    /// END YOUR CODE
}

void print_matrix(const float *X, size_t m, size_t n) {
    for (size_t i=0; i<m; ++i) {
        for (size_t j=0; j<n; ++j) {
            std::cout << X[i*n + j] << " ";
        }
        std::cout << std::endl;
    }
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
 