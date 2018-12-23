// Plik zawierający opis prostej sieci neuronowej bez warstw konwolucyjnych.
#include "data_container.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/transform_reduce.h>
#include <curand.h>
#define N 1
#define C 1
#define H 1
#define W 28*28
#define CLASS_NUMBER 10
using namespace std;

cublasHandle_t handle;
float alfa = 1.0, beta = 0.0;

struct FullConnectedLayer{
  thrust::device_vector<float> *weights;
  thrust::device_vector<float> *outputs;

  FullConnectedLayer(int input, int output){
    weights = new thrust::device_vector<float>(input*output);
    thrust::transform(weights.begin(), weights.end(),
                      weights.begin(), thrust::unary_function<float, float>);
    outputs = new thrust::device_vector<float>(output);
  }

  void multiply(thrust::device_vector<float> *inputs){
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, outputs->size(), N,
                inputs.size(), &alfa, &weights[0], outputs->size(), &inputs[0],
                inputs->size(), &beta, &outputs[0], N);
  }

  void forwardPropagation(thrust::device_vector<float> *inputs,
                          float(float) activation){
    multiply(inputs);
    thrust::transform(outputs.begin(), outputs.end(), outputs.begin(), activation);
  }
};

FullConnectedLayer *fc1, *fc2;

struct softmaxForward
{
    float sum_exp;
    softmaxForward(thrust::device_vector<float> *outputs){
      sum_exp = thrust::transform_reduce(outputs->begin(), outputs->.end(),
                                         expf, init, thrust::plus<float>());
    }
    __host__ __device__ float operator()(float x){
      return expf(x)/sum_exp;
    }
};

void single_forward_propagate(vector<float> &inputs, int label){
  thrust::device_vector<float> inputs(input.begin(), input.end());
  fc1->forwardPropagation(&inputs, tanhf);
  auto softmax = softmaxForward(fc1->outputs);
  fc2->forwardPropagation(fc1->outputs, softmax);
}

void single_backward_propagate(){
  thrust::device_vector<float> inputs(input.begin(), input.end());
  fc1->forwardPropagation(&inputs, tanhf);
  auto softmax = softmaxForward(fc1->outputs);
  fc2->forwardPropagation(fc1->outputs, softmax);
}

int main(){
  // Stworzenie handlera dla cublasa
  cublasCreate(&handle);
  // Zmienne ze ścieżkami do plików z danymi.
  string images_training_path = "train-images.idx3-ubyte",
         labels_training_path = "train-labels.idx1-ubyte",
         images_test_path = "test-images.idx3-ubyte",
         labels_test_path = "test-labels.idx1-ubyte";
  // Wczytanie danych do uczenia i do weryfikacji.
  DataContainer training_container(images_training_path, labels_training_path),
                test_container(images_test_path, labels_test_path);

  fc1 = new FullConnectedLayer(W, W);
  fc2 = new FullConnectedLayer(W, CLASS_NUMBER);
  train(training_container);
  return 0;
}
