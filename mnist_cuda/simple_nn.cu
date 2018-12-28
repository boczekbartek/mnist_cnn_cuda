// Plik zawierający opis prostej sieci neuronowej bez warstw konwolucyjnych.
#include "data_container.h"
#include <thrust/random.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/transform_reduce.h>
#include <cublasXt.h>
#include <curand.h>
#define N 1
#define C 1
#define H 1
#define W 28*28
#define CLASS_NUMBER 10
#define ITERATION_COUNT 1000000
#define LERNING_RATE 0.1
using namespace std;

cublasHandle_t handle;
float alfa = 1.0, beta = 0.0;

struct RandomGenerator
{
    float a, b;

    __host__ __device__
    RandomGenerator(float _a=0.f, float _b=1.f) : a(_a), b(_b) {};

    __host__ __device__
    float operator()(float n)
        {
            thrust::default_random_engine rng;
            thrust::uniform_real_distribution<float> dist(a, b);
            rng.discard((int)n);

            return dist(rng);
        }
};

struct FullConnectedLayer{
  // Struktura prostej warstwy pełnopołączonej sieci neuronowej.
  thrust::device_vector<float> *weights;
  thrust::device_vector<float> *derivative_weights;
  thrust::device_vector<float> *outputs_no_activation;
  thrust::device_vector<float> *outputs;
  thrust::device_vector<float> *derivative_outputs;
  thrust::device_vector<float> *delta;

  FullConnectedLayer(int input, int output){
    // Konstruktor przyjmuje rozmiar wyjścia oraz wejścia warstwy sieci neuronowej.
    // Macierz wag (o rozmiarze input*output) jest inicjalizowana losowo.
    weights = new thrust::device_vector<float>(input*output);
    derivative_weights = new thrust::device_vector<float>(input*output);
    thrust::transform(weights->begin(), weights->end(), weights->begin(),
                      RandomGenerator());
    outputs = new thrust::device_vector<float>(output);
    derivative_outputs = new thrust::device_vector<float>(output);
    outputs_no_activation = new thrust::device_vector<float>(output);
    delta = new thrust::device_vector<float>(output);
  }

  void multiplyForward(thrust::device_vector<float> *inputs){
    // Mnożenie macierzy wykorzystawe w propagacji prostej: weights*inputs = outputs
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, outputs->size(), N,
                inputs->size(), &alfa, thrust::raw_pointer_cast(weights->data()),
                outputs->size(), thrust::raw_pointer_cast(inputs->data()),
                inputs->size(), &beta,
                thrust::raw_pointer_cast(outputs_no_activation->data()), N);
  }

  void multiplyBackward(thrust::device_vector<float> *inputs){
    // Obliczenie dW
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, outputs->size(), N,
                inputs->size(), &alfa, thrust::raw_pointer_cast(delta->data()),
                outputs->size(), thrust::raw_pointer_cast(inputs->data()),
                inputs->size(), &beta, thrust::raw_pointer_cast(derivative_weights->data()), N);

    thrust::transform(derivative_outputs->begin(), derivative_outputs->end(),
                      thrust::make_constant_iterator(LERNING_RATE),
                      derivative_outputs->begin(), thrust::multiplies<int>());
  }

  template <typename T>
  void forwardPropagation(thrust::device_vector<float> *inputs,
                          T activation){
    // Mnożenie w propagacji prostej.
    multiplyForward(inputs);
    // Zaaplikowanie funkcji aktywacji do wektora (lub macierzy) wyjściowego.
    thrust::transform(outputs_no_activation->begin(), outputs_no_activation->end(),
                      outputs->begin(), activation);
  }

  template <typename T>
  void backwardPropagation(thrust::device_vector<float> *inputs, int label,
                           T derivative_activation){
      // Propagacja wsteczna dla ostatniej warstwy sieci neuronowej
      thrust::transform(outputs_no_activation->begin(), outputs_no_activation->end(),
                        derivative_outputs->begin(), derivative_activation);
      // Odjecie wektora prawidlowego od wektora wyjscia warstwy
      (*outputs)[label] -= 1;
      countDelta();
      multiplyBackward(inputs);
  }

  template <typename T>
  void backwardPropagation(vector<float> &inputs,
                           thrust::device_vector<float> *weights_nextlayer,
                           thrust::device_vector<float> *delta_nextlayer,
                           T derivative_activation){
    // Propagacja wsteczna dla nieostatnich (w tym przykladzie pierwszej)
    // warstw sieci neuronowej.
    thrust::device_vector<float>* d_inputs = new thrust::device_vector<float>(inputs.begin(), inputs.end());
    thrust::transform(outputs_no_activation->begin(), outputs_no_activation->end(),
                      derivative_outputs->begin(), derivative_activation);
    countDelta(delta_nextlayer, weights_nextlayer);
    multiplyBackward(d_inputs);
  }

  void countDelta(){
    // Mnożenie hadamarda macierzy: (outputs - good_results) * f'(W*input)
    thrust::transform(outputs->begin(), outputs->end(),
                      derivative_outputs->begin(), delta->begin(),
                      thrust::multiplies<float>());
  }

  void countDelta(thrust::device_vector<float> *weights_nextlayer,
                  thrust::device_vector<float> *delta_nextlayer){
    // Mnożenie hadamarda macierzy: (W_nextDelta_next) * f'(W*input)
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, outputs->size(), N,
                delta_nextlayer->size(), &alfa, thrust::raw_pointer_cast(weights->data()),
                outputs->size(), thrust::raw_pointer_cast(delta_nextlayer->data()),
                delta_nextlayer->size(), &beta, thrust::raw_pointer_cast(outputs->data()), N);
    countDelta();
  }

  void updateWeights(){
    // Aktualizacja wartości wag. W -= dW
    thrust::transform(weights->begin(), weights->end(),
                      derivative_weights->begin(), weights->begin(),
                      thrust::minus<float>());
  }
};

FullConnectedLayer *fc1, *fc2;

struct expf_functor{
  __host__ __device__ float operator()(float x){
    return expf(x);
  }
};


struct softmaxForward
{
    // Implementacja funkcji softmax. (f(x1) = e^xi / suma(e^x) )
    float sum_exp;
    softmaxForward(thrust::device_vector<float> *outputs){
      sum_exp = thrust::transform_reduce(outputs->begin(), outputs->end(),
                                         expf_functor(), 0.0, thrust::plus<float>());
    }
    __host__ __device__ float operator()(float x){
      return expf(x)/sum_exp;
    }
};

struct softmaxderivative
{
    float sum_exp;

    softmaxderivative(thrust::device_vector<float> *outputs){
      sum_exp = thrust::transform_reduce(outputs->begin(), outputs->end(),
                                         expf_functor(), 0.0, thrust::plus<float>());
    }

    __host__ __device__ float operator()(float x){
      return (expf(x)/sum_exp)*(1 - expf(x)/sum_exp);
    }
};

struct tangensHderivative
{
    __host__ __device__ float operator()(float x){
      return 1 - tanhf(x)*tanhf(x);
    }
};

void single_forward_propagate(vector<float> inputs){
  /* Propagacja danych. Pierwszą warstwe możemy interpretować jako warstwa
     niezmienionych danych. Pierwsza warstwa ukryta posiada 28*28 neuronów.
     Warstwa ostatnia (wyjściowa) posiada 10 wyjść, tyle samo co klas obrazów.
     Użyte funkcje aktywacji to kolejno: tangens hiperdboliczny oraz softmax.
  */
  thrust::device_vector<float> inputs_device(inputs.begin(), inputs.end());
  fc1->forwardPropagation(&inputs_device, tanhf);
  softmaxForward softmax = softmaxForward(fc1->outputs);
  fc2->forwardPropagation(fc1->outputs, softmax);
}

void single_backward_propagate(std::vector<float> inputs, int label){
  // 'Optymalniejsze' obliczenie roznicy wektora wyjściowego od prawidłowej predykacji.
  // W "prawidłowym" wektorze wszystkie skalary oprócz tego występujacego na pozycji
  // label są zerowe.
  softmaxderivative softmax_derivative = softmaxderivative(fc2->outputs);
  tangensHderivative tanhf_derivative = tangensHderivative();
  fc2->backwardPropagation(fc1->outputs, label, softmax_derivative);
  fc1->backwardPropagation(inputs, fc2->weights, fc2->delta, tanhf_derivative);
  fc2->updateWeights();
  fc1->updateWeights();
}

void train(DataContainer &training_container){
  /* Algorytm trenowania --> narazie implementacja naiwna
     Algorytm wykonuje INTERATION_COUNT razy propagacje prostą i wsteczną
     na całym zbiorze treningowym */
  for(int i = 0; i<ITERATION_COUNT; ++i){
    for(int j = 0; j<training_container.getAllLabeledData().size(); ++j){
      single_forward_propagate(training_container.getLabeledData(i).pixels);
      single_backward_propagate(training_container.getLabeledData(i).pixels,
                                training_container.getLabeledData(i).label);
    }
  }
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

  // Inicjalizacja dwóch warstw sieci neuronowej
  fc1 = new FullConnectedLayer(W, W);
  fc2 = new FullConnectedLayer(W, CLASS_NUMBER);
  // Algorytm trenowania wag sieci
  train(training_container);
  return 0;
}
