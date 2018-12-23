// Plik zawierający opis prostej sieci neuronowej bez warstw konwolucyjnych.

#include "data_container.h"
#define N 1
#define C 1
#define H 1
#define W 28*28
#define CLASS_NUMBER 10
using namespace std;
// Wektor deskryptorów warstw bibliotekii Cudnn
vector<cudnnTensorDescriptor_t*> descriptions;
// Wektor zawierający dane wejściowe do kolejnych warstw sieci neuronowej.
vector<float*> in_datas;
// Wektor zastosowanych aktywacji w sieci neuronowej.
vector<cudnnActivationDescriptor_t*> activations;

void add_dense_layer(int batch_size, int feature_of_maps,
                     int height, int width){
    // Funkcja dodająca warstwe dense do naszej sieci neuronowej.
    cudnnTensorDescriptor_t* in_desc = new cudnnTensorDescriptor_t;
    descriptions.push_back(in_desc);
    // Stworzenie tensora naszej warstwe.
    CUDNN_CALL(cudnnCreateTensorDescriptor(in_desc));
    // Ustawienie parametrów tensora. Kolejno, handler do deskryptora,
    // opis wczytywania danych, batch size, liczba kanałów w obrazie,
    // wysokość i szerokość obrazu.
    CUDNN_CALL(cudnnSetTensor4dDescriptor(*in_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, batch_size,
                                          feature_of_maps, height, width));

    // Alokacja pamięci na wejście sieci neuronowej.
    float *in_data;
    in_datas.push_back(in_data);
    CUDA_CALL(cudaMalloc(in_data, batch_size * feature_of_maps *
                         height * width * sizeof(float)));
}

void add_activation(cudnnActivationMode_t activation_mode){
  // Dodanie funkcji aktywacji do sieci neuronowej.
  cudnnActivationDescriptor_t *activation = new cudnnActivationDescriptor_t;
  activations.push_back(activation);
  checkCUDNN(cudnnCreateActivationDescriptor(activation));
  checkCUDNN(cudnnSetActivationDescriptor(activation, activation_mode,
                                          CUDNN_PROPAGATE_NAN, 0.0));
}

void free_resources(){
  // Zwolnienie zasobów
  for(int i = 0; i<descriptions.size(); ++i){
    CUDNN_CALL(cudnnDestroyTensorDescriptor(*descriptions[i]));
    CUDA_CALL(cudaFree(*in_datas[i]));
    checkCUDNN(cudnnDestroyActivationDescriptor(*activations[i]));
  }
}

void train_network(int batch_size, cudnnHandle_t &cudnn){
  float alpha = 1.0f, beta = 0.0f;
  checkCudaErrors(cublasSgemm(cudnn, CUBLAS_OP_T, CUBLAS_OP_N,
                              CLASS_NUMBER, batch_size, W,
                              &alpha,
                              pfc1, W, pool2, W, //TODO
                              &beta,
                              fc1, CLASS_NUMBER)); //TODO
}

int main(){
  // Zmienne ze ścieżkami do plików z danymi.
  string images_training_path = "train-images.idx3-ubyte",
         labels_training_path = "train-labels.idx1-ubyte",
         images_test_path = "test-images.idx3-ubyte",
         labels_test_path = "test-labels.idx1-ubyte";

  // Wczytanie danych do uczenia i do weryfikacji.
  DataContainer training_container(images_training_path, labels_training_path),
                test_container(images_test_path, labels_test_path);
  cudnnHandle_t cudnn;
  CUDNN_CALL(cudnnCreate(&cudnn));
  add_dense_layer(N, C, H, W);
  add_activation(CUDNN_ACTIVATION_TANH);
  add_dense_layer(1, 1, 1, CLASS_NUMBER);
  train_network(N, cudnn);

  // cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_ACCURATE,
  //                     CUDNN_SOFTMAX_MODE_CHANNEL,
  //                     *descriptions[0], *in_datas[0],
  //                     *descriptions[1], *in_datas[1]);

  free_resources();
  return 0;
}
