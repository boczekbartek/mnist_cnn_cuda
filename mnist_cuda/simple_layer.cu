#define N 1
#define C 1
#define H 1
#define W 28*28
#define CLASS_NUMBER 10
using namespace std;


vector<cudnnTensorDescriptor_t*> descriptions;
vector<float*> in_datas;
vector<cudnnActivationDescriptor_t*> activations;

void add_dense_layer(int batch_size, int feature_of_maps,
                     int height, int width){

    cudnnTensorDescriptor_t* in_desc = new cudnnTensorDescriptor_t;
    descriptions.push_back(in_desc);
    CUDNN_CALL(cudnnCreateTensorDescriptor(in_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(*in_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, N, C, H, W));
    float *in_data;
    in_datas.push_back(in_data);
    CUDA_CALL(cudaMalloc(in_data, batch_size * feature_of_maps *
                         height * width * sizeof(float)));
}

void add_activation(cudnnActivationMode_t activation_mode){
  cudnnActivationDescriptor_t *activation = new cudnnActivationDescriptor_t;
  activations.push_back(activation);
  checkCUDNN(cudnnCreateActivationDescriptor(activation));
  checkCUDNN(cudnnSetActivationDescriptor(activation, activation_mode,
    CUDNN_PROPAGATE_NAN, 0.0));
}

void add_softmax_activation(){

}

void free_resources(){
  for(int i = 0; i<descriptions.size(); ++i){
    CUDNN_CALL(cudnnDestroyTensorDescriptor(*descriptions[i]));
    CUDA_CALL(cudaFree(*in_datas[i]));
    checkCUDNN(cudnnDestroyActivationDescriptor(*activations[i]));
  }
}

int main(){
  //TODO: Reading data

  cudnnHandle_t cudnn;
  CUDNN_CALL(cudnnCreate(&cudnn));
  add_dense_layer(N, C, H, W);
  add_activation(CUDNN_ACTIVATION_TANH);
  add_dense_layer(1, 1, 1, CLASS_NUMBER);

  cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_ACCURATE,
                      CUDNN_SOFTMAX_MODE_CHANNEL,
                      *descriptions[0], *in_datas[0], //TODO: Check this, datas pointers should be device pointers
                      *descriptions[1], *in_datas[1]);
  //TODO: Training

  free_resources();
  return 0;
}
