#include<iostream>
#include "data_container.h"

using namespace std;

int main(){
	DataContainer container("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
	cout << "Data were read." << endl;
	return 0;
}