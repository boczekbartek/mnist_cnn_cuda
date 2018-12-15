#include<iostream>
#include "data_container.h"

using namespace std;

int main(){
	string images_path = "train-images.idx3-ubyte", labels_path = "train-labels.idx1-ubyte";
	cout << "Processing starting." << endl;
	DataContainer container(images_path, labels_path);
	if(container.getSize().first != 28 && container.getSize().second != 28){
			cout << "Size of images is incorect.";
			return 1;
	}
	if(container.getAllLabeledData().size() != 60000){
			cout << "Number of images is incorrect.";
			return 1;
	}
	std::vector<LabeledData> data = container.getAllLabeledData();
	for(int i = 0; i<data.size(); ++i){
		if(data[i].label > 9 || data[i].label < 0) {
			cout << "Incorrect value of label.";
			return 1;
		}
		for (int j = 0; j <= data[i].pixels.size(); ++j){
			if (data[i].pixels[j] < 0 || data[i].pixels[j] > 255){
				cout << "Incorrect value of pixel";
				return 1;
			}
		}
	}
	cout << "Data were read correctly." << endl;
	return 0;
}
