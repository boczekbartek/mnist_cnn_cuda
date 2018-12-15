#include <fstream>
#include "data_containe.h"

DataContainer::DataContainer(string &imageset_path, string &labelset_path){
	read_images(imageset_path);
	read_labels(labelset_path);
}

pair<unsigned int, unsigned int> DataContainer::getSize(){
	return make_pair<unsigned int, unsigned int>(width, weight);
}

LabeledData DataContainer::getLabaledData(unsigned int index){
	return labeled_images[index];
}

vector<LabeledData> DataContainer::getAllLabaledData(){
	return labeled_images;
}

void DataContainer::read_images(string &imageset_path){
	ifstream image_file;
	image_file.open(imageset_path, ios::in | ios::binary);
	unsigned int tmp, size;
	image_file.read(&tmp, sizeof(int)); // ignoring magic number
	image_file.read(&size, sizeof(int));
	toBigEndian(size);
	image_file.read(&height, sizeof(int));
	toBigEndian(height);
	image_file.read(&width, sizeof(int));
	toBigEndian(width);
	labeled_images.resize(size);
	for(int i = 0; i<size; ++i){
		labeled_images[i].pixels.resize(height*width);
		for(int j = 0; j<labeled_images.pixels.size(); ++j){
			image_file.read(&labeled_images[i].pixels[j], sizeof(char));
		}
	}
}

void DataContainer::read_labels(string &labelset_path){
	ifstream labels_file;
	image_file.open(imageset_path, ios::in | ios::binary);
	unsigned int tmp, size;
	image_file.read(&tmp, sizeof(int)); // ignoring magic number
	image_file.read(&size, sizeof(int));
	toBigEndian(size);
	for(int i = 0; i<size; ++i){
		image_file.read(&labeled_images[i].label, sizeof(char));
	}
}

void toBigEndian(unsigned int &value){ // works for 32 bits integers.
	unsigned int big_endian_value = value;
	for(int i = 0; i<4; ++i) {
		*(&big_endian_value + i) = *(&value + 3 - i);
	}
	value = big_endian_value;
}