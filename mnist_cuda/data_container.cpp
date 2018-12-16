#include <fstream>
#include "data_container.h"

void toBigEndian(unsigned int &value, unsigned char *buffer){
	// works for 32 bits integers.
	value = (unsigned int)((unsigned char)(buffer[0]) << 24 |
            (unsigned char)(buffer[1]) << 16 |
            (unsigned char)(buffer[2]) << 8 |
            (unsigned char)(buffer[3]));
	}

std::vector<unsigned char> bufferFromFile(string &path){
	std::vector<unsigned char> buffer;
	std::ifstream ifile(path.c_str());
	ifile.seekg(0, ifile.end);
	int length = ifile.tellg();
	ifile.seekg(0, ifile.beg);
	buffer.resize(length);
	ifile.read((char*)&buffer[0], length);
	return buffer;
}


DataContainer::DataContainer(string &imageset_path, string &labelset_path){
	read_images(imageset_path);
	read_labels(labelset_path);
}

pair<unsigned int, unsigned int> DataContainer::getSize(){
	return make_pair<unsigned int, unsigned int>(height, width);
}

LabeledData DataContainer::getLabeledData(unsigned int index){
	return DataContainer::labeled_images[index];
}

vector<LabeledData> DataContainer::getAllLabeledData(){
	return DataContainer::labeled_images;
}

void DataContainer::read_images(string &imageset_path){
	vector<unsigned char> buffer = bufferFromFile(imageset_path);
	unsigned int size;
	toBigEndian(size, &buffer[4]); // ignoring magic number
	toBigEndian(height, &buffer[8]);
	toBigEndian(width, &buffer[12]);
	labeled_images.resize(size);
	for(int i = 0; i<size; ++i){
		labeled_images[i].pixels.resize(height*width);
		labeled_images[i].pixels = std::vector<unsigned char>(
			buffer.begin() + 16 + (height*width)*i,
			buffer.begin() + 16 + (height*width)*i + (height*width-1)
		);
	}
}

void DataContainer::read_labels(string &labelset_path){
	vector<unsigned char> buffer = bufferFromFile(labelset_path);
	unsigned int size;
	toBigEndian(size, &buffer[4]); // ignoring magic number
	for(int i = 0; i<size; ++i){
		labeled_images[i].label = buffer[i + 8];
	}
}
