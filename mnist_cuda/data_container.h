#ifndef __DATA_CONTAINER_H
#define __DATA_CONTAINER_H

#include<iostream>
using namespace std;

struct LabeledData{
	unsigned int label;
	vector<unsigned char> pixels;
};

class DataContainer{
	public:
		DataContainer(string &imageset_path, string &labelset_path);
		pair<unsigned int, unsigned int> getSize();
		vector<LabaledData> getAllLabeledData();
		LabaledData getLabeledData(unsigned int index);

	private:
		unsigned int weight;
		unsigned int height;
		vector<LabeledData> labeled_images;
		unsigned int toBigEndian(unsigned int value);	
		void read_images(&imageset_path);
		void read_labels(&labelset_path)
	
};

#endif  // __DATA_READER_H