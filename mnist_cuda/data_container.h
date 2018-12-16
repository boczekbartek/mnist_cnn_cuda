#ifndef __DATA_CONTAINER_H
#define __DATA_CONTAINER_H

#include<iostream>
#include<vector>
using namespace std;

struct LabeledData{
	unsigned int label;
	vector<unsigned char> pixels;
};

class DataContainer{
	public:
		DataContainer(string &imageset_path, string &labelset_path);
		pair<unsigned int, unsigned int> getSize();
		vector<LabeledData> getAllLabeledData();
		LabeledData getLabeledData(unsigned int index);

	private:
		unsigned int width;
		unsigned int height;
		vector<LabeledData> labeled_images;
		void read_images(string &imageset_path);
		void read_labels(string &labelset_path);

};

#endif  // __DATA_READER_H
