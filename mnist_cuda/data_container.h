#ifndef __DATA_CONTAINER_H
#define __DATA_CONTAINER_H

#include<iostream>
#include<vector>
using namespace std;

struct LabeledData{
	// Struktura opisująca jeden obraz, zawierająca także informacje
	// o poprawnej klasyfikacji obrazu --> zmienna label.
	unsigned int label;
	vector<float> pixels;
};

class DataContainer{
	// Klasa umożliwiająca wczytanie danych z bazy mnist.
	// Klasa przechowuje informacje o obrazach oraz ich klasyfikacji
	// Metody są opisane z pliku ich dokumentacją.
	public:
		DataContainer(string &imageset_path, string &labelset_path);
		pair<unsigned int, unsigned int> getSize();
		vector<LabeledData> getAllLabeledData();
		LabeledData getLabeledData(unsigned int index);

	private:
		// Szerokość obrazów.
		unsigned int width;
		// Wysokość obrazów.
		unsigned int height;
		// Lista obrazów z ich klasyfikacją.
		vector<LabeledData> labeled_images;
		void read_images(string &imageset_path);
		void read_labels(string &labelset_path);

};

#endif  // __DATA_READER_H
