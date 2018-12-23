#include <fstream>
#include "data_container.h"

void toBigEndian(unsigned int &value, unsigned char *buffer){
	// Liczby w użytym zbiorze danych są zakodowane w notacji Little Endian.
	// Ta funkcja odwraca kolejność bajtów 32 bitów liczb.
	// Zmienna value to zmienna do której będą zapisane bajty w odwrotnej kolejnością
	// Zmienna buffer to zmienna z tablicą bajtów, które należy odwrócić.
	value = (unsigned int)((unsigned char)(buffer[0]) << 24 |
            (unsigned char)(buffer[1]) << 16 |
            (unsigned char)(buffer[2]) << 8 |
            (unsigned char)(buffer[3]));
}

std::vector<unsigned char> bufferFromFile(string &path){
	// Metoda ma na celu wczytać bajty z pliku do zmiennej typu vector.
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
	// Konstruktor klasy przechowującej dane obrazów z ich klasyfikacją.
	read_images(imageset_path);
	read_labels(labelset_path);
}

pair<unsigned int, unsigned int> DataContainer::getSize(){
	// Metoda zwracająca rozmiary przechowywanów obrazów.
	return make_pair<unsigned int, unsigned int>(height, width);
}

LabeledData DataContainer::getLabeledData(unsigned int index){
	// Metoda zwracająca obraz z jego poprawną klasyfikacją o zadanym indeksie.
	return DataContainer::labeled_images[index];
}

vector<LabeledData> DataContainer::getAllLabeledData(){
	// Metoda zwracająca listę obrazów z poprawnymi klasyfikacjami.
	return DataContainer::labeled_images;
}

void DataContainer::read_images(string &imageset_path){
	// Metoda wczytująca obrazy.
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
	// Metoda wczytująca poprawne klasyfikacje obrazów.
	vector<unsigned char> buffer = bufferFromFile(labelset_path);
	unsigned int size;
	toBigEndian(size, &buffer[4]); // ignoring magic number
	for(int i = 0; i<size; ++i){
		labeled_images[i].label = buffer[i + 8];
	}
}
