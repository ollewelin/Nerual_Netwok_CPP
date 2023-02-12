#ifndef LOAD_MNIST_DATASET
#define LOAD_MNIST_DATASET
#include<vector>
#include<string>
#include <iostream>
#include <fstream>
using namespace std;
class load_mnist_dataset
{
private:
  char * memblock;

public:
    vector<vector<double>> load_training_data(vector<vector<double>> train_dat);
    
    int get_training_data_set_size(void);
    int get_verify_data_set_size(void);
    int get_one_sample_data_size(void);
    load_mnist_dataset();
    ~load_mnist_dataset();
};


#endif//LOAD_MNIST_DATASET
