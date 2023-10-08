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
  void download_mnist(void);
  string filename_tr_data;
  string filename_tr_lable;
  string filename_ver_data;
  string filename_ver_lable;

public:
    vector<vector<double>> load_input_data(vector<vector<double>> input_dat, int verify_mode);
    vector<vector<double>> load_lable_data(vector<vector<double>> target_dat, int verify_mode);
    
    int get_training_data_set_size(void);
    int get_verify_data_set_size(void);
    int get_one_sample_data_size(void);
    load_mnist_dataset();
    ~load_mnist_dataset();
};


#endif//LOAD_MNIST_DATASET
