
#include <iostream>
// #include <stdio.h>
#include "load_mnist_dataset.hpp"
using namespace std;

// basic file operations
#include <iostream>
#include <fstream>
#include <stdlib.h> // exit(0);

const int mnist_training_size = 60000;
const int mnist_verify_size = 10000;
const int mnist_height = 28;
const int mnist_width = 28;
const int mnist_pix_size = mnist_height * mnist_width;
const int mnist_header_offset = 16;
const int mnist_lable_offset = 8;

/// Input data from
/// t10k-images-idx3-ubyte
/// t10k-labels-idx1-ubyte
/// train-images-idx3-ubyte
/// train-labels-idx1-ubyte
/// http://yann.lecun.com/exdb/mnist/
/// const int Const_nr_pic = 60000;
/// char data_10k_MNIST[10000][MNIST_pix_size];
/// char data_60k_MNIST[60000][MNIST_pix_size];

/*
TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  10000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.
TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  10000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel
Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
*/

double convert_mnist_data(char indata)
{
  double outdata = 0.0;
  unsigned char mnist_uchar = 0.0;
  mnist_uchar = (~(-indata) + 1);
  outdata = ((double)mnist_uchar) / 256.0;
  return outdata;
}

vector<vector<double>> load_mnist_dataset::load_input_data(vector<vector<double>> train_dat, int verify_mode)
{
  streampos size;
  string filename;
  if(verify_mode == 0)
  {
    // training mode
    filename = "train-images-idx3-ubyte";
  }
  else
  {
    // verify mode
    filename = "t10k-images-idx3-ubyte";
  }
  ifstream file(filename, ios::in | ios::binary | ios::ate);
  if (file.is_open())
  {
    size = file.tellg();
    memblock = new char[size];
    file.seekg(0, ios::beg);
    file.read(memblock, size);
    file.close();
    cout << filename << " is in memory" << endl;
  }
  else
  {
    cout << "Unable to open file";
  }

  int dataset_size = train_dat.size();
  int one_sample_data_size = train_dat[0].size();
  int MN_index = mnist_header_offset;

  for (int i = 0; i < dataset_size; i++)
  {
    for (int j = 0; j < one_sample_data_size; j++)
    {
      if ((MN_index - mnist_header_offset) < dataset_size * one_sample_data_size)
      {
        train_dat[i][j] = convert_mnist_data(memblock[MN_index]);
      }
      else
      {
        cout << "Error data from file is larger then dataset vector exit program" << endl;
        exit(0);
      }
      MN_index++;
    }
  }
  return train_dat;
}


vector<vector<double>> load_mnist_dataset::load_lable_data(vector<vector<double>> target_dat, int verify_mode)
{
  streampos size;
  string filename;
  if(verify_mode == 0)
  {
    // training mode
    filename = "train-labels-idx1-ubyte";
  }
  else
  {
    // verify mode
    filename = "t10k-labels-idx1-ubyte";
  }

  ifstream file(filename, ios::in | ios::binary | ios::ate);
  if (file.is_open())
  {
    size = file.tellg();
    memblock = new char[size];
    file.seekg(0, ios::beg);
    file.read(memblock, size);
    file.close();
    cout << filename << " is in memory" << endl;
  }
  else
  {
    cout << "Unable to open file";
  }

  int lable_size = target_dat.size();
  int one_sample_lable_size = target_dat[0].size();
  cout << "lable_size = " << lable_size << endl;
  cout << "one_sample_lable_size = " << one_sample_lable_size << endl;
  int MN_index = mnist_lable_offset;

  for (int i = 0; i < lable_size; i++)
  {
    for (int j = 0; j < one_sample_lable_size; j++)
    {
      if ((MN_index - mnist_header_offset) < lable_size * one_sample_lable_size)
      {
        if(j == (int)memblock[MN_index])
        {
          target_dat[i][j] = (float)1.0;
        }
        else
        {
          target_dat[i][j] = (float)0.0;
        }
      }
      else
      {
        cout << "Error data from file is larger then lable vector exit program" << endl;
        exit(0);
      }
      
    }
    MN_index++;
  }
  return target_dat;
}
int load_mnist_dataset::get_training_data_set_size(void)
{
  return mnist_training_size;
}
int load_mnist_dataset::get_verify_data_set_size(void)
{
  return mnist_verify_size;
}
int load_mnist_dataset::get_one_sample_data_size(void)
{
  return mnist_pix_size;
}

load_mnist_dataset::load_mnist_dataset()
{
  cout << "Constructor load_mnist_dataset " << endl;
}

load_mnist_dataset::~load_mnist_dataset()
{
  delete[] memblock;
  cout << "Destructor load_mnist_dataset " << endl;
}

/*
  char mnist_d = 0;
  double mnist_f = 0.0;
  unsigned char mnist_uchar = 0.0;
  for (int i = 0; i < training_dataset_size; i++)
  {
    for(int j=0;j<MNIST_pix_size;j++)
    {
    mnist_d = MNIST_data[(MNIST_pix_size * i) + j];
    mnist_uchar = (~(-mnist_d) + 1);
    mnist_f = ((double)mnist_uchar)/256.0;
    training_input_data[i][j] = mnist_f;       // 0..1
    }
    target_lable = ((int) MNIST_lable[i]);
    for(int j=0;j<end_out_nodes;j++)
    {
      float targ_dig = 0.0;
      if(target_lable == j)
      {
        targ_dig = 1.0;
      }
      training_target_data[i][j] = targ_dig;
    }
  }
  MNIST_nr = (int)(rand() % MNIST_nr_of_img_p_batch);

  for (int n = 0; n < MNIST_pix_size; n++)
  {
    mnist_d = MNIST_data[(MNIST_pix_size * MNIST_nr) + n];
    mnist_uchar = (~(-mnist_d) + 1);
    mnist_f = ((double)mnist_uchar)/256.0;
  }

*/