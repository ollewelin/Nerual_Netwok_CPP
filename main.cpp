
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include "fc_m_resnet.hpp"
#include<vector>
#include<time.h>


//using namespace cv;
using namespace std;

vector<int> fisher_yates_shuffle(vector<int> table);
int main() {

  cout << "General Neural Network Beta version under work..." << endl;
  srand(time(NULL));

  //=========== Test Neural Network size settings ==============
  fc_m_resnet basic_fc_nn;
  basic_fc_nn.get_version();
  const int inp_nodes = 3;
  const int out_nodes = 3;
  const int hid_layers = 2;
  const int hid_nodes_L1 = 1500;
  const int hid_nodes_L2 = 15;
  //const int hid_nodes_L3 = 7;
  for (int i=0;i<inp_nodes;i++)
  {
    basic_fc_nn.input_layer.push_back(0.0);
    basic_fc_nn.i_layer_delta.push_back(0.0);
  }
  for (int i=0;i<out_nodes;i++)
  {
    basic_fc_nn.output_layer.push_back(0.0);
    basic_fc_nn.target_layer.push_back(0.0);
    basic_fc_nn.o_layer_delta.push_back(0.0);//Not need for End block
  }
  basic_fc_nn.set_nr_of_hidden_layers(hid_layers);
  basic_fc_nn.set_nr_of_hidden_nodes_on_layer_nr(hid_nodes_L1);
  basic_fc_nn.set_nr_of_hidden_nodes_on_layer_nr(hid_nodes_L2);
  //basic_fc_nn.set_nr_of_hidden_nodes_on_layer_nr(hid_nodes_L3);
  // Note that set_nr_of_hidden_nodes_on_layer_nr() cal must be exactly same number as the set_nr_of_hidden_layers(hid_layers)
  //============ Neural Network Size setup is finnish ! ==================

  //=== Now setup the hyper parameters of the Neural Network ====
  basic_fc_nn.momentum = 0.9;
  basic_fc_nn.learning_rate = 0.01;
  basic_fc_nn.dropout_proportion = 0.35;
  double init_random_weight_propotion = 0.001;
  cout << "Do you want to load weights from saved weight file = Y/N " << endl;
  char answer='N';
  cin >> answer;
  if (answer == 'Y' || answer == 'y')
  {
    basic_fc_nn.load_weights("weights.dat");
  }
  else
  {
    basic_fc_nn.randomize_weights(init_random_weight_propotion);
  }

  const int tabl_size = 10;
  vector<int> table;
  table.clear();
  for(int i=0;i<tabl_size;i++)
  {
    table.push_back(0);
  }
  table = fisher_yates_shuffle(table);
  for(int i=0;i<tabl_size;i++)
  {
    cout << "rand table[" << i << "] = " << table[i] << endl;
  }

  


  //fc_1.randomize_weights(0.01);

  cv::Mat test;
  test.create(200, 400, CV_32FC1);
  float *index_ptr_testGrapics = test.ptr<float>(0);
  float pixel_level = 0.0f;
  for(int i=0; i< test.rows * test.cols;i++)
  {
    if(pixel_level < 1.0)
    {
      pixel_level = pixel_level + 0.00001f;
    }
    else{
      pixel_level = 0.0f;
    }
     *index_ptr_testGrapics = pixel_level;
    index_ptr_testGrapics++;
  }
  cv::imshow("diff", test);
  cv::waitKey(5000);
}

vector<int> fisher_yates_shuffle(vector<int> table) {
    int size = table.size();
    for (int i = 0; i < size; i++) {
        table[i] = i;
    }
    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = table[i];
        table[i] = table[j];
        table[j] = temp;
    }
    return table;
}