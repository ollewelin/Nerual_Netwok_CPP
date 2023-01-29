
#include <iostream>
#include <stdio.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include "fc_m_resnet.hpp"
#include<vector>
#include<time.h>

//using namespace cv;
using namespace std;


#include <termios.h>// kbhit linux
#include <unistd.h>// kbhit linux
#include <fcntl.h>// kbhit linux


struct termios oldt, newt;
int ch;
int oldf;



int kbhit(void)
{
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    if(ch != EOF)
    {
        ungetc(ch, stdin);
        return 1;
    }
    return 0;
}

void run_pause(void)
{
  
    if(kbhit())
    {
      cout << "Pause training, start with hit Enter " << endl;
      char key_press=' ';
      key_press=getchar();
      if(key_press == ' ')
      {
        key_press=getchar();
      }
      cout << "Run training again " << endl;
     }
  
}

vector<int> fisher_yates_shuffle(vector<int> table);
int main() {

  cout << "General Neural Network Beta version under work..." << endl;
  srand(time(NULL));
  char answer;


  //=========== Test Neural Network size settings ==============
  fc_m_resnet basic_fc_nn;
  string weight_filename;
  weight_filename = "weights.dat";
  basic_fc_nn.get_version();
  basic_fc_nn.block_type = 2;
  basic_fc_nn.use_softmax = 0;
  basic_fc_nn.activation_function_mode = 0;
  basic_fc_nn.use_skip_connect_mode = 0;
  basic_fc_nn.use_dopouts = 0;
  const int inp_nodes = 3;
  const int out_nodes = 3;
  const int hid_layers = 2;
  const int hid_nodes_L1 = 10;
  const int hid_nodes_L2 = 30;
  const int hid_nodes_L3 = 7;
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
  basic_fc_nn.dropout_proportion = 0.15;
  basic_fc_nn.fix_leaky_proportion = 0.05;
  double init_random_weight_propotion = 0.001;
  cout << "Do you want to load weights from saved weight file = Y/N " << endl;
  cin >> answer;
  if (answer == 'Y' || answer == 'y')
  {
    basic_fc_nn.load_weights(weight_filename);
  }
  else
  {
    basic_fc_nn.randomize_weights(init_random_weight_propotion);
  }

  const int training_epocs = 10000;//One epocs go through the hole data set once
  const int save_after_epcs = 100;
  int save_epoc_counter = 0;
  const int training_dataset_size = 1000;
  const int verify_dataset_size = 100;
  const int verify_after_x_nr_epocs = 10;
  int verify_after_epc_cnt = 0;
  double best_training_loss = 1000000000;
  double best_verify_loss = best_training_loss;
  const double stop_training_when_verify_rise_propotion = 0.02;
  vector<vector<double>> training_target_data;
  vector<vector<double>> training_input_data;
  vector<vector<double>> verify_target_data;
  vector<vector<double>> verify_input_data;
  vector<int> training_order_list;
  vector<int> verify_order_list;
  for(int i=0;i<training_dataset_size;i++)
  {
    training_order_list.push_back(0);
  }
  for(int i=0;i<verify_dataset_size;i++)
  {
    verify_order_list.push_back(0);
  }
  verify_order_list = fisher_yates_shuffle(verify_order_list);
  training_order_list = fisher_yates_shuffle(training_order_list);

  vector<double> dummy_one_target_data_point;
  vector<double> dummy_one_training_data_point;
  for(int i=0;i<out_nodes;i++)
  {
    dummy_one_target_data_point.push_back(0.0);
  }
  for(int i=0;i<inp_nodes;i++)
  {
    dummy_one_training_data_point.push_back(0.0);
  }
  for(int i=0;i<training_dataset_size;i++)
  {
    training_target_data.push_back(dummy_one_target_data_point);
    training_input_data.push_back(dummy_one_training_data_point);
  }
  for(int i=0;i<verify_dataset_size;i++)
  {
    verify_target_data.push_back(dummy_one_target_data_point);
    verify_input_data.push_back(dummy_one_training_data_point);
  }


  // --------- Toy training data example learning a sinus, cos and x^2 function ------------
  #define M_PI_local		3.14159265358979323846
  for(int i=0;i<training_dataset_size;i++)
  {
    double linear_line = ((double)i / (double)training_dataset_size);
    training_input_data[i][0] = linear_line;// 0..1
    training_input_data[i][1] = -linear_line;// 0..1
    training_input_data[i][2] = linear_line * 1.0;// 0..10
    training_target_data[i][0] = (sin(linear_line * 2.0 * M_PI_local)) * 0.5 + 0.5;
    training_target_data[i][1] = (cos(linear_line * 2.0 * M_PI_local)) * 0.5 + 0.5;
    training_target_data[i][2] = (linear_line * linear_line) * 0.5 + 0.5;
  }
  for(int i=0;i<verify_dataset_size;i++)
  {
    double linear_line = ((double)i / (double)training_dataset_size);
    verify_input_data[i][0] = linear_line;// 0..1
    verify_input_data[i][1] = -linear_line;// 0..1
    verify_input_data[i][2] = linear_line * 10.0;// 0..10
    verify_target_data[i][0] = (sin(linear_line * 2.0 * M_PI_local)) * 0.5 + 0.5;
    verify_target_data[i][1] = (cos(linear_line * 2.0 * M_PI_local)) * 0.5 + 0.5;
    verify_target_data[i][3] = (linear_line * linear_line) * 0.5 + 0.5;
  }
  //------------------------- Toy example setup finnis -------------------------------------



  int test_nr_hidden_layer = 1;
  int check_n_Lx = 0;
  int check_n_src_Lx = 0;
  switch(test_nr_hidden_layer)
  {
    case(0):
      check_n_Lx = hid_nodes_L1;
      check_n_src_Lx = inp_nodes;
    break;
    case(1):
      check_n_Lx = hid_nodes_L2;
      check_n_src_Lx = hid_nodes_L1;
    break;
    case(2):
      check_n_Lx = hid_nodes_L3;
      check_n_src_Lx = hid_nodes_L2;
    break;

  }
  int test_nr_hidden_delta = 5; 
  int test_nr_hidden_weight = 2; 
  if(hid_layers > test_nr_hidden_layer && test_nr_hidden_delta < check_n_Lx && test_nr_hidden_weight < check_n_src_Lx)
  {
  //check derivates 
  basic_fc_nn.learning_rate = 0.0;
  basic_fc_nn.use_dopouts = 0;
  basic_fc_nn.use_softmax = 0;
  basic_fc_nn.use_skip_connect_mode = 0;
  basic_fc_nn.fix_leaky_proportion = 0.05;
  basic_fc_nn.activation_function_mode = 0;//0= Sigmoid function 
  int i = (training_dataset_size/2);
    double linear_line = ((double)i / (double)training_dataset_size);
    training_input_data[i][0] = linear_line;// 0..1
    training_input_data[i][1] = -linear_line;// 0..1
    training_input_data[i][2] = linear_line * 1.0;// 0..10
    training_target_data[i][0] = (sin(linear_line * 2.0 * M_PI_local)) * 0.5 + 0.5;
    training_target_data[i][1] = (cos(linear_line * 2.0 * M_PI_local)) * 0.5 + 0.5;
    training_target_data[i][2] = (linear_line * linear_line) * 0.5 + 0.5;


  //double loss_d1 = basic_fc_nn.loss;
  //double epsion_adjust_weight_d1 = 0.0;
  double epsion_adjust_weight_d2 = 0.015;
  double gradient_d1 = 0.0;
  double error_d1 = 0.0;
  double gradient_d2 = 0.0;
  double error_d2 = 0.0;

  //gradient_d1 = basic_fc_nn.verify_gradient(0,0,1,0.0);
  basic_fc_nn.forward_pass();
  basic_fc_nn.backpropagtion_and_update();
  gradient_d1 = basic_fc_nn.verify_gradient(test_nr_hidden_layer,test_nr_hidden_delta,test_nr_hidden_weight,0.0);
  error_d1 = basic_fc_nn.calc_error_verify_grad();

  gradient_d2 = basic_fc_nn.verify_gradient(test_nr_hidden_layer,test_nr_hidden_delta,test_nr_hidden_weight,epsion_adjust_weight_d2);
  basic_fc_nn.forward_pass();
  basic_fc_nn.backpropagtion_and_update();
  error_d2 = basic_fc_nn.calc_error_verify_grad();
  gradient_d2 = basic_fc_nn.verify_gradient(test_nr_hidden_layer,test_nr_hidden_delta,test_nr_hidden_weight,epsion_adjust_weight_d2);
  
  double delta_error = error_d1 - error_d2;
  double deriv_numeric_test = delta_error / -epsion_adjust_weight_d2;
  
  cout << "delta_error = " << delta_error << endl;
  cout << "error_d1 = " << error_d1 << endl;
  cout << "error_d2 = " << error_d2 << endl;
  cout << "gradient_d1 = " << gradient_d1 << endl;
  cout << "gradient_d2 = " << gradient_d2 << endl;
//  cout << "deriv_numeric_test = " << deriv_numeric_test << endl;
  cout << "deriv_numeric_test / 2 = " << deriv_numeric_test / 2 << endl;
  double dummy = basic_fc_nn.verify_gradient(1,0,1, - epsion_adjust_weight_d2);
  cout << "Hit Y/y to continue " << endl;
  cin >> answer;
  while (1)
  {
      if(answer == 'Y' || answer == 'y')
      {

        break;
      }
      cin >> answer;
  }
  basic_fc_nn.momentum = 0.9;
  basic_fc_nn.learning_rate = 0.001;
  basic_fc_nn.dropout_proportion = 0.25;
  basic_fc_nn.fix_leaky_proportion = 0.05;
  basic_fc_nn.use_dopouts = 0;
  }
  else
  {
    cout << "skip test verify hidden gradient test" << endl;
  }
  
  //Start trining 
  int do_verify_if_best_trained = 0;
  int stop_training = 0;

  for(int epc=0;epc<training_epocs;epc++)
  {
    if(stop_training == 1)
    {
      break;
    }
    
    run_pause();
     //============ Traning ==================

    //verify_order_list = fisher_yates_shuffle(verify_order_list);
    training_order_list = fisher_yates_shuffle(training_order_list);
    basic_fc_nn.loss = 0.0;
    for(int i=0;i<training_dataset_size;i++)
    {
      for(int j=0;j<inp_nodes;j++)
      {
        basic_fc_nn.input_layer[j] = training_input_data[training_order_list[i]][j];
      }
      basic_fc_nn.forward_pass();
      for(int j=0;j<out_nodes;j++)
      {
        basic_fc_nn.target_layer[j] = training_target_data[training_order_list[i]][j];
      }
      basic_fc_nn.backpropagtion_and_update();
    }
    cout << "Epoch " << epc << endl;
    cout << "input node [0] = " << basic_fc_nn.input_layer[0] <<endl;
    for(int k=0;k<out_nodes;k++)
    {
      cout << "Output node [" << k << "] = " << basic_fc_nn.output_layer[k] << "  Target node [" << k << "] = " << basic_fc_nn.target_layer[k] <<endl;
    }
    
    cout << "Training loss = " << basic_fc_nn.loss <<endl;
    if(save_epoc_counter < save_after_epcs-1)
    {
      save_epoc_counter++;
    }
    else
    {
      save_epoc_counter=0;
      basic_fc_nn.save_weights(weight_filename);
    }
    
  }

  basic_fc_nn.~fc_m_resnet();


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


  basic_fc_nn.~fc_m_resnet();
  
  for(int i=0;i<training_dataset_size;i++)
  {
    training_input_data[i].clear();
    training_target_data[i].clear();
  }
    training_input_data.clear();
    training_target_data.clear();
  for(int i=0;i<verify_dataset_size;i++)
  {
    verify_input_data[i].clear();
    verify_target_data[i].clear();
  }
  training_order_list.clear();
  verify_order_list.clear();
  dummy_one_target_data_point.clear();
  dummy_one_training_data_point.clear();



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