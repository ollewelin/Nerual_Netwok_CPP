
#include <iostream>
#include <stdio.h>

#include "fc_m_resnet.hpp"
#include <vector>
#include <time.h>

using namespace std;


#include <cstdlib>
#include <ctime>
#include <math.h>   // exp
#include <stdlib.h> // exit(0);

using namespace std;
#include <cstdlib>

char MNIST_filename[100];
/// Input data from
/// t10k-images-idx3-ubyte
/// t10k-labels-idx1-ubyte
/// train-images-idx3-ubyte
/// train-labels-idx1-ubyte
/// http://yann.lecun.com/exdb/mnist/
int use_MNIST_verify_set = 0;
const int MNIST_height = 28;
const int MNIST_width = 28;
int MNIST_nr_of_img_p_batch = 60000;
/// const int Const_nr_pic = 60000;
const int MNIST_pix_size = MNIST_height * MNIST_width;
const int MNIST_RGB_pixels = MNIST_pix_size;
/// char data_10k_MNIST[10000][MNIST_pix_size];
/// char data_60k_MNIST[60000][MNIST_pix_size];

const int MNIST_header_offset = 16;
const int MNIST_lable_offset = 8;
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

int get_MNIST_lable_file_size(void);
int get_MNIST_file_size(void);

vector<int> fisher_yates_shuffle(vector<int> table);

int main()
{

  cout << "General Neural Network Beta version under work..." << endl;
  srand(time(NULL));
  char answer;
  char answer_character;
  //=====get MNIST ==
  int MNIST_file_size = 0;
  /// Read database train-images-idx3-ubyte
  MNIST_file_size = get_MNIST_file_size();
  cout << "MNIST_file_size = " << (int)MNIST_file_size << endl;

  // read_10k_MNIST();
  char *MNIST_data;
  MNIST_data = new char[MNIST_file_size];
  FILE *fp;
  char c_data = 0;
  if (use_MNIST_verify_set == 0)
  {
    fp = fopen("train-images-idx3-ubyte", "rb");
  }
  else
  {
    fp = fopen("t10k-images-idx3-ubyte", "rb");
  }
  if (fp == NULL)
  {
    perror("Error in opening train-images-idx3-ubyte file");
    return (-1);
  }
  int MN_index = 0;
  for (int i = 0; i < MNIST_file_size; i++)
  {
    c_data = fgetc(fp);
   // cout << "c_data = " << (int)c_data << endl;
    if (feof(fp))
    {
      break;
    }
    // printf("c_data %d\n", c_data);
    MNIST_data[MN_index] = c_data;

    if ((MNIST_header_offset - 1) < i)
    {
      MN_index++;
    }
  }

  //while(1){};

  fclose(fp);
  printf("train.. or t10k.. ..-images-idx3-ubyte file is successfully loaded in to MNIST_data[MN_index] memory\n");
  /// Read lable
  /// Read train-labels-idx1-ubyte
  MNIST_file_size = get_MNIST_lable_file_size();
  // read_10k_MNIST();
  char *MNIST_lable;
  MNIST_lable = new char[MNIST_file_size];
  // FILE *fp;
  c_data = 0;
  if (use_MNIST_verify_set == 0)
  {
    fp = fopen("train-labels-idx1-ubyte", "rb");
  }
  else
  {
    fp = fopen("t10k-labels-idx1-ubyte", "rb");
  }

  if (fp == NULL)
  {
    if (use_MNIST_verify_set == 0)
    {
      perror("Error in opening train-labels-idx1-ubyte file");
    }
    else
    {
      perror("Error in opening t10k-labels-idx1-ubyte file");
    }

    return (-1);
  }
  MN_index = 0;
  for (int i = 0; i < MNIST_file_size; i++)
  {

    c_data = fgetc(fp);
    if (feof(fp))
    {
      break;
    }
    // printf("c_data %d\n", c_data);
    MNIST_lable[MN_index] = c_data;
    if ((MNIST_lable_offset - 1) < i)
    {
      MN_index++;
    }
  }
  fclose(fp);
  printf("train... or t10k...  ...-labels-idx1-ubyte file is successfully loaded in to MNIST_lable[MN_index] memory\n");

  //==========

  //=========== Test Neural Network size settings ==============
  fc_m_resnet basic_fc_nn;
  string weight_filename;

  weight_filename = "weights2.dat";
  basic_fc_nn.get_version();
  basic_fc_nn.block_type = 2;
  basic_fc_nn.use_softmax = 1;
  basic_fc_nn.activation_function_mode = 0;
  basic_fc_nn.use_skip_connect_mode = 0;
  basic_fc_nn.use_dopouts = 1;
  basic_fc_nn.dropout_proportion = 0.25;
  const int inp_nodes = MNIST_pix_size;
  const int out_nodes = 10;
  const int hid_layers = 2;
  const int hid_nodes_L1 = 300;
  const int hid_nodes_L2 = 15;
  // const int hid_nodes_L3 = 7;
  for (int i = 0; i < inp_nodes; i++)
  {
    basic_fc_nn.input_layer.push_back(0.0);
    basic_fc_nn.i_layer_delta.push_back(0.0);
  }
  for (int i = 0; i < out_nodes; i++)
  {
    basic_fc_nn.output_layer.push_back(0.0);
    basic_fc_nn.target_layer.push_back(0.0);
    basic_fc_nn.o_layer_delta.push_back(0.0); // Not need for End block
  }
  basic_fc_nn.set_nr_of_hidden_layers(hid_layers);
  basic_fc_nn.set_nr_of_hidden_nodes_on_layer_nr(hid_nodes_L1);
  basic_fc_nn.set_nr_of_hidden_nodes_on_layer_nr(hid_nodes_L2);
  // basic_fc_nn.set_nr_of_hidden_nodes_on_layer_nr(hid_nodes_L3);
  //  Note that set_nr_of_hidden_nodes_on_layer_nr() cal must be exactly same number as the set_nr_of_hidden_layers(hid_layers)
  //============ Neural Network Size setup is finnish ! ==================

  //=== Now setup the hyper parameters of the Neural Network ====
  basic_fc_nn.momentum = 0.9;
  basic_fc_nn.learning_rate = 0.001;
  basic_fc_nn.dropout_proportion = 0.15;
  basic_fc_nn.fix_leaky_proportion = 0.05;
  double init_random_weight_propotion = 0.05;
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

  const int training_epocs = 10000; // One epocs go through the hole data set once
  const int save_after_epcs = 10;
  int save_epoc_counter = 0;
  const int training_dataset_size = MNIST_nr_of_img_p_batch;
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
  for (int i = 0; i < training_dataset_size; i++)
  {
    training_order_list.push_back(0);
  }
  for (int i = 0; i < verify_dataset_size; i++)
  {
    verify_order_list.push_back(0);
  }
  verify_order_list = fisher_yates_shuffle(verify_order_list);
  training_order_list = fisher_yates_shuffle(training_order_list);

  vector<double> dummy_one_target_data_point;
  vector<double> dummy_one_training_data_point;
  for (int i = 0; i < out_nodes; i++)
  {
    dummy_one_target_data_point.push_back(0.0);
  }
  for (int i = 0; i < inp_nodes; i++)
  {
    dummy_one_training_data_point.push_back(0.0);
  }
  for (int i = 0; i < training_dataset_size; i++)
  {
    training_target_data.push_back(dummy_one_target_data_point);
    training_input_data.push_back(dummy_one_training_data_point);
  }
  for (int i = 0; i < verify_dataset_size; i++)
  {
    verify_target_data.push_back(dummy_one_target_data_point);
    verify_input_data.push_back(dummy_one_training_data_point);
  }

    int MNIST_nr = 0;
    srand (static_cast <unsigned> (time(NULL)));//Seed the randomizer
  // Start traning
  int do_verify_if_best_trained = 0;
  int stop_training = 0;
  int  target_lable = 0;
  char mnist_d = 0;
  double mnist_f = 0.0;
  unsigned char mnist_uchar = 0.0;
  for (int i = 0; i < training_dataset_size; i++)
  {
    for(int j=0;j<MNIST_pix_size;j++)
    {
    mnist_d = MNIST_data[(MNIST_pix_size * i) + j];
  //  cout << " mnist_d = " << (int)mnist_d  << endl;
    mnist_uchar = (~(-mnist_d) + 1);
  //  cout << " mnist_uchar = " <<  (int)mnist_uchar  << endl;
    mnist_f = ((double)mnist_uchar)/256.0;

  //  cout << "mnist_f = " << mnist_f  << endl;
    training_input_data[i][j] = mnist_f;       // 0..1
    }
    target_lable = ((int) MNIST_lable[i]);
    for(int j=0;j<out_nodes;j++)
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


  target_lable = ((int) MNIST_lable[MNIST_nr]);
  cout << "target_lable = " << target_lable  << endl;



//=================

  for (int epc = 0; epc < training_epocs; epc++)
  {
    if (stop_training == 1)
    {
      break;
    }
    cout << "Epoch ----" << epc << endl;
    cout << "input node --- [0] = " << basic_fc_nn.input_layer[0] << endl;

    //============ Traning ==================

    MNIST_nr = (int)(rand() % MNIST_nr_of_img_p_batch);

    // verify_order_list = fisher_yates_shuffle(verify_order_list);
    training_order_list = fisher_yates_shuffle(training_order_list);
    basic_fc_nn.loss = 0.0;
    int correct_classify_cnt = 0;
    for (int i = 0; i < training_dataset_size; i++)
    {
      for (int j = 0; j < inp_nodes; j++)
      {
        basic_fc_nn.input_layer[j] = training_input_data[training_order_list[i]][j];
      }
      basic_fc_nn.forward_pass();
      double highest_output = 0.0;
      int highest_out_class = 0;
      int taget = 0;
      for (int j = 0; j < out_nodes; j++)
      {
        basic_fc_nn.target_layer[j] = training_target_data[training_order_list[i]][j];
        if(basic_fc_nn.target_layer[j] > 0.5)
        {
          taget = j;
        }
        if(basic_fc_nn.output_layer[j] > highest_output)
        {
          highest_output = basic_fc_nn.output_layer[j];
          highest_out_class = j;
        }
      }
      basic_fc_nn.backpropagtion_and_update();
      //cout << "highest_out_class = " << highest_out_class << "taget = " << taget <<  endl;
      if(highest_out_class == taget)
      {
        correct_classify_cnt++;
      }
    }
    cout << "Epoch " << epc << endl;
    cout << "input node [0] = " << basic_fc_nn.input_layer[0] << endl;
    for (int k = 0; k < out_nodes; k++)
    {
      cout << "Output node [" << k << "] = " << basic_fc_nn.output_layer[k] << "  Target node [" << k << "] = " << basic_fc_nn.target_layer[k] << endl;
    }

    cout << "Training loss = " << basic_fc_nn.loss << endl;
    cout << "correct_classify_cnt = " << correct_classify_cnt << endl;
    double correct_ratio = (((double)correct_classify_cnt) * 100.0)/((double)training_dataset_size);
    cout << "correct_ratio = " << correct_ratio << endl;
    if (save_epoc_counter < save_after_epcs - 1)
    {
      save_epoc_counter++;
    }
    else
    {
      save_epoc_counter = 0;
      basic_fc_nn.save_weights(weight_filename);
    }
  }

  basic_fc_nn.~fc_m_resnet();
  basic_fc_nn.~fc_m_resnet();

  for (int i = 0; i < training_dataset_size; i++)
  {
    training_input_data[i].clear();
    training_target_data[i].clear();
  }
  training_input_data.clear();
  training_target_data.clear();
  for (int i = 0; i < verify_dataset_size; i++)
  {
    verify_input_data[i].clear();
    verify_target_data[i].clear();
  }
  training_order_list.clear();
  verify_order_list.clear();
  dummy_one_target_data_point.clear();
  dummy_one_training_data_point.clear();
}

vector<int> fisher_yates_shuffle(vector<int> table)
{
  int size = table.size();
  for (int i = 0; i < size; i++)
  {
    table[i] = i;
  }
  for (int i = size - 1; i > 0; i--)
  {
    int j = rand() % (i + 1);
    int temp = table[i];
    table[i] = table[j];
    table[j] = temp;
  }
  return table;
}

void check_file_exist(char *filename)
{
  FILE *fp2 = fopen(filename, "rb");
  if (fp2 == NULL)
  {
    cout << "Error " << filename << " file doesn't exist" << endl;
    ;
    cout << "Maybee download failure try restart program later download again" << endl;
    ;
    printf("Exit program\n");
    exit(0);
  }
  else
  {
    cout << "OK " << filename << " file does exist" << endl;
    ;
  }
  fclose(fp2);
}
int get_MNIST_file_size(void)
{
  char answer_character;
  int file_size = 0;
  FILE *fp2;
  if (use_MNIST_verify_set == 0)
  {
    fp2 = fopen("train-images-idx3-ubyte", "rb");
  }
  else
  {
    fp2 = fopen("t10k-images-idx3-ubyte", "rb");
  }
  if (fp2 == NULL)
  {
    if (use_MNIST_verify_set == 0)
    {
      puts("Error!  while opening file train-images-idx3-ubyte");
    }
    else
    {
      puts("Error! while opening file t10k-images-idx3-ubyte");
    }
    printf("Suggestions, download and extraxt dataset into program folder\n");
    printf("Do you want to download MNIST data set from web Y/N ?\n");
    printf("From web sites:\n");
    printf("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz\n");
    printf("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz\n");
    printf("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz\n");
    printf("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz\n");
    answer_character = getchar();
    // answer_character = getchar();
    if (answer_character == 'Y' || answer_character == 'y')
    {
      printf("remove old MNIST dataset .gz files\n");
      system("rm train-images-idx3-ubyte.gz");
      system("rm train-labels-idx1-ubyte.gz");
      system("rm t10k-images-idx3-ubyte.gz");
      system("rm t10k-labels-idx1-ubyte.gz");
      printf("Start download MNIST dataset .gz files\n");
      system("wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz");
      system("wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz");
      system("wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz");
      system("wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz");

      /* try to open file to read */
      sprintf(MNIST_filename, "train-images-idx3-ubyte.gz");
      check_file_exist(MNIST_filename);
      sprintf(MNIST_filename, "train-labels-idx1-ubyte.gz");
      check_file_exist(MNIST_filename);
      sprintf(MNIST_filename, "t10k-images-idx3-ubyte.gz");
      check_file_exist(MNIST_filename);
      sprintf(MNIST_filename, "t10k-labels-idx1-ubyte.gz");
      check_file_exist(MNIST_filename);

      printf("Do you want to unzip this files with gzip Y/N\n");
      answer_character = getchar();
      answer_character = getchar();

      if (answer_character == 'Y' || answer_character == 'y')
      {
        printf("Unzip MNIST datasets\n");
        system("gzip -d train-images-idx3-ubyte.gz");
        system("gzip -d train-labels-idx1-ubyte.gz");
        system("gzip -d t10k-images-idx3-ubyte.gz");
        system("gzip -d t10k-labels-idx1-ubyte.gz");
        printf("...\n");
      }
      else
      {
        printf("You now may need to unzip all mnist....gz files\n");
        printf("Try command:\n");
        printf("$ gzip -d train-images-idx3-ubyte.gz\n");
        printf("Exit program\n");
        exit(0);
      }
    }
    else
    {
      printf("OK you may need to download MNIST dataset\n");
      printf("Exit program\n");
      exit(0);
    }

    if (use_MNIST_verify_set == 0)
    {
      fp2 = fopen("train-images-idx3-ubyte", "rb");
    }
    else
    {
      fp2 = fopen("t10k-images-idx3-ubyte", "rb");
    }
    if (fp2 == NULL)
    {
      printf("exit program MNIST dataset failure to open\n");
      exit(0);
    }
    else
    {
      printf("OK! MNIST dataset daownloade and extracted\n");
    }
  }

  fseek(fp2, 0L, SEEK_END);
  file_size = ftell(fp2);
  printf("file_size %d\n", file_size);
  rewind(fp2);
  fclose(fp2);
  return file_size;
}

int get_MNIST_lable_file_size(void)
{
  int file_size = 0;
  FILE *fp2;
  if (use_MNIST_verify_set == 0)
  {
    fp2 = fopen("train-labels-idx1-ubyte", "rb");
  }
  else
  {
    fp2 = fopen("t10k-labels-idx1-ubyte", "rb");
  }

  if (fp2 == NULL)
  {
    if (use_MNIST_verify_set == 0)
    {
      puts("Error while opening file train-labels-idx1-ubyte");
    }
    else
    {
      puts("Error while opening file t10k-labels-idx1-ubyte");
    }
    exit(0);
  }

  fseek(fp2, 0L, SEEK_END);
  file_size = ftell(fp2);
  printf("file_size %d\n", file_size);
  rewind(fp2);
  fclose(fp2);
  return file_size;
}
