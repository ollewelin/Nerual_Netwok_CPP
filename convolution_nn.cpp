
#include <iostream>
#include <stdio.h>

#include "fc_m_resnet.hpp"
#include "convolution.hpp"
#include "load_mnist_dataset.hpp"

#include <vector>
#include <time.h>

using namespace std;

#include <cstdlib>
#include <ctime>
#include <math.h>   // exp
#include <stdlib.h> // exit(0);


#include <cstdlib>

vector<int> fisher_yates_shuffle(vector<int> table);
int main()
{
    cout << "Convolution neural network under work..." << endl;
    // ======== create 2 convolution layer objects =========
    convolution conv_L1;
    convolution conv_L2;
    // convolution conv_L3;
    //======================================================
    srand(time(NULL));
    char answer;

    //=========== Test Neural Network size settings ==============
    fc_m_resnet fc_nn_end_block;
    string weight_filename_end;
    weight_filename_end = "end_block_weights.dat";
    fc_nn_end_block.get_version();
    

    fc_nn_end_block.block_type = 2;
    fc_nn_end_block.use_softmax = 1;
    fc_nn_end_block.activation_function_mode = 0;
    fc_nn_end_block.use_skip_connect_mode = 0; // 1 for residual network architetcture
    fc_nn_end_block.use_dopouts = 0;

    load_mnist_dataset l_mnist_data;
    vector<vector<double>> training_target_data;
    vector<vector<double>> training_input_data;
    vector<vector<double>> verify_target_data;
    vector<vector<double>> verify_input_data;
    int data_size_one_sample_one_channel = l_mnist_data.get_one_sample_data_size();
    int training_dataset_size = l_mnist_data.get_training_data_set_size();
    int verify_dataset_size = l_mnist_data.get_verify_data_set_size();
    
    cout << endl;
    conv_L1.get_version();
    //==== Set up convolution layers ===========
    cout << "conv_L1 setup:" << endl;
    int input_channels = 1;//=== one channel MNIST dataset is used ====
    conv_L1.set_in_tensor(data_size_one_sample_one_channel, input_channels);//data_size_one_sample_one_channel, input channels
    conv_L1.set_kernel_size(3);//Odd number
    conv_L1.set_stride(2);
    conv_L1.set_out_tensor(30);//output channels
    conv_L1.output_tensor.size();
    //========= L1 convolution (vectors) all tensor size for convolution object is finnish =============


    //========================================
    const int end_inp_nodes = data_size_one_sample_one_channel;
    const int end_hid_layers = 2;
    const int end_hid_nodes_L1 = 200;
    const int end_hid_nodes_L2 = 50;
    const int end_out_nodes = 10;

    vector<double> dummy_one_target_data_point;
    vector<double> dummy_one_training_data_point;
    for (int i = 0; i < end_out_nodes; i++)
    {
        dummy_one_target_data_point.push_back(0.0);
    }
    for (int i = 0; i < end_inp_nodes; i++)
    {
        dummy_one_training_data_point.push_back(0.0);
    }
    for (int i = 0; i < training_dataset_size; i++)
    {
        training_target_data.push_back(dummy_one_target_data_point);
        training_input_data.push_back(dummy_one_training_data_point);
    }
    for (int i = 0; i < l_mnist_data.get_verify_data_set_size(); i++)
    {
        verify_target_data.push_back(dummy_one_target_data_point);
        verify_input_data.push_back(dummy_one_training_data_point);
    }

    training_input_data = l_mnist_data.load_input_data(training_input_data, 0);   // last argument 0 = training, 1 = verify
    training_target_data = l_mnist_data.load_lable_data(training_target_data, 0); // last argument 0 = training, 1 = verify
    verify_input_data = l_mnist_data.load_input_data(verify_input_data, 1);       // last argument 0 = training, 1 = verify
    verify_target_data = l_mnist_data.load_lable_data(verify_target_data, 1);     // last argument 0 = training, 1 = verify

    l_mnist_data.~load_mnist_dataset();

    for (int i = 0; i < end_inp_nodes; i++)
    {
        fc_nn_end_block.input_layer.push_back(0.0);
        fc_nn_end_block.i_layer_delta.push_back(0.0);
    }

    for (int i = 0; i < end_out_nodes; i++)
    {
        fc_nn_end_block.output_layer.push_back(0.0);
        fc_nn_end_block.target_layer.push_back(0.0);
    }
    fc_nn_end_block.set_nr_of_hidden_layers(end_hid_layers);
    fc_nn_end_block.set_nr_of_hidden_nodes_on_layer_nr(end_hid_nodes_L1);
    fc_nn_end_block.set_nr_of_hidden_nodes_on_layer_nr(end_hid_nodes_L2);
    //  Note that set_nr_of_hidden_nodes_on_layer_nr() cal must be exactly same number as the set_nr_of_hidden_layers(end_hid_layers)
    //============ Neural Network Size setup is finnish ! ==================

    //=== Now setup the hyper parameters of the Neural Network ====

    const double learning_rate_end = 0.01;

    fc_nn_end_block.momentum = 0.3;
    fc_nn_end_block.learning_rate = learning_rate_end;

    double init_random_weight_propotion = 0.05;
    cout << "Do you want to load weights from saved weight file = Y/N " << endl;
    cin >> answer;
    if (answer == 'Y' || answer == 'y')
    {
        fc_nn_end_block.load_weights(weight_filename_end);
    }
    else
    {
        fc_nn_end_block.randomize_weights(init_random_weight_propotion);
    }

    const int training_epocs = 10000; // One epocs go through the hole data set once
    const int save_after_epcs = 10;
    int save_epoc_counter = 0;

    // const int verify_after_x_nr_epocs = 10;
    // int verify_after_epc_cnt = 0;
    double best_training_loss = 1000000000;
    double best_verify_loss = best_training_loss;
    double train_loss = best_training_loss;
    double verify_loss = best_training_loss;

    const double stop_training_when_verify_rise_propotion = 0.04;
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

    srand(static_cast<unsigned>(time(NULL))); // Seed the randomizer

    int do_verify_if_best_trained = 0;
    int stop_training = 0;
    // Start traning
    //=================

    for (int epc = 0; epc < training_epocs; epc++)
    {
        if (stop_training == 1)
        {
            break;
        }
        cout << "Epoch ----" << epc << endl;
        cout << "input node --- [0] = " << fc_nn_end_block.input_layer[0] << endl;

        //============ Traning ==================

        training_order_list = fisher_yates_shuffle(training_order_list);
        fc_nn_end_block.loss = 0.0;
        int correct_classify_cnt = 0;
        fc_nn_end_block.dropout_proportion = 0.20;

        for (int i = 0; i < training_dataset_size; i++)
        {
            // Start Forward pass fully connected network
            for (int j = 0; j < end_inp_nodes; j++)
            {
                fc_nn_end_block.input_layer[j] = training_input_data[training_order_list[i]][j];
            }
            fc_nn_end_block.forward_pass();
            // Forward pass though fully connected network

            double highest_output = 0.0;
            int highest_out_class = 0;
            int target = 0;
            for (int j = 0; j < end_out_nodes; j++)
            {
                fc_nn_end_block.target_layer[j] = training_target_data[training_order_list[i]][j];
                if (fc_nn_end_block.target_layer[j] > 0.5)
                {
                    target = j;
                }
                if (fc_nn_end_block.output_layer[j] > highest_output)
                {
                    highest_output = fc_nn_end_block.output_layer[j];
                    highest_out_class = j;
                }
            }

            fc_nn_end_block.backpropagtion_and_update();
            for (int j = 0; j < end_out_nodes; j++)
            {
                // TODO...
                // Backpropagation to convolution network  = fc_nn_end_block.i_layer_delta[j];
            }

            // cout << "highest_out_class = " << highest_out_class << "taget = " << taget <<  endl;
            if (highest_out_class == target)
            {
                correct_classify_cnt++;
            }
        }
        cout << "Epoch " << epc << endl;
        cout << "input node [0] = " << fc_nn_end_block.input_layer[0] << endl;
        for (int k = 0; k < end_out_nodes; k++)
        {
            cout << "Output node [" << k << "] = " << fc_nn_end_block.output_layer[k] << "  Target node [" << k << "] = " << fc_nn_end_block.target_layer[k] << endl;
        }
        train_loss = fc_nn_end_block.loss;
        do_verify_if_best_trained = 1;
        /*
           if(best_training_loss > train_loss)
           {
             best_training_loss = train_loss;
             do_verify_if_best_trained = 1;
           }
           else
           {
             do_verify_if_best_trained = 0;
           }
       */
        cout << "Training loss = " << train_loss << endl;
        cout << "correct_classify_cnt = " << correct_classify_cnt << endl;
        double correct_ratio = (((double)correct_classify_cnt) * 100.0) / ((double)training_dataset_size);
        cout << "correct_ratio = " << correct_ratio << endl;

        //=========== verify ===========

        if (do_verify_if_best_trained == 1)
        {
            fc_nn_end_block.dropout_proportion = 0.0;
            verify_order_list = fisher_yates_shuffle(verify_order_list);
            fc_nn_end_block.loss = 0.0;
            correct_classify_cnt = 0;
            for (int i = 0; i < verify_dataset_size; i++)
            {

                // Start Forward pass fully connected network
                for (int j = 0; j < end_inp_nodes; j++)
                {
                    fc_nn_end_block.input_layer[j] = verify_input_data[verify_order_list[i]][j];
                }
                fc_nn_end_block.forward_pass();
                // Forward pass though fully connected network

                double highest_output = 0.0;
                int highest_out_class = 0;
                int target = 0;
                for (int j = 0; j < end_out_nodes; j++)
                {
                    fc_nn_end_block.target_layer[j] = verify_target_data[verify_order_list[i]][j];
                    if (fc_nn_end_block.target_layer[j] > 0.5)
                    {
                        target = j;
                    }
                    if (fc_nn_end_block.output_layer[j] > highest_output)
                    {
                        highest_output = fc_nn_end_block.output_layer[j];
                        highest_out_class = j;
                    }
                }

                fc_nn_end_block.only_loss_calculation();
                // cout << "highest_out_class = " << highest_out_class << "taget = " << taget <<  endl;
                if (highest_out_class == target)
                {
                    correct_classify_cnt++;
                }
            }
            for (int k = 0; k < end_out_nodes; k++)
            {
                cout << "Output node [" << k << "] = " << fc_nn_end_block.output_layer[k] << "  Target node [" << k << "] = " << fc_nn_end_block.target_layer[k] << endl;
            }
            verify_loss = fc_nn_end_block.loss;
            cout << "Verify loss = " << verify_loss << endl;
            cout << "Verify correct_classify_cnt = " << correct_classify_cnt << endl;
            double correct_ratio = (((double)correct_classify_cnt) * 100.0) / ((double)verify_dataset_size);
            cout << "Verify correct_ratio = " << correct_ratio << endl;
            if (verify_loss > (best_verify_loss + stop_training_when_verify_rise_propotion * best_verify_loss))
            {
                cout << "Verfy loss increase !! " << endl;
                cout << "best_verify_loss = " << best_verify_loss << endl;
                // stop_training = 1;
                // break;
            }
            if (verify_loss < best_verify_loss)
            {
                best_verify_loss = verify_loss;
                fc_nn_end_block.save_weights(weight_filename_end);
            }

            //=========== verify finnish ====
        }

        if (save_epoc_counter < save_after_epcs - 1)
        {
            save_epoc_counter++;
        }
        else
        {
            save_epoc_counter = 0;
            //  fc_nn_top_block.save_weights(weight_filename_top);
        }
    }

    fc_nn_end_block.~fc_m_resnet();
    fc_nn_end_block.~fc_m_resnet();

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
