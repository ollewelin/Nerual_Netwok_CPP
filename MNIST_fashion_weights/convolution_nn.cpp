
#include <iostream>
#include <stdio.h>

#include "fc_m_resnet.hpp"
#include "convolution.hpp"
#include "load_mnist_dataset.hpp"
#include "batch_norm_layer.hpp"

#include <vector>
#include <time.h>

using namespace std;

#include <cstdlib>
#include <ctime>
#include <math.h>   // exp
#include <stdlib.h> // exit(0);

#include <cstdlib>
#include <chrono>

vector<int> fisher_yates_shuffle(vector<int> table);
int main()
{

    cout << "Convolution neural network under work..." << endl;
    // ======== create 2 convolution layer objects =========
    convolution conv_L1;
    batch_norm_layer L1_batch_norm;
    convolution conv_L2;
    batch_norm_layer L2_batch_norm;
    convolution conv_L3;
    batch_norm_layer L3_batch_norm;
    //======================================================
    L1_batch_norm.get_version();

    //=========== Neural Network size settings ==============
    fc_m_resnet fc_nn_end_block;
    string weight_filename_end;
    weight_filename_end = "end_block_weights.dat";
    string L1_kernel_k_weight_filename;
    L1_kernel_k_weight_filename = "L1_kernel_k.dat";
    string L1_kernel_b_weight_filename;
    L1_kernel_b_weight_filename = "L1_kernel_b.dat";
    string L2_kernel_k_weight_filename;
    L2_kernel_k_weight_filename = "L2_kernel_k.dat";
    string L2_kernel_b_weight_filename;
    L2_kernel_b_weight_filename = "L2_kernel_b.dat";
    string L3_kernel_k_weight_filename;
    L2_kernel_k_weight_filename = "L3_kernel_k.dat";
    string L3_kernel_b_weight_filename;
    L2_kernel_b_weight_filename = "L3_kernel_b.dat";

    string L1_batch_n_weight_filename;
    L1_batch_n_weight_filename = "L1_batch_norm.dat";
    string L2_batch_n_weight_filename;
    L2_batch_n_weight_filename = "L2_batch_norm.dat";
    string L3_batch_n_weight_filename;
    L3_batch_n_weight_filename = "L3_batch_norm.dat";

    fc_nn_end_block.get_version();

    fc_nn_end_block.block_type = 2;
    fc_nn_end_block.use_softmax = 1;
    fc_nn_end_block.activation_function_mode = 2;
    fc_nn_end_block.use_skip_connect_mode = 0; // 1 for residual network architetcture
    fc_nn_end_block.use_dropouts = 1;
    fc_nn_end_block.dropout_proportion = 0.2;

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
    int input_channels = 1;     //=== one channel MNIST dataset is used ====
    conv_L1.set_kernel_size(5); // Odd number
    conv_L1.set_stride(2);
    conv_L1.set_in_tensor(data_size_one_sample_one_channel, input_channels); // data_size_one_sample_one_channel, input channels
    conv_L1.set_out_tensor(30);                                              // output channels
    conv_L1.top_conv = 1;

    //========= L1 convolution (vectors) all tensor size for convolution object is finnish =============

    //==== Set up convolution layers ===========
    cout << "conv_L2 setup:" << endl;
    conv_L2.set_kernel_size(5); // Odd number
    conv_L2.set_stride(1);
    conv_L2.set_in_tensor((conv_L1.output_tensor[0].size() * conv_L1.output_tensor[0].size()), conv_L1.output_tensor.size()); // data_size_one_sample_one_channel, input channels
    conv_L2.set_out_tensor(60);                                                                                               // output channels

    cout << "conv_L3 setup:" << endl;
    conv_L3.set_kernel_size(5); // Odd number
    conv_L3.set_stride(2);
    conv_L3.set_in_tensor((conv_L2.output_tensor[0].size() * conv_L2.output_tensor[0].size()), conv_L2.output_tensor.size()); // data_size_one_sample_one_channel, input channels
    conv_L3.set_out_tensor(50);                                                                                               // output channels

    const int end_inp_nodes = (conv_L3.output_tensor[0].size() * conv_L3.output_tensor[0].size()) * conv_L3.output_tensor.size();
    cout << "end_inp_nodes = " << end_inp_nodes << endl;
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
    for (int i = 0; i < data_size_one_sample_one_channel; i++)
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

    cout << endl;
    cout << endl;
    cout << "Structure of the Fully Connected Neural Network is this: " << endl;

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
    const int batch_size = 64;
    const double learning_rate_end = 0.0005;
    fc_nn_end_block.momentum = 0.01;
    fc_nn_end_block.learning_rate = learning_rate_end;
    conv_L1.learning_rate = 0.001;
    conv_L1.momentum = 0.05;
    conv_L2.learning_rate = 0.001;
    conv_L2.momentum = 0.05;
    conv_L3.learning_rate = 0.001;
    conv_L3.momentum = 0.05;
    L1_batch_norm.lr = 0.001;
    L2_batch_norm.lr = 0.001;
    L3_batch_norm.lr = 0.001;
    L1_batch_norm.set_up_tensors(batch_size, conv_L1.output_tensor.size(), conv_L1.output_tensor[0].size(), conv_L1.output_tensor[0].size());
    L2_batch_norm.set_up_tensors(batch_size, conv_L2.output_tensor.size(), conv_L2.output_tensor[0].size(), conv_L2.output_tensor[0].size());
    L3_batch_norm.set_up_tensors(batch_size, conv_L3.output_tensor.size(), conv_L3.output_tensor[0].size(), conv_L3.output_tensor[0].size());
    conv_L1.activation_function_mode = 2;
    conv_L2.activation_function_mode = 2;
    conv_L3.activation_function_mode = 2;

    int training_batches = training_dataset_size / batch_size;
    int training_dsize_fit_batch = training_batches * batch_size;
    int verify_batches = verify_dataset_size / batch_size;
    int verify_dsize_fit_batch = verify_batches * batch_size;
    cout << "batch_size = " << batch_size << endl;
    cout << "training_dataset_size = " << training_dataset_size << endl;
    cout << "training_batches = " << training_batches << endl;
    cout << "training_dsize_fit_batch = " << training_dsize_fit_batch << endl;
    cout << "verify_dataset_size = " << verify_dataset_size << endl;
    cout << "verify_batches = " << verify_batches << endl;
    cout << "verify_dsize_fit_batch = " << verify_dsize_fit_batch << endl;

    char answer;
    double init_fc_random_weight_propotion = 0.001;
    double init_conv_random_weight_propotion = 0.05;
    cout << "Do you want to load kernel weights from saved weight file = Y/N " << endl;
    cin >> answer;
    if (answer == 'Y' || answer == 'y')
    {
        conv_L1.load_weights(L1_kernel_k_weight_filename, L1_kernel_b_weight_filename);
        conv_L2.load_weights(L2_kernel_k_weight_filename, L2_kernel_b_weight_filename);
        conv_L3.load_weights(L3_kernel_k_weight_filename, L3_kernel_b_weight_filename);
        L1_batch_norm.load_weights(L1_batch_n_weight_filename);
        L2_batch_norm.load_weights(L2_batch_n_weight_filename);
        L3_batch_norm.load_weights(L3_batch_n_weight_filename);
        cout << "Do you want to randomize fully connected layers Y or N load weights  = Y/N " << endl;
        cin >> answer;
        if (answer == 'Y' || answer == 'y')
        {
            fc_nn_end_block.randomize_weights(init_fc_random_weight_propotion);
        }
        else
        {
            fc_nn_end_block.load_weights(weight_filename_end);
        }
    }
    else
    {
        fc_nn_end_block.randomize_weights(init_fc_random_weight_propotion);
        conv_L1.randomize_weights(init_conv_random_weight_propotion);
        conv_L2.randomize_weights(init_conv_random_weight_propotion);
        conv_L3.randomize_weights(init_conv_random_weight_propotion);
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
    int print_after = 4999;
    int print_cnt = print_after;
    int one_side = sqrt(data_size_one_sample_one_channel);
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

        //   for (int i = 0; i < training_dataset_size; i++)
        //   for (int i = 0; i < training_dsize_fit_batch; i++)
        for (int batch_cnt = 0; batch_cnt < training_batches; batch_cnt++)
        {
            for (int samp_cnt = 0; samp_cnt < batch_size; samp_cnt++)
            {
                int i = batch_cnt * batch_size + samp_cnt;
                for (int ic = 0; ic < input_channels; ic++)
                {
                    for (int yi = 0; yi < one_side; yi++)
                    {
                        for (int xi = 0; xi < one_side; xi++)
                        {
                            conv_L1.input_tensor[ic][yi][xi] = training_input_data[training_order_list[i]][ic * input_channels * one_side * one_side + one_side * yi + xi];
                        }
                    }
                }
                conv_L1.conv_forward1();
                L1_batch_norm.input_tensor[samp_cnt] = conv_L1.output_tensor;
            }
            L1_batch_norm.forward_batch();//Run througe the hole batch
            for (int samp_cnt = 0; samp_cnt < batch_size; samp_cnt++)
            {
                conv_L2.input_tensor = L1_batch_norm.output_tensor[samp_cnt];
                conv_L2.conv_forward1();
                L2_batch_norm.input_tensor[samp_cnt] = conv_L2.output_tensor;
            }
            conv_L2.conv_forward1();
            for (int samp_cnt = 0; samp_cnt < batch_size; samp_cnt++)
            {
                conv_L3.input_tensor = conv_L2.output_tensor;
                conv_L3.conv_forward1();
                L3_batch_norm.input_tensor[samp_cnt] = conv_L3.output_tensor;
            }
            conv_L3.conv_forward1();
            for (int samp_cnt = 0; samp_cnt < batch_size; samp_cnt++)
            {

                int L3_out_one_side = conv_L3.output_tensor[0].size();
                int L3_out_ch = conv_L3.output_tensor.size();
                for (int oc = 0; oc < L3_out_ch; oc++)
                {
                    for (int yi = 0; yi < L3_out_one_side; yi++)
                    {
                        for (int xi = 0; xi < L3_out_one_side; xi++)
                        {
                            //fc_nn_end_block.input_layer[oc * L3_out_one_side * L3_out_one_side + yi * L3_out_one_side + xi] = conv_L3.output_tensor[oc][yi][xi];
                            fc_nn_end_block.input_layer[oc * L3_out_one_side * L3_out_one_side + yi * L3_out_one_side + xi] = L3_batch_norm.output_tensor[samp_cnt][oc][yi][xi];
                        }
                    }
                }
                // Start Forward pass fully connected network
                fc_nn_end_block.forward_pass();
                // Forward pass though fully connected network
                int i = batch_cnt * batch_size + samp_cnt;
                if (print_cnt > 0)
                {
                    print_cnt--;
                }
                else
                {
                    cout << "convolution L1 L2 L3 done, i = " << i << endl;
                    print_cnt = print_after;
                }
                for (int j = 0; j < end_out_nodes; j++)
                {
                    fc_nn_end_block.target_layer[j] = training_target_data[training_order_list[i]][j];
                }
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
                // cout << "highest_out_class = " << highest_out_class << "taget = " << taget <<  endl;
                if (highest_out_class == target)
                {
                    correct_classify_cnt++;
                }

                //============ Begin Backpropagation ===============
                fc_nn_end_block.backpropagtion_and_update();
                for (int oc = 0; oc < L3_out_ch; oc++)
                {
                    for (int yi = 0; yi < L3_out_one_side; yi++)
                    {
                        for (int xi = 0; xi < L3_out_one_side; xi++)
                        {
                            L3_batch_norm.o_tensor_delta[samp_cnt][oc][yi][xi] = fc_nn_end_block.i_layer_delta[oc * L3_out_one_side * L3_out_one_side + yi * L3_out_one_side + xi];
                        }
                    }
                }
            }
            //============ Continue Backpropagation thorugh batch norm and convolution layer ===============
            L3_batch_norm.backprop_batch();
            for (int samp_cnt = 0; samp_cnt < batch_size; samp_cnt++)
            {
                conv_L3.o_tensor_delta = L3_batch_norm.i_tensor_delta[samp_cnt];
                conv_L3.conv_backprop();
                conv_L3.conv_update_weights();
                L2_batch_norm.o_tensor_delta[samp_cnt] = conv_L3.i_tensor_delta;
            }
            L2_batch_norm.backprop_batch();
            for (int samp_cnt = 0; samp_cnt < batch_size; samp_cnt++)
            {
                conv_L2.o_tensor_delta = L2_batch_norm.i_tensor_delta[samp_cnt];
                conv_L2.conv_backprop();
                conv_L2.conv_update_weights();
                L1_batch_norm.o_tensor_delta[samp_cnt] = conv_L2.i_tensor_delta;
            }
            L1_batch_norm.backprop_batch();
            for (int samp_cnt = 0; samp_cnt < batch_size; samp_cnt++)
            {
                conv_L1.o_tensor_delta = L1_batch_norm.i_tensor_delta[samp_cnt];
                conv_L1.conv_backprop();
                conv_L1.conv_update_weights();
            }
            //============ Finnised Backpropagation thorugh batch norm and convolution layer ===============
            if(batch_cnt % 10 == 0)
            {
               cout << "Batch counter = " << batch_cnt << " Of number of total batches = " << training_batches << endl;
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
        double correct_ratio = (((double)correct_classify_cnt) * 100.0) / ((double)training_dsize_fit_batch);
        cout << "correct_ratio = " << correct_ratio << endl;


        //TODO... add batch normalizer
        //=========== verify ===========
        print_cnt = 0;
        if (do_verify_if_best_trained == 1)
        {
            fc_nn_end_block.dropout_proportion = 0.0;
            verify_order_list = fisher_yates_shuffle(verify_order_list);
            fc_nn_end_block.loss = 0.0;
            correct_classify_cnt = 0;
            for (int i = 0; i < verify_dsize_fit_batch; i++)
            {
                for (int ic = 0; ic < input_channels; ic++)
                {
                    for (int yi = 0; yi < one_side; yi++)
                    {
                        for (int xi = 0; xi < one_side; xi++)
                        {

                            conv_L1.input_tensor[ic][yi][xi] = verify_input_data[verify_order_list[i]][ic * input_channels * one_side * one_side + one_side * yi + xi];
                        }
                    }
                }
                conv_L1.conv_forward1();
                conv_L2.input_tensor = conv_L1.output_tensor;
                conv_L2.conv_forward1();
                conv_L3.input_tensor = conv_L2.output_tensor;
                conv_L3.conv_forward1();
                int L3_out_one_side = conv_L3.output_tensor[0].size();
                int L3_out_ch = conv_L3.output_tensor.size();
                for (int oc = 0; oc < L3_out_ch; oc++)
                {
                    for (int yi = 0; yi < L3_out_one_side; yi++)
                    {
                        for (int xi = 0; xi < L3_out_one_side; xi++)
                        {
                            fc_nn_end_block.input_layer[oc * L3_out_one_side * L3_out_one_side + yi * L3_out_one_side + xi] = conv_L3.output_tensor[oc][yi][xi];
                        }
                    }
                }
                // Start Forward pass fully connected network
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
            double correct_ratio = (((double)correct_classify_cnt) * 100.0) / ((double)verify_dsize_fit_batch);
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
                conv_L1.save_weights(L1_kernel_k_weight_filename, L1_kernel_b_weight_filename);
                conv_L2.save_weights(L2_kernel_k_weight_filename, L2_kernel_b_weight_filename);
                conv_L2.save_weights(L3_kernel_k_weight_filename, L3_kernel_b_weight_filename);
                L1_batch_norm.save_weights(L1_batch_n_weight_filename);
                L2_batch_norm.save_weights(L2_batch_n_weight_filename);
                L3_batch_norm.save_weights(L3_batch_n_weight_filename);
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
