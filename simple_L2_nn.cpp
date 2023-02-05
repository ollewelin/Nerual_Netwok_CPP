#include "simple_L2_nn.hpp"
#include <iostream>
#include <fstream>
#include <stdlib.h> // for exit(0)
#include <math.h>
#include <time.h>

using namespace std;

simple_L2_nn::simple_L2_nn()
{
    cout << "Constructor simple_L2_nn" << endl;
    input_nodes = 784;
    hidden_L1_nodes = 300;
    hidden_L2_nodes = 15;
    output_nodes = 10;
    use_softmax = 0;
    filename_hid_L1 = "hid_weight_L1.dat";
    filename_hid_L2 = "hid_weight_L2.dat";
    filename_out = "out_weight.dat";

    vector<double> dummy_from_inp_node;
    dummy_from_inp_node.push_back(0.0); // Add one for Bias node
    for (int i = 0; i < input_nodes; i++)
    {
        input_layer.push_back(0.0);
        dummy_from_inp_node.push_back(0.0);
    }
    vector<double> dummy_from_hid_L1_node;
    dummy_from_hid_L1_node.push_back(0.0); // Add one for Bias node
    for (int i = 0; i < hidden_L1_nodes; i++)
    {
        hidden_L1_layer.push_back(0.0);
        hidden_L1_delta.push_back(0.0);
        dummy_from_hid_L1_node.push_back(0.0);
        hid_L1_weights.push_back(dummy_from_inp_node);
        hid_L1_change_weights.push_back(dummy_from_inp_node);
    }
    vector<double> dummy_from_hid_L2_node;
    dummy_from_hid_L2_node.push_back(0.0); // Add one for Bias node
    for (int i = 0; i < hidden_L2_nodes; i++)
    {
        hidden_L2_layer.push_back(0.0);
        hidden_L2_delta.push_back(0.0);
        dummy_from_hid_L2_node.push_back(0.0);
        hid_L2_weights.push_back(dummy_from_hid_L1_node);
        hid_L2_change_weights.push_back(dummy_from_hid_L1_node);
    }

    for (int i = 0; i < output_nodes; i++)
    {
        output_layer.push_back(0.0);
        target_layer.push_back(0.0);
        output_delta.push_back(0.0);
        out_weights.push_back(dummy_from_hid_L2_node);
        out_change_weights.push_back(dummy_from_hid_L2_node);
    }
}

simple_L2_nn::~simple_L2_nn()
{
    cout << "Destructor simple_L2_nn" << endl;
}

void simple_L2_nn::randomize_weights(double rand_proportion)
{
    cout << "Randomize weights 3D vector all weights of simple_L2_nn object...." << endl;
    int nodes_on_hid_layer = hid_L1_weights.size();
    for (int n_cnt = 0; n_cnt < nodes_on_hid_layer; n_cnt++)
    {
        int weights_to_this_node = hid_L1_weights[0].size();
        for (int w_cnt = 0; w_cnt < weights_to_this_node; w_cnt++)//
        {
            hid_L1_weights[n_cnt][w_cnt] = ((((double)rand() / RAND_MAX) - 0.5) * rand_proportion);
        }
    }
    nodes_on_hid_layer = hid_L2_weights.size();
    for (int n_cnt = 0; n_cnt < nodes_on_hid_layer; n_cnt++)
    {
        int weights_to_this_node = hid_L2_weights[0].size();
        for (int w_cnt = 0; w_cnt < weights_to_this_node; w_cnt++)
        {
            hid_L2_weights[n_cnt][w_cnt] = ((((double)rand() / RAND_MAX) - 0.5) * rand_proportion);
        }
    }
    int nodes_on_out_layer = out_weights.size();
    for (int n_cnt = 0; n_cnt < nodes_on_out_layer; n_cnt++)
    {
        int weights_to_this_node = out_weights[0].size();
        for (int w_cnt = 0; w_cnt < weights_to_this_node; w_cnt++)
        {
            out_weights[n_cnt][w_cnt] = ((((double)rand() / RAND_MAX) - 0.5) * rand_proportion);
        }
    }
    cout << "Randomize weights is DONE!" << endl;
}

const int precision_to_text_file = 16;
void simple_L2_nn::save_weights(void)
{
    cout << "Save data weights ..." << endl;
    ofstream outputFile;
    outputFile.precision(precision_to_text_file);
    outputFile.open(filename_hid_L1);
    int nodes_on_this_layer = hid_L1_weights.size();
    for (int n_cnt = 0; n_cnt < nodes_on_this_layer; n_cnt++)
    {
        int weights_to_this_node = hid_L1_weights[0].size();
        for (int w_cnt = 0; w_cnt < weights_to_this_node; w_cnt++)
        {
            outputFile << hid_L1_weights[n_cnt][w_cnt] << endl;
        }
    }
    outputFile.close();

    outputFile.precision(precision_to_text_file);
    outputFile.open(filename_hid_L2);
    nodes_on_this_layer = hid_L2_weights.size();
    for (int n_cnt = 0; n_cnt < nodes_on_this_layer; n_cnt++)
    {
        int weights_to_this_node = hid_L2_weights[0].size();
        for (int w_cnt = 0; w_cnt < weights_to_this_node; w_cnt++)
        {
            outputFile << hid_L2_weights[n_cnt][w_cnt] << endl;
        }
    }
    outputFile.close();


    outputFile.precision(precision_to_text_file);
    outputFile.open(filename_out);
    nodes_on_this_layer = out_weights.size();
    for (int n_cnt = 0; n_cnt < nodes_on_this_layer; n_cnt++)
    {
        int weights_to_this_node = out_weights[0].size();
        for (int w_cnt = 0; w_cnt < weights_to_this_node; w_cnt++)
        {
            outputFile << out_weights[n_cnt][w_cnt] << endl;
        }
    }
    outputFile.close();
    cout << "Save data finnish !" << endl;
}
void simple_L2_nn::load_weights(void)
{
    cout << "Load data weights ..." << endl;
    ifstream inputFile;
    string filename_hid_L1 = "hid_weight_L1.dat";
    string filename_hid_L2 = "hid_weight_L2.dat";

    inputFile.precision(precision_to_text_file);
    inputFile.open(filename_hid_L1);
    double d_element = 0.0;
    int data_load_error = 0;
    int data_load_numbers = 0;
    int nodes_on_this_layer = hid_L1_weights.size();
    for (int n_cnt = 0; n_cnt < nodes_on_this_layer; n_cnt++)
    {
        int weights_to_this_node = hid_L1_weights[0].size();
        for (int w_cnt = 0; w_cnt < weights_to_this_node; w_cnt++)
        {
            if (inputFile >> d_element)
            {
                hid_L1_weights[n_cnt][w_cnt] = d_element;
                data_load_numbers++;
            }
            else
            {
                data_load_error = 1;
            }
            if (data_load_error != 0)
            {
                break;
            }
        }
        if (data_load_error != 0)
        {
            break;
        }
    }
    inputFile.close();
    if (data_load_error == 0)
    {
        cout << "Load data finnish !" << endl;
    }
    else
    {
        cout << "ERROR! weight file error have not sufficient amount of data to put into  hid_L1_weights[n_cnt][w_cnt] vector" << endl;
        cout << "Loaded this amout of data weights data_load_numbers = " << data_load_numbers << endl;
    }


    inputFile.precision(precision_to_text_file);
    inputFile.open(filename_hid_L2);
    d_element = 0.0;
    data_load_error = 0;
    data_load_numbers = 0;
    nodes_on_this_layer = hid_L2_weights.size();
    for (int n_cnt = 0; n_cnt < nodes_on_this_layer; n_cnt++)
    {
        int weights_to_this_node = hid_L2_weights[0].size();
        for (int w_cnt = 0; w_cnt < weights_to_this_node; w_cnt++)
        {
            if (inputFile >> d_element)
            {
                hid_L2_weights[n_cnt][w_cnt] = d_element;
                data_load_numbers++;
            }
            else
            {
                data_load_error = 1;
            }
            if (data_load_error != 0)
            {
                break;
            }
        }
        if (data_load_error != 0)
        {
            break;
        }
    }
    inputFile.close();
    if (data_load_error == 0)
    {
        cout << "Load data finnish !" << endl;
    }
    else
    {
        cout << "ERROR! weight file error have not sufficient amount of data to put into  hid_L2_weights[n_cnt][w_cnt] vector" << endl;
        cout << "Loaded this amout of data weights data_load_numbers = " << data_load_numbers << endl;
    }


    inputFile.precision(precision_to_text_file);
    inputFile.open(filename_out);
    d_element = 0.0;
    data_load_error = 0;
    data_load_numbers = 0;
    nodes_on_this_layer = out_weights.size();
    for (int n_cnt = 0; n_cnt < nodes_on_this_layer; n_cnt++)
    {
        int weights_to_this_node = out_weights[0].size();
        for (int w_cnt = 0; w_cnt < weights_to_this_node; w_cnt++)
        {
            if (inputFile >> d_element)
            {
                out_weights[n_cnt][w_cnt] = d_element;
                data_load_numbers++;
            }
            else
            {
                data_load_error = 1;
            }
            if (data_load_error != 0)
            {
                break;
            }
        }
        if (data_load_error != 0)
        {
            break;
        }
    }
    inputFile.close();
    if (data_load_error == 0)
    {
        cout << "Load data finnish !" << endl;
    }
    else
    {
        cout << "ERROR! weight file error have not sufficient amount of data to put into  out_weights[n_cnt][w_cnt] vector" << endl;
        cout << "Loaded this amout of data weights data_load_numbers = " << data_load_numbers << endl;
    }
}

void simple_L2_nn::forward_pass(void)
{
    //Forward from input to to hidden node
    for(int dst_cnt=0;dst_cnt<hidden_L1_nodes;dst_cnt++)
    {
        double acc_dot_product = hid_L1_weights[dst_cnt][input_nodes];//Start with Bias weight 
        for(int src_cnt=0;src_cnt<input_nodes;src_cnt++)
        {
            acc_dot_product += hid_L1_weights[dst_cnt][src_cnt] * input_layer[src_cnt];
        }
        //Dot product finnish
        //Make activation function
        hidden_L1_layer[dst_cnt] = 1.0 / (1.0 + exp(-acc_dot_product)); // Sigmoid function and put it into
    }

    //Forward from hidden L1 to to hidden L2 node
    for(int dst_cnt=0;dst_cnt<hidden_L2_nodes;dst_cnt++)
    {
        double acc_dot_product = hid_L2_weights[dst_cnt][hidden_L1_nodes];//Start with Bias weight 
        for(int src_cnt=0;src_cnt<hidden_L1_nodes;src_cnt++)
        {
            acc_dot_product += hid_L2_weights[dst_cnt][src_cnt] * hidden_L1_layer[src_cnt];
        }
        //Dot product finnish
        //Make activation function
        hidden_L2_layer[dst_cnt] = 1.0 / (1.0 + exp(-acc_dot_product)); // Sigmoid function and put it into
    }
    //Forward from hidden to to output node
    for(int dst_cnt=0;dst_cnt<output_nodes;dst_cnt++)
    {
        
        double acc_dot_product = out_weights[dst_cnt][hidden_L2_nodes];//Start with Bias weight 
        for(int src_cnt=0;src_cnt<hidden_L2_nodes;src_cnt++)
        {
            acc_dot_product += out_weights[dst_cnt][src_cnt] * hidden_L2_layer[src_cnt];
        }
        //Dot product finnish
        //Make activation function
        if(use_softmax == 0)
        {
            //Make activation function
            output_layer[dst_cnt] = 1.0 / (1.0 + exp(-acc_dot_product)); // Sigmoid function and put it into
        }
        else
        {
            //Just store dot product for softmax ouside for loop
            output_layer[dst_cnt] = acc_dot_product;//Compleat softmax calculation later need to be outside dst_cnt for loop
        }
    }

    if(use_softmax == 1)
    {
        //Make softmax activation function
        double sum_exp_input = 0.0;
        for(int out_cnt=0;out_cnt<output_nodes;out_cnt++)
        {
            output_layer[out_cnt] = exp(output_layer[out_cnt]);
            sum_exp_input += output_layer[out_cnt];
            if(sum_exp_input == 0.0)
            {
                //Zero div protection 
                sum_exp_input = 0.000000000000001;
            }
        }
        for(int out_cnt=0;out_cnt<output_nodes;out_cnt++)
        {
            output_layer[out_cnt] = output_layer[out_cnt] / sum_exp_input;
        }
    }
}

void simple_L2_nn::backpropagtion_and_update(void)
{
    for(int dst_cnt=0;dst_cnt<output_nodes;dst_cnt++)
    {
        if(use_softmax == 0)
        {
            loss += 0.5 * ((target_layer[dst_cnt] - output_layer[dst_cnt]) * (target_layer[dst_cnt] - output_layer[dst_cnt])); // Squared error * 0.5
        }
        else
        {
            loss -= target_layer[dst_cnt] * log(output_layer[dst_cnt]);
        }
    }

    //Backpropagate over output nodes
    for(int n_cnt = 0;n_cnt<output_nodes;n_cnt++)
    {
        if(use_softmax == 0)
        {
            //Sigmoid function derivatives
            output_delta[n_cnt] = (target_layer[n_cnt] - output_layer[n_cnt]) * output_layer[n_cnt] * (1.0 - output_layer[n_cnt]);
        }
        else
        {
            //Softmax function derivatives
            output_delta[n_cnt] = (target_layer[n_cnt] - output_layer[n_cnt]);
        }
    }

    //Backpropagate over hidden nodes L2
    for(int hid_n_cnt = 0;hid_n_cnt<hidden_L2_nodes;hid_n_cnt++)
    {
        double accumu_delta = 0.0;
        for(int delta_src_cnt=0;delta_src_cnt<output_nodes;delta_src_cnt++)
        {
            accumu_delta += output_delta[delta_src_cnt] * out_weights[delta_src_cnt][hid_n_cnt];
        }
        //Sigmoid function derivatives
        hidden_L2_delta[hid_n_cnt] = accumu_delta * hidden_L2_layer[hid_n_cnt] * (1.0 - hidden_L2_layer[hid_n_cnt]);
    }

    //Backpropagate over hidden nodes L1
    for(int hid_n_cnt = 0;hid_n_cnt<hidden_L1_nodes;hid_n_cnt++)
    {
        double accumu_delta = 0.0;
        for(int delta_src_cnt=0;delta_src_cnt<hidden_L2_nodes;delta_src_cnt++)
        {
            accumu_delta += hidden_L2_delta[delta_src_cnt] * hid_L2_weights[delta_src_cnt][hid_n_cnt];
        }
        //Sigmoid function derivatives
        hidden_L1_delta[hid_n_cnt] = accumu_delta * hidden_L1_layer[hid_n_cnt] * (1.0 - hidden_L1_layer[hid_n_cnt]);
    }



    //Update weights from the backpropated delta
    //Update output weights 
    for(int dst_cnt=0;dst_cnt<output_nodes;dst_cnt++)
    {
        out_change_weights[dst_cnt][hidden_L2_nodes] = learning_rate * output_delta[dst_cnt] + momentum * out_change_weights[dst_cnt][hidden_L2_nodes];//Start with update change Bias weight 
        out_weights[dst_cnt][hidden_L2_nodes] += out_change_weights[dst_cnt][hidden_L2_nodes];//Update bias weight
        for(int src_cnt=0;src_cnt<hidden_L2_nodes;src_cnt++)
        {
            out_change_weights[dst_cnt][src_cnt] = learning_rate * hidden_L2_layer[src_cnt] * output_delta[dst_cnt] + momentum * out_change_weights[dst_cnt][src_cnt];//update change weight 
            out_weights[dst_cnt][src_cnt] += out_change_weights[dst_cnt][src_cnt];//Update weight
        }
    }


    //Update hidden weights L2
    for(int dst_cnt=0;dst_cnt<hidden_L2_nodes;dst_cnt++)
    {
        hid_L2_change_weights[dst_cnt][hidden_L1_nodes] = learning_rate * hidden_L2_delta[dst_cnt] + momentum * hid_L2_change_weights[dst_cnt][hidden_L1_nodes];//Start with update change Bias weight 
        hid_L2_weights[dst_cnt][hidden_L1_nodes] += hid_L2_change_weights[dst_cnt][hidden_L1_nodes];//Update bias weight
        for(int src_cnt=0;src_cnt<hidden_L1_nodes;src_cnt++)
        {
            hid_L2_change_weights[dst_cnt][src_cnt] = learning_rate * hidden_L1_layer[src_cnt] * hidden_L2_delta[dst_cnt] + momentum * hid_L2_change_weights[dst_cnt][src_cnt];//update change weight 
            hid_L2_weights[dst_cnt][src_cnt] += hid_L2_change_weights[dst_cnt][src_cnt];//Update weight
        }
    }


    //Update hidden weights L1
    for(int dst_cnt=0;dst_cnt<hidden_L1_nodes;dst_cnt++)
    {
        hid_L1_change_weights[dst_cnt][input_nodes] = learning_rate * hidden_L1_delta[dst_cnt] + momentum * hid_L1_change_weights[dst_cnt][input_nodes];//Start with update change Bias weight 
        hid_L1_weights[dst_cnt][input_nodes] += hid_L1_change_weights[dst_cnt][input_nodes];//Update bias weight
        for(int src_cnt=0;src_cnt<input_nodes;src_cnt++)
        {
            hid_L1_change_weights[dst_cnt][src_cnt] = learning_rate * input_layer[src_cnt] * hidden_L1_delta[dst_cnt] + momentum * hid_L1_change_weights[dst_cnt][src_cnt];//update change weight 
            hid_L1_weights[dst_cnt][src_cnt] += hid_L1_change_weights[dst_cnt][src_cnt];//Update weight
        }
    }

}