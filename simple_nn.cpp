#include "simple_nn.hpp"
#include <iostream>
#include <fstream>
#include <stdlib.h> // for exit(0)
#include <math.h>
#include <time.h>

using namespace std;

simple_nn::simple_nn()
{
    cout << "Constructor simple_nn" << endl;
    input_nodes = 784;
    hidden_nodes = 100;
    output_nodes = 10;
    use_softmax = 0;
    vector<double> dummy_from_inp_node;
    dummy_from_inp_node.push_back(0.0); // Add one for Bias node
    for (int i = 0; i < input_nodes; i++)
    {
        input_layer.push_back(0.0);
        dummy_from_inp_node.push_back(0.0);
    }
    vector<double> dummy_from_hid_node;
    dummy_from_hid_node.push_back(0.0); // Add one for Bias node
    for (int i = 0; i < hidden_nodes; i++)
    {
        hidden_layer.push_back(0.0);
        hidden_delta.push_back(0.0);
        dummy_from_hid_node.push_back(0.0);
        hid_weights.push_back(dummy_from_inp_node);
        hid_change_weights.push_back(dummy_from_inp_node);
    }
    for (int i = 0; i < output_nodes; i++)
    {
        output_layer.push_back(0.0);
        target_layer.push_back(0.0);
        output_delta.push_back(0.0);
        out_weights.push_back(dummy_from_hid_node);
        out_change_weights.push_back(dummy_from_hid_node);
    }
}

simple_nn::~simple_nn()
{
    cout << "Destructor simple_nn" << endl;
}

void simple_nn::randomize_weights(double rand_proportion)
{
    cout << "Randomize weights 3D vector all weights of simple_nn object...." << endl;
    int nodes_on_hid_layer = hid_weights.size();
    for (int n_cnt = 0; n_cnt < nodes_on_hid_layer; n_cnt++)
    {
        int weights_to_this_node = hid_weights[0].size();
        for (int w_cnt = 0; w_cnt < weights_to_this_node; w_cnt++)
        {
            hid_weights[n_cnt][w_cnt] = ((((double)rand() / RAND_MAX) - 0.5) * rand_proportion);
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
void simple_nn::save_weights(string filename)
{
    cout << "Save data weights ..." << endl;
    string filename_extenstion_hid = "_hid.dat";
    string filename_extenstion_out = "_out.dat";
    ofstream outputFile;
    outputFile.precision(precision_to_text_file);
    outputFile.open(filename + filename_extenstion_hid);
    int nodes_on_this_layer = hid_weights.size();
    for (int n_cnt = 0; n_cnt < nodes_on_this_layer; n_cnt++)
    {
        int weights_to_this_node = hid_weights[0].size();
        for (int w_cnt = 0; w_cnt < weights_to_this_node; w_cnt++)
        {
            outputFile << hid_weights[n_cnt][w_cnt] << endl;
        }
    }
    outputFile.close();

    outputFile.precision(precision_to_text_file);
    outputFile.open(filename + filename_extenstion_out);
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
void simple_nn::load_weights(string filename)
{
    cout << "Load data weights ..." << endl;
    ifstream inputFile;
    string filename_extenstion_hid = "_hid.dat";
    string filename_extenstion_out = "_out.dat";

    inputFile.precision(precision_to_text_file);
    inputFile.open(filename + filename_extenstion_hid);
    double d_element = 0.0;
    int data_load_error = 0;
    int data_load_numbers = 0;
    int nodes_on_this_layer = hid_weights.size();
    for (int n_cnt = 0; n_cnt < nodes_on_this_layer; n_cnt++)
    {
        int weights_to_this_node = hid_weights[0].size();
        for (int w_cnt = 0; w_cnt < weights_to_this_node; w_cnt++)
        {
            if (inputFile >> d_element)
            {
                hid_weights[n_cnt][w_cnt] = d_element;
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
        cout << "ERROR! weight file error have not sufficient amount of data to put into  hid_weights[n_cnt][w_cnt] vector" << endl;
        cout << "Loaded this amout of data weights data_load_numbers = " << data_load_numbers << endl;
    }

    inputFile.precision(precision_to_text_file);
    inputFile.open(filename + filename_extenstion_out);
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

void simple_nn::forward_pass(void)
{
    //Forward from input to to hidden node
    int dst_nodes = hidden_layer.size();
    for(int dst_cnt=0;dst_cnt<dst_nodes;dst_cnt++)
    {
        int src_nodes = input_layer.size();
        double acc_dot_product = hid_weights[dst_cnt][src_nodes];//Start with Bias weight 
        for(int src_cnt=0;src_cnt<src_nodes;src_cnt++)
        {
            acc_dot_product += hid_weights[dst_cnt][src_cnt] * input_layer[src_cnt];
        }
        //Dot product finnish
        //Make activation function
        hidden_layer[dst_cnt] = 1.0 / (1.0 + exp(-acc_dot_product)); // Sigmoid function and put it into
    }

    //Forward from hidden to to output node
    dst_nodes = output_layer.size();
    for(int dst_cnt=0;dst_cnt<dst_nodes;dst_cnt++)
    {
        int src_nodes = hidden_layer.size();
        double acc_dot_product = out_weights[dst_cnt][src_nodes];//Start with Bias weight 
        for(int src_cnt=0;src_cnt<src_nodes;src_cnt++)
        {
            acc_dot_product += out_weights[dst_cnt][src_cnt] * hidden_layer[src_cnt];
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
            if(sum_exp_input != 0.0)
            {
                output_layer[out_cnt] = output_layer[out_cnt] / sum_exp_input;
            }
            else
            {
                //Zero div protection 
                output_layer[out_cnt] = output_layer[out_cnt] / 0.000000000000001;
            }
        }
    }
}

void simple_nn::backpropagtion_and_update(void)
{
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

    //Backpropagate over hidden nodes
    for(int hid_n_cnt = 0;hid_n_cnt<hidden_nodes;hid_n_cnt++)
    {
        double accumu_delta = 0.0;
        for(int delta_src_cnt=0;delta_src_cnt<output_nodes;delta_src_cnt++)
        {
            accumu_delta += output_delta[delta_src_cnt] * out_weights[delta_src_cnt][hid_n_cnt];
        }
        //Sigmoid function derivatives
        hidden_delta[hid_n_cnt] = accumu_delta * hidden_layer[hid_n_cnt] * (1.0 - hidden_layer[hid_n_cnt]);
    }

    //Update weights from the backpropated delta
    //Update output weights 
    for(int dst_cnt=0;dst_cnt<output_nodes;dst_cnt++)
    {
        out_change_weights[dst_cnt][output_nodes] = learning_rate * output_delta[dst_cnt] + momentum * out_change_weights[dst_cnt][output_nodes];//Start with update change Bias weight 
        out_weights[dst_cnt][output_nodes] += out_change_weights[dst_cnt][output_nodes];//Update bias weight
        for(int src_cnt=0;src_cnt<hidden_nodes;src_cnt++)
        {
            out_change_weights[dst_cnt][src_cnt] = learning_rate * output_delta[dst_cnt] + momentum * out_change_weights[dst_cnt][src_cnt];//update change weight 
            out_weights[dst_cnt][src_cnt] += out_change_weights[dst_cnt][src_cnt];//Update weight
        }
    }

    //Update hidden weights 
    for(int dst_cnt=0;dst_cnt<hidden_nodes;dst_cnt++)
    {
        hid_change_weights[dst_cnt][hidden_nodes] = learning_rate * output_delta[dst_cnt] + momentum * hid_change_weights[dst_cnt][hidden_nodes];//Start with update change Bias weight 
        hid_weights[dst_cnt][hidden_nodes] += hid_change_weights[dst_cnt][hidden_nodes];//Update bias weight
        for(int src_cnt=0;src_cnt<hidden_nodes;src_cnt++)
        {
            hid_change_weights[dst_cnt][src_cnt] = learning_rate * output_delta[dst_cnt] + momentum * hid_change_weights[dst_cnt][src_cnt];//update change weight 
            hid_weights[dst_cnt][src_cnt] += hid_change_weights[dst_cnt][src_cnt];//Update weight
        }
    }

}