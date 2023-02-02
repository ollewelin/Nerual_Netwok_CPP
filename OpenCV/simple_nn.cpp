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
    input_nodes = 3;
    hidden_nodes = 100;
    output_nodes = 3;
    vector<double> dummy_from_inp_node;
    dummy_from_inp_node.push_back(0.0);//Add one for Bias node
    for(int i=0;i<input_nodes;i++)
    {
        input_layer.push_back(0.0);
        dummy_from_inp_node.push_back(0.0);
    }
    vector<double> dummy_from_hid_node;
    dummy_from_hid_node.push_back(0.0);//Add one for Bias node
    for(int i=0;i<hidden_nodes;i++)
    {
        hidden_layer.push_back(0.0);
        hidden_delta.push_back(0.0);
        dummy_from_hid_node.push_back(0.0);
        hid_weights.push_back(dummy_from_inp_node);
        hid_change_weights.push_back(dummy_from_inp_node);
    }
    for(int i=0;i<output_nodes;i++)
    {
        output_layer.push_back(0.0);
        output_delta.push_back(0.0);
        out_weights.push_back(dummy_from_hid_node);
        out_change_weights.push_back(dummy_from_hid_node);
    }
    
}

simple_nn::~simple_nn()
{
    cout << "Destructor simple_nn" << endl;
}