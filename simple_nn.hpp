#ifndef SIMPLE_NN_H
#define SIMPLE_NN_H
#include<vector>
#include<string>

using namespace std;
class simple_nn
{
private:
    int input_nodes;
    int hidden_nodes;
    int output_nodes;
    vector<double> hidden_layer;
    vector<double> hidden_delta;
    vector<double> output_delta;
    vector<vector<double>> hid_weights;//hid_weights[to_node][from_node]
    vector<vector<double>> hid_change_weights;//[to_node][from_node]
    vector<vector<double>> out_weights;//[to_node][from_node]
    vector<vector<double>> out_change_weights;//[to_node][from_node]

public:
    double loss;
    double learning_rate;
    double momentum;
    vector<double> input_layer;
    vector<double> output_layer;

    
    simple_nn();
    ~simple_nn();
};




#endif//SIMPLE_NN_H
