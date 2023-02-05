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
    int use_softmax;
    vector<double> input_layer;
    vector<double> output_layer;
    vector<double> target_layer;


    void randomize_weights(double);//the input argument must be in range 0..1 but to somthing low for example 0.01
    void load_weights(string);//load weights with file name argument 
    void save_weights(string);//save weights with file name argument 
    void forward_pass(void);
   // void only_loss_calculation(void);
    void backpropagtion_and_update(void);//If batchmode update only when batch end

    
    simple_nn();
    ~simple_nn();
};




#endif//SIMPLE_NN_H
