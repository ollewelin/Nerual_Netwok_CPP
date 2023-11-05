#ifndef FC_M_RESNET_H
#define FC_M_RESNET_H
#include<vector>
#include<string>
using namespace std;
class fc_m_resnet
{
private:
    int version_major;
    int version_mid;
    int version_minor;

    int nr_of_hidden_layers;//Numbers of hidden layers inside this m_resnet block object
    int setup_inc_layer_cnt;//Used during setup 
    int setup_state;
    //0 = start up state, nothing done yet
    //1 = set_nr_of_hidden_layers() is set up
    //2 = set_nr_of_hid_nodeson_layer() is done
    //3 = init_weights or load weights is done
    int batch_cnt;//Only used in batch mode not SGD mode
    vector<int> number_of_hidden_nodes;
    vector<vector<double>> hidden_layer;//2D [layer_nr][node_nr]
    vector<vector<double>> internal_delta;//2D [layer_nr][node_nr] nr_of_hidden_layers + 2 for i_layer_delta and o_layer_delta as well
    vector<vector<vector<double>>> all_weights;//3D [layer_nr][node_nr][weights_from_previous_layer]
    vector<vector<vector<double>>> change_weights;//3D [layer_nr][node_nr][weights_from_previous_layer]
    double activation_function(double, int);
    double delta_activation_func(double,double);

    int skip_conn_rest_part;
    int skip_conn_multiple_part;
    int skip_conn_in_out_relation;
    //0 = same input/output
    //1 = input > output
    //2 = output > input
    int shift_ununiform_skip_connection_sample_counter;//Sample coutner related to shift_ununiform_skip_connection_after_samp_n
    int switch_skip_con_selector;
public:
    fc_m_resnet(/* args */);
    ~fc_m_resnet();
    int block_type;//0..2
    //0 = start block 
    //1 = middle block means both i_layer_delta is produced (backpropagate) and o_layer_delta is needed
    //2 = end block. target_nodes is used but o_layer_delta not used only loss and i_layer_delta i calculated. 
    int use_softmax;
    //0 = No softmax
    //1 = Use softmax at end. Only possible to enable if block_type = 2 end block
    int activation_function_mode;
    //0 = sigmoid activation function
    //1 = Relu simple activation function
    //2 = Relu fix leaky activation function
    double fix_leaky_proportion;
    int use_skip_connect_mode;
    //0 = turn OFF skip connections, ordinary fully connected nn block only
    //1 = turn ON skip connectons
    int shift_ununiform_skip_connection_after_samp_n;
    //0 = all ununiform in/out skip connections are connected every sample every where (forward and backwards). default
    //1 = If ununiform in/out skip connectetions are the case, only uniform in/out skip connection is made forward and backwards and switched every sample
    //2....n = uniform in/out skip connection switch after this number of samples

    int training_mode;
    //0 = SGD Stocastic Gradient Decent
    //1 = Batch Gradient Decent, not yet implemented
    int batch_size; //Only used if trainging_mode 1
    int use_dropouts;
    //0 = No dropout
    //1 = Use dropout
    double loss_A;//
    double loss_B;//Same calculation as A just to diffrent how user can zero out as needed outide this class
    double learning_rate;
    double momentum;
    double dropout_proportion;


    vector<double> input_layer;//Always used, block type 0,1,2
    vector<double> output_layer;//Always used, block type 0,1,2
    vector<double> target_layer;//Only used when block type, 2 = end block. target_nodes is used 
    vector<double> i_layer_delta;//Delta for all input nodes this should be backprop connect to previous multiple resenet block if there is any. Always used, block type 0,1,2
    vector<double> o_layer_delta;//Detla from m_resnet block after this block. Always used, block type 0,1,2

    //========== Functions ==================
   
    void set_nr_of_hidden_layers(int);//First you need to set this number of hidden layers
    void set_nr_of_hidden_nodes_on_layer_nr(int);//set nr of hidden nodes on eact layer. setup_inc_layer_cnt will increment each call of this function
    void randomize_weights(double);//the input argument must be in range 0..1 but to somthing low for example 0.01
    void load_weights(string);//load weights with file name argument 
    void save_weights(string);//save weights with file name argument 
    void forward_pass(void);
    void only_loss_calculation(void);
    void backpropagtion_and_update(void);//If batchmode update only when batch end
    void print_weights(void);

    //================= Functions only for debugging/verify the backpropagation gradient functions ============
    double verify_gradient(int,int,int,double);
    double calc_error_verify_grad(void);
    //=========================================================================================================
    
    void get_version(void);
    int ver_major;
    int ver_mid;
    int ver_minor;
};


#endif//FC_M_RESNET_H
