#ifndef CONVOLUTION_H
#define CONVOLUTION_H
#include<vector>
#include<string>
using namespace std;
class convolution
{
private:
    int version_major;
    int version_mid;
    int version_minor;

    int kernel_size;
    int stride;
    int output_tensor_channels;


    int setup_state;
    //0 = start up state, nothing done yet
    //1 = set_kernel_size() is set up
    //2 = set_out_tensor_channels() is done
    //4 = init_weights or load weights is done

    vector<vector<vector<vector<double>>>> kernel_weights;//[output_channel][input_channel][kernel_row][kernel_col]
    vector<vector<vector<vector<double>>>> change_weights;//[output_channel][input_channel][kernel_row][kernel_col]
    vector<vector<vector<vector<double>>>> kernel_deltas;//[output_channel][input_channel][kernel_row][kernel_col]

public:
    vector<vector<vector<double>>> input_tensor;//[input_channel][row][col]
    vector<vector<vector<double>>> i_tensor_delta;
    void set_kernel_size(int);
    int get_kernel_size(void);
    void set_stride(int);
    int get_stride(void);
    void set_out_tensor_channels(int);
    void randomize_weights(double);//the input argument must be in range 0..1 but to somthing low for example 0.01
    void load_weights(string);//load weights with file name argument 
    void save_weights(string);//save weights with file name argument 
    vector<vector<vector<double>>> output_tensor;
    vector<vector<vector<double>>> o_tensor_delta;
    void conv_forward(void);
    void conv_backprop(void);
    void conv_update_weights(void);
    void conv_transpose_fwd(void);//Same algorithm as conv_backprop but go forward from output_tensor to input_tensor. Used for show patches or as forward conv autoencodes
    void conv_transpose_bkp(void);//Same algorithm as conv_forward but backprop from input_tensor to output_tensor. Used for training conv autoencodes

    int activation_function_mode;
    //0 = sigmoid activation function
    //1 = Relu simple activation function
    //2 = Relu fix leaky activation function
    double fix_leaky_proportion;
    double learning_rate;
    double momentum;

    void get_version(void);
    int ver_major;
    int ver_mid;
    int ver_minor;


    convolution();
    ~convolution();
};



#endif