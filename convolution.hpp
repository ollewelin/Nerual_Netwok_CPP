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
    int input_tensor_channels;
    int root_of_intdata_size;
    int output_tensor_channels;
    int input_side_size;
    int output_side_size;

   
    int setup_state;
    //0 = start up state, nothing done yet
    //1 = set_kernel_size() is set up
    //2 = set_out_tensor_channels() is done
    //4 = init_weights or load weights is done

    vector<vector<vector<vector<double>>>> kernel_weights;//4D [output_channel][input_channel][kernel_row][kernel_col]
    vector<vector<vector<vector<double>>>> change_weights;//4D [output_channel][input_channel][kernel_row][kernel_col]
    vector<vector<vector<vector<double>>>> kernel_deltas;//4D [output_channel][input_channel][kernel_row][kernel_col]
    vector<double> accum_bias_deltas;//1D [output_channel]
    vector<double> kernel_bias_weights;//1D [output_channel]
    vector<double> change_bias_weights;//1D [output_channel]
    double activation_function(double);
    double delta_activation_func(double,double);
public:
    vector<vector<vector<double>>> input_tensor;//3D [input_channel][row][col]
    vector<vector<vector<double>>> i_tensor_delta;//3D [input_channel][row][col] 
    void set_kernel_size(int);
    int get_kernel_size(void);
    void set_stride(int);
    int get_stride(void);
    void set_in_tensor(int, int);//input data size total (both side of sqare) on one channel, input channels
    void set_out_tensor(int);//output channels
    void randomize_weights(double);//the input argument must be in range 0..1 but to somthing low for example 0.01
    void load_weights(string);//load weights with file name argument 
    void save_weights(string);//save weights with file name argument 
    vector<vector<vector<double>>> output_tensor;//3D [output_channel][output_row][output_col] 
    vector<vector<vector<double>>> o_tensor_delta;//3D [output_channel][output_row][output_col] 
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
    int use_dopouts;
    //0 = No dropout
    //1 = Use dropout
    double dropout_proportion;


    void get_version(void);
    int ver_major;
    int ver_mid;
    int ver_minor;


    convolution();
    ~convolution();
};



#endif