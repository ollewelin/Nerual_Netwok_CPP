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
    //Setup functions below must be called in this order to make setup working.
    //0 = start up state, nothing done yet. 
    //1 = set_kernel_size() is set up
    //2 = set_stride() stride is set up
    //3 = set_in_tensor() done
    //4 = set_out_tensor_channels() is done
    //5 = randomize_weights() or load_weights() is done
public:
    vector<vector<vector<vector<double>>>> kernel_weights;//4D [output_channel][input_channel][kernel_row][kernel_col]
private:
    vector<vector<vector<vector<double>>>> change_weights;//4D [output_channel][input_channel][kernel_row][kernel_col]
    vector<vector<vector<vector<double>>>> kernel_deltas;//4D [output_channel][input_channel][kernel_row][kernel_col]
    vector<double> accum_bias_deltas;//1D [output_channel]
    vector<double> kernel_bias_weights;//1D [output_channel]
    vector<double> change_bias_weights;//1D [output_channel]
    vector<vector<vector<double>>> internal_tensor_delta;//3D [output_channel][output_row][output_col]
    double activation_function(double);
    double delta_activation_func(double,double);
    void xy_start_stop_kernel(int);
    int start_ret;
    int stop_ret;
public:
    void set_kernel_size(int);
    int get_kernel_size(void);
    void set_stride(int);
    int get_stride(void);
    void set_in_tensor(int, int);//input data size total (both side of sqare) on one channel, input channels
    vector<vector<vector<double>>> input_tensor;//3D [input_channel][row][col].     The size of this vectors will setup inside set_in_tensor(int, int) function when called.
    vector<vector<vector<double>>> i_tensor_delta;//3D [input_channel][row][col].   The size of this vectors will setup inside set_in_tensor(int, int) function when called.
    void set_out_tensor(int);//output channels
    vector<vector<vector<double>>> output_tensor;//3D [output_channel][output_row][output_col].     The size of this vectors will setup inside set_out_tensor(int) function when called.
    vector<vector<vector<double>>> o_tensor_delta;//3D [output_channel][output_row][output_col].    The size of this vectors will setup inside set_out_tensor(int) function when called.S
    void randomize_weights(double);//the input argument must be in range 0..1 but to somthing low for example 0.01
    void load_weights(string, string);//load weights with file name argument 
    void save_weights(string, string);//save weights with file name argument 
    void conv_forward1(void);//Low memory swaping high aritmetric operation variant
    void clear_i_tens_delta(void);
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
    int use_dropouts;
    int top_conv;//If set to 1 disable calcualtioen of i_delta to speed up first convolution calculate only kernel delta if this is 1
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