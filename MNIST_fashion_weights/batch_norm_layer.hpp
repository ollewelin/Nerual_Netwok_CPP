#ifndef BATCH_NORM_LAYER_H
#define BATCH_NORM_LAYER_H

#include<vector>
#include<string>
using namespace std;
class batch_norm_layer
{
private:

    int batch_size;
    int channels;
    int rows;
    int cols;
    int version_major;
    int version_mid;
    int version_minor;
    int setup_state;
    double init_gamma;
    double init_beta;
    //Setup functions below must be called in this order to make setup working.
    //0 = start up state, nothing done yet. 
    //1 = set_up_tensor() done
    //2 = randomize_weights() or load_weights() is done

    vector<vector<vector<double>>> gamma;//3D [channels][row][col]. Learnabe batch norm parameters
    vector<vector<vector<double>>> beta;//3D [channels][row][col]. Learnabe batch norm parameters
    vector<vector<vector<double>>> delta_sum_gamma;//3D [channels][row][col]. delta for gamma
    vector<vector<vector<double>>> delta_sum_beta;//3D [channels][row][col]. delta for beta
    vector<vector<vector<double>>> mean;//3D [channels][row][col]. Calculated when cycle_mean_variance_through_batch() fucntion called
    vector<vector<vector<double>>> variance;//3D [channels][row][col]. Calculated when cycle_mean_variance_through_batch() fucntion called
    vector<vector<vector<vector<double>>>> x_norm;//4D [batch_size][channels][output_row][output_col].

public:
    void set_special_gamma_beta_init(double, double);//Do before set_up_tensors() call, If not use this function default gamma = 1.0 and beta = 0.0
    void set_up_tensors(int,int,int,int);//4D [batch_size][channels][row][col].
    void load_weights(string);//load weights with file name argument 
    void save_weights(string);//save weights with file name argument 
    void forward_batch(void);
    void backprop_batch(void);
    vector<vector<vector<vector<double>>>> input_tensor;//4D [batch_size][channels][row][col].     The size of this vectors will setup inside set_in_tensor(int, int) function when called.
    vector<vector<vector<vector<double>>>> i_tensor_delta;//4D [batch_size][channels][row][col].   The size of this vectors will setup inside set_in_tensor(int, int) function when called.
    vector<vector<vector<vector<double>>>> output_tensor;//4D [batch_size][channels][output_row][output_col].     The size of this vectors will setup inside set_out_tensor(int) function when called.
    vector<vector<vector<vector<double>>>> o_tensor_delta;//4D [batch_size][channels][output_row][output_col].    The size of this vectors will setup inside set_out_tensor(int) function when called.S
    double lr;//learning rate
    batch_norm_layer();
    ~batch_norm_layer();

    void get_version(void);
    int ver_major;   
    int ver_mid;
    int ver_minor;
};

#endif //BATCH_NORM_LAYER_H