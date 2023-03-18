#ifndef BATCH_NORM_LAYER_H
#define BATCH_NORM_LAYER_H

#include<vector>
#include<string>
using namespace std;
class batch_norm_layer
{
private:
    int samp_cnt;
    int batch_size;
    int channels;
    int rows;
    int cols;
    int version_major;
    int version_mid;
    int version_minor;
    vector<vector<vector<double>>> gamma;//3D [channels][row][col]. Learnabe batch norm parameters
    vector<vector<vector<double>>> beta;//3D [channels][row][col]. Learnabe batch norm parameters
    vector<vector<vector<double>>> mean;//3D [channels][row][col]. Calculated when cycle_mean_variance_through_batch() fucntion called
    vector<vector<vector<double>>> variance;//3D [channels][row][col]. Calculated when cycle_mean_variance_through_batch() fucntion called

public:
    void set_up_tensors(int,int,int,int);//4D [batch_size][channels][row][col].
    int forward_batch(void);//return sample coundter
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