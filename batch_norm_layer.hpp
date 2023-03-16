#ifndef BATCH_NORM_LAYER_H
#define BATCH_NORM_LAYER_H

#include<vector>
#include<string>
using namespace std;
class batch_norm_layer
{
private:
    int samp_cnt;
    
public:
    int set_in_tensor(vector<vector<vector<double>>>);//3D [channels][row][col].
    void forward_one_sample(void);
    vector<vector<vector<double>>> input_tensor;//3D [channels][row][col].     The size of this vectors will setup inside set_in_tensor(int, int) function when called.
    vector<vector<vector<double>>> i_tensor_delta;//3D [channels][row][col].   The size of this vectors will setup inside set_in_tensor(int, int) function when called.
    vector<vector<vector<double>>> output_tensor;//3D [channels][output_row][output_col].     The size of this vectors will setup inside set_out_tensor(int) function when called.
    vector<vector<vector<double>>> o_tensor_delta;//3D [channels][output_row][output_col].    The size of this vectors will setup inside set_out_tensor(int) function when called.S
    double lr;//learning rate
    int batch_size;
    batch_norm_layer();
    ~batch_norm_layer();
};

#endif //BATCH_NORM_LAYER_H