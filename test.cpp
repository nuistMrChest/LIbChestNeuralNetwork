#include"./nn/matrix.hpp"
#include"./nn/layer.hpp"
#include<iostream>

using namespace std;
using namespace LibCN;

int main(){
    Layer<float>test(2,3);
    test.activation=[](float x)->float{if(x>0)return x;else return 0;};
    test.W=Matrix<float>({
        {0.2,-0.4},
        {0.7,0.3},
        {-0.5,0.8}
    });
    test.b=Matrix<float>({
        {0.1},
        {-0.2},
        {0.05}
    });
    Matrix<float>input{
        {0.5},
        {1.0}
    };
    Matrix<float>res=test.forward(input);
    cout<<res<<endl;
}