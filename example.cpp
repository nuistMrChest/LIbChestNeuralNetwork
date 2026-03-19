#include"lib_chest_nn.hpp"
#include<iostream>

using namespace std;
using namespace LibCN;

int main(){
    cout<<"this is an example for demostrating the train and use of xor using LibCN"<<endl;
    MLP<float>net(2,2,1,0.05f);

    net.setLoss(Losses::MSE<float>,Losses::MSE_d<float>);

    net.setLayer(0,2,4);
    net.setLayer(1,4,1);

    net.init(-0.5f,0.5f);

    net.setLayerFun(0,Activations::tanh<float>,Activations::tanh_d<float>);
    net.setLayerFun(1,Activations::sigmoid<float>,Activations::sigmoid_d<float>);

    cout<<"MLP network initialized"<<endl;

    Tensor<float>x1=Tensor<float>::matrix({
        {0},
        {0}
    });
    Tensor<float>x2=Tensor<float>::matrix({
        {0},
        {1}
    });
    Tensor<float>x3=Tensor<float>::matrix({
        {1},
        {0}
    });
    Tensor<float>x4=Tensor<float>::matrix({
        {1},
        {1}
    });

    Tensor<float>y1=Tensor<float>::matrix({{0}});
    Tensor<float>y2=Tensor<float>::matrix({{1}});
    Tensor<float>y3=Tensor<float>::matrix({{1}});
    Tensor<float>y4=Tensor<float>::matrix({{0}});   

    cout<<"training data prepared"<<endl;

    cout<<"before training"<<endl;
    cout<<"0 xor 0 -> "<<net.use(x1)<<endl;
    cout<<"0 xor 1 -> "<<net.use(x2)<<endl;
    cout<<"1 xor 0 -> "<<net.use(x3)<<endl;
    cout<<"1 xor 1 -> "<<net.use(x4)<<endl;

    for(int i=0;i<50000;++i){
        if(i%5000==0){
            net.train_p(x1,y1);
            net.train_p(x2,y2);
            net.train_p(x3,y3);
            net.train_p(x4,y4);
        }
        else{
            net.train(x1, y1);
            net.train(x2, y2);
            net.train(x3, y3);
            net.train(x4, y4);
        }
    }

    cout<<"\nafter training"<<endl;
    cout<<"0 xor 0 -> "<<net.use(x1)<<endl;
    cout<<"0 xor 1 -> "<<net.use(x2)<<endl;
    cout<<"1 xor 0 -> "<<net.use(x3)<<endl;
    cout<<"1 xor 1 -> "<<net.use(x4)<<endl;

    auto w0=net.saveLayerWeights(0);
    auto b0=net.saveLayerBias(0);
    auto w1=net.saveLayerWeights(1);
    auto b1=net.saveLayerBias(1);

    cout<<"theta saved"<<endl;

    MLP<float>test_net(2,2,1,0);

    test_net.setLayer(0,2,4);
    test_net.setLayer(1,4,1);

    test_net.setLayerFun(0,Activations::tanh<float>,Activations::tanh_d<float>);
    test_net.setLayerFun(1,Activations::sigmoid<float>,Activations::sigmoid_d<float>);

    cout<<"new MLP network created"<<endl;

    test_net.loadLayerWeights(0,w0);
    test_net.loadLayerBias(0,b0);
    test_net.loadLayerWeights(1,w1);
    test_net.loadLayerBias(1,b1);

    cout<<"theta loaded"<<endl;

    while(true){
        cout<<"please input two booleans (1 or 0), or input other value to quit"<<endl;
        Tensor<float>x(2,{2,1});
        if(!(cin>>x(0,0)>>x(1,0)))break;
        auto yt=test_net.use(x);
        cout<<(yt(0,0)>0.5?"true" : "false")<<endl;
    }

    return 0;
}