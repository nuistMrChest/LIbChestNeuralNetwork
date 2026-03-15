#ifndef LAYER_HPP
#define LAYER_HPP

#include"tensor.hpp"
#include<functional>
#include<random>

namespace LibCN{
    template<Element T>struct MLPLayer{
        std::function<Tensor<T>(const Tensor<T>&)>activation;
        std::function<Tensor<T>(const Tensor<T>&)>activation_d;
        size_t in_size;
        size_t out_size;
        Tensor<T>W;
        Tensor<T>b;
        Tensor<T>last_input;
        Tensor<T>z;
        bool sm;

        MLPLayer(){
            in_size=0;
            out_size=0;
            W=Tensor<T>();
            b=Tensor<T>();
            last_input=Tensor<T>();
            z=Tensor<T>();
            sm=false;
        }

        MLPLayer(size_t i,size_t o){
            in_size=i;
            out_size=o;
            W.resize(2,{o,i});
            b.resize(2,{o,1});
            last_input.resize(2,{i,1});
            z.resize(2,{o,1});
            sm=false;
        }

        Tensor<T>forward(const Tensor<T>&input){
            Tensor<T>res(2,{out_size,1});
            last_input=input;
            if(input.getShape()[0]==in_size&&input.getShape()[1]==1)z=((W.matrixMultiplication(input))+b);
            res=activation(z);
            return res;
        }

        Tensor<T>backward(const Tensor<T>&dl_da,const T&step){
            Tensor<T>res;
            Tensor<T>dl_dz=dl_da.hadamard(activation_d(z));
            res=W.transpose(0,1).matrixMultiplication(dl_dz);
            W-=step*(dl_dz.matrixMultiplication(last_input.transpose(0,1)));
            b-=step*dl_dz;
            return res;
        }

        Tensor<T>backward_dz(const Tensor<T>&dl_dz,const T&step){
            Tensor<T>res=W.transpose(0,1).matrixMultiplication(dl_dz);
            W-=step*(dl_dz.matrixMultiplication(last_input.transpose(0,1)));
            b-=step*dl_dz;
            return res;
        }
    
        void init(T low=T(-1),T high=T(1)){
            static std::mt19937 rng(std::random_device{}());
            std::uniform_real_distribution<T>dist(low,high);
            for(size_t i=0;i<out_size;++i){
                for(size_t j=0;j<in_size;++j){
                    W(i,j)=dist(rng);
                }
                b(i,0)=dist(rng);
            }
        }
    };
}

#endif