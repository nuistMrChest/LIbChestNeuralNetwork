#ifndef NETWORK_HPP
#define NETWORK_HPP

#include"matrix.hpp"
#include"layer.hpp"
#include<vector>
#include<functional>
#include"activations.hpp"

namespace LibCN{
    template<Element T>struct Network{
        size_t in_size;
        size_t out_size;
        std::vector<Layer<T>>layers;
        T step;

        void train(const Matrix<T>&input,const Matrix<T>&expected){
            Matrix<T>last_output=input;
            Matrix<T>output;
            for(size_t i=0;i<layers.size();i++){
                output=layers[i].forward(last_output);
                last_output=output;
            }
            Matrix<T>last_dl_da=output-expected;
            Matrix<T>dl_da;
            for(size_t i=0;i<layers.size();i++){
                size_t j=layers.size()-1-i;
                dl_da=layers[j].backward(last_dl_da,step);
                last_dl_da=dl_da;
            }
        }

        void setLayer(size_t index,size_t i,size_t o){
            layers[index]=Layer<T>(i,o);
        }

        void setLayerFun(size_t index,const std::function<T(T)>&a,const std::function<T(T)>&a_d){
            layers[index].activation=a;
            layers[index].activation_d=a_d;
        }

        Matrix<T>use(const Matrix<T>&input){
            Matrix<T>res;
            Matrix<T>last_output=input;
            Matrix<T>output;
            for(size_t i=0;i<layers.size();i++){
                output=layers[i].forward(last_output);
                last_output=output;
            }
            res=output;
            return res;
        }

        Network(){
            in_size=0;
            out_size=0;
            layers.resize(0);
            step=T{};
        }

        Network(size_t layer_size,size_t in_size,size_t out_size,const T&step){
            this->in_size=in_size;
            this->out_size=out_size;
            this->step=step;
            this->layers.resize(layer_size);
        }
    
        void init(T low=T(-1),T high=T(1)){
            for(size_t i=0;i<layers.size();i++)layers[i].init(low,high);
        }
    }; 
}

#endif