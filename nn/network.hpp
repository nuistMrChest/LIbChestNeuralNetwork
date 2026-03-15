#ifndef NETWORK_HPP
#define NETWORK_HPP

#include"tensor.hpp"
#include"layer.hpp"
#include<vector>
#include<functional>
#include"activations.hpp"
#include"losses.hpp"
#include<iostream>

namespace LibCN{
    template<Element T>struct MLP{
        size_t in_size;
        size_t out_size;
        std::vector<MLPLayer<T>>layers;
        T step;
        std::function<T(const Tensor<T>&,const Tensor<T>&)>loss;
        std::function<Tensor<T>(const Tensor<T>&,const Tensor<T>&)>loss_d;
        bool ce;

        Tensor<T>saveLayerWeights(size_t index){
            return layers[index].W;
        }

        Tensor<T>saveLayerBias(size_t index){
            return layers[index].b;
        }

        void loadLayerWeights(size_t index,const Tensor<T>&weights){
            layers[index].W=weights;
        }

        void loadLayerBias(size_t index,const Tensor<T>&bias){
            layers[index].b=bias;
        }

        void train(const Tensor<T>&input,const Tensor<T>&expected){
            Tensor<T>last_output=input;
            Tensor<T>output;
            for(size_t i=0;i<layers.size();i++){
                output=layers[i].forward(last_output);
                last_output=output;
            }
            Tensor<T>last_grad;
            if(layers.back().sm&&ce){
                Tensor<T>dl_dz=output-expected;
                last_grad=layers.back().backward_dz(dl_dz,step);
                for(size_t i=0;i<layers.size()-1;i++){
                    size_t j=layers.size()-2-i;
                    last_grad = layers[j].backward(last_grad,step);
                }
            }
            else{
                Tensor<T>last_dl_da=loss_d(output,expected);
                for(size_t i=0;i<layers.size();i++){
                    size_t j=layers.size()-1-i;
                    last_dl_da=layers[j].backward(last_dl_da,step);
                }
            }
        }

        void train_p(const Tensor<T>&input,const Tensor<T>&expected){
            Tensor<T>last_output=input;
            Tensor<T>output;
            for(size_t i=0;i<layers.size();i++){
                output=layers[i].forward(last_output);
                last_output=output;
            }
            std::cout<<"Loss: "<<loss(output,expected)<<std::endl;
            Tensor<T>last_grad;
            if(ce&&layers.back().sm){
                Tensor<T>dl_dz=output-expected;
                last_grad=layers.back().backward_dz(dl_dz,step);
                for(size_t i=0;i<layers.size()-1;i++){
                    size_t j=layers.size()-2-i;
                    last_grad=layers[j].backward(last_grad,step);
                }
            }
            else{
                Tensor<T>last_dl_da=loss_d(output,expected);
                for(size_t i=0;i<layers.size();i++){
                    size_t j=layers.size()-1-i;
                    last_dl_da = layers[j].backward(last_dl_da,step);
                }
            }
        }

        void setLayer(size_t index,size_t i,size_t o){
            layers[index]=MLPLayer<T>(i,o);
        }

        void setLayerFun(size_t index,const std::function<Tensor<T>(const Tensor<T>&)>&a,const std::function<Tensor<T>(const Tensor<T>&)>&a_d){
            layers[index].activation=a;
            layers[index].activation_d=a_d;
        }

        void setLoss(const std::function<T(const Tensor<T>&,const Tensor<T>&)>l,const std::function<Tensor<T>(const Tensor<T>&,const Tensor<T>&)>l_d){
            loss=l;
            loss_d=l_d;
        }

        Tensor<T>use(const Tensor<T>&input){
            Tensor<T>res;
            Tensor<T>last_output=input;
            Tensor<T>output;
            for(size_t i=0;i<layers.size();i++){
                output=layers[i].forward(last_output);
                last_output=output;
            }
            res=output;
            return res;
        }

        MLP(){
            in_size=0;
            out_size=0;
            layers.resize(0);
            step=T{};
            ce=false;
        }

        MLP(size_t layer_size,size_t in_size,size_t out_size,const T&step){
            this->in_size=in_size;
            this->out_size=out_size;
            this->step=step;
            this->layers.resize(layer_size);
            ce=false;
        }
    
        void init(T low=T(-1),T high=T(1)){
            for(size_t i=0;i<layers.size();i++)layers[i].init(low,high);
        }
    }; 
}

#endif