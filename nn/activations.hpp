#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include"tensor.hpp"
#include<cmath>

namespace LibCN{
    namespace Activations{
        template<Element T>Tensor<T>relu(const Tensor<T>&a){
            Tensor<T>res(a.getDimension(),a.getShape());
            for(size_t i=0;i<a.getShape()[0];i++)for(size_t j=0;j<a.getShape()[1];j++)res(i,j)=a(i,j)>T(0)?a(i,j):T(0);
            return res;
        }

        template<Element T>Tensor<T>relu_d(const Tensor<T>&a){
            Tensor<T>res(a.getDimension(),a.getShape());
            for(size_t i=0;i<a.getShape()[0];i++)for(size_t j=0;j<a.getShape()[1];j++)res(i,j)=a(i,j)>T(0)?T(1):T(0);
            return res;
        }

        template<Element T>Tensor<T>leaky_relu(const Tensor<T>&a){
            Tensor<T>res(a.getDimension(),a.getShape());
            for(size_t i=0;i<a.getShape()[0];i++)for(size_t j=0;j<a.getShape()[1];j++)res(i,j)=a(i,j)>T(0)?a(i,j):T(0.01)*a(i,j);
            return res;
        }

        template<Element T>Tensor<T>leaky_relu_d(const Tensor<T>&a){
            Tensor<T>res(a.getDimension(),a.getShape());
            for(size_t i=0;i<a.getShape()[0];i++)for(size_t j=0;j<a.getShape()[1];j++)res(i,j)=a(i,j)>T(0)?T(1):T(0.01);
            return res;
        }

        template<Element T>Tensor<T>sigmoid(const Tensor<T>&a){
            Tensor<T>res(a.getDimension(),a.getShape());
            for(size_t i=0;i<a.getShape()[0];i++)for(size_t j=0;j<a.getShape()[1];j++)res(i,j)=T(1)/(T(1)+std::exp(T(-1)*a(i,j)));
            return res;
        }

        template<Element T>Tensor<T>sigmoid_d(const Tensor<T>&a){
            Tensor<T>res(a.getDimension(),a.getShape());
            Tensor<T>s=sigmoid(a);
            for(size_t i=0;i<a.getShape()[0];i++)for(size_t j=0;j<a.getShape()[1];j++)res(i,j)=s(i,j)*(T(1)-s(i,j));
            return res;
        }

        template<Element T>Tensor<T>tanh(const Tensor<T>&a){
            Tensor<T>res(a.getDimension(),a.getShape());
            for(size_t i=0;i<a.getShape()[0];i++)for(size_t j=0;j<a.getShape()[1];j++)res(i,j)=std::tanh(a(i,j));
            return res;
        }

        template<Element T>Tensor<T>tanh_d(const Tensor<T>&a){
            Tensor<T>res(a.getDimension(),a.getShape());
            Tensor<T>t=tanh(a);
            for(size_t i=0;i<a.getShape()[0];i++)for(size_t j=0;j<a.getShape()[1];j++)res(i,j)=T(1)-t(i,j)*t(i,j);
            return res;
        }

        template<Element T>Tensor<T>identity(const Tensor<T>&a){
            return a;
        }

        template<Element T>Tensor<T>identity_d(const Tensor<T>&a){
            Tensor<T>res(a.getDimension(),a.getShape());
            for(size_t i=0;i<a.getShape()[0];i++)for(size_t j=0;j<a.getShape()[1];j++)res(i,j)=T(1);
            return res;
        }

        template<Element T>Tensor<T>softmax(const Tensor<T>&a){
            Tensor<T>res(a.getDimension(),a.getShape());
            T mx=a(0,0);
            for(size_t i=0;i<a.getShape()[0];i++)for(size_t j=0;j<a.getShape()[1];j++)if(a(i,j)>mx)mx=a(i,j);
            T sum=T(0);
            for(size_t i=0;i<a.getShape()[0];i++)for(size_t j=0;j<a.getShape()[1];j++){
                res(i,j)=std::exp(a(i,j)-mx);
                sum+=res(i,j);
            }
            for(size_t i=0;i<a.getShape()[0];i++)for(size_t j=0;j<a.getShape()[1];j++)res(i,j)/=sum;
            return res;
        }

        template<Element T>Tensor<T>softmax_d(const Tensor<T>&a){
            Tensor<T>res(a.getDimension(),a.getShape());
            Tensor<T>s=softmax(a);
            for(size_t i=0;i<a.getShape()[0];i++)for(size_t j=0;j<a.getShape()[1];j++)res(i,j)=s(i,j)*(T(1)-s(i,j));
            return res;
        }
    }
}

#endif