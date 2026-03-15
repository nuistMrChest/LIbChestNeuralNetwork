#ifndef TENSOR_HPP
#define TENSOR_HPP

#include<vector>
#include<iostream>
#include<numeric>
#include<utility>

namespace LibCN{
    template<typename T>concept Element=requires(T a,T b,std::iostream&os){
        {a+b}->std::same_as<T>;
        {a+=b}->std::same_as<T&>;
        {a-b}->std::same_as<T>;
        {a-=b}->std::same_as<T&>;
        {a*b}->std::same_as<T>;
        {a*=b}->std::same_as<T&>;
        {os<<a}->std::same_as<std::ostream&>;
        {a>b}->std::same_as<bool>;
        {a<b}->std::same_as<bool>;
        {a>=b}->std::same_as<bool>;
        {a<=b}->std::same_as<bool>;
        {a==b}->std::same_as<bool>;
        {a!=b}->std::same_as<bool>;

        {a/b}->std::same_as<T>;
    };

    template<Element T>struct Tensor{
        size_t dimension;
        std::vector<size_t>shape;
        std::vector<size_t>stride;
        std::vector<T>values;

        size_t getDimention()const{
            return dimension;
        }

        const std::vector<size_t>&getShape()const{
            return shape;
        }

        std::vector<size_t>unravel_index(size_t index)const{
            std::vector<size_t>idx(dimension);
            for(size_t i=0;i<dimension;++i){
                idx[i]=index/stride[i];
                index%=stride[i];
            }
            return idx;
        }

        size_t ravel_index(const std::vector<size_t>&idx)const{
            size_t index=0;
            for(size_t i=0;i<idx.size();++i)index+=idx[i]*stride[i];
            return index;
        }

        void setStride(){
            stride.resize(dimension);
            if(dimension==0)return;
            stride.back()=1;
            for(size_t i=1;i<dimension;i++){
                size_t j=dimension-i-1;
                stride[j]=stride[j+1]*shape[j+1];
            }
        }

        void resize(size_t d,const std::vector<size_t>&s){
            this->dimension=d;
            this->shape=s;
            size_t size=1;
            for(size_t i=0;i<d;i++)size*=s[i];
            values.resize(size);
            this->setStride();
        }

        Tensor(){
            dimension=0;
            shape.resize(0);
            values.resize(0);
            this->setStride();
        }

        Tensor(size_t d,const std::vector<size_t>&s){
            this->dimension=d;
            this->shape=s;
            size_t size=1;
            for(size_t i=0;i<d;i++)size*=s[i];
            values.resize(size);
            this->setStride();
        }

        Tensor(const Tensor<T>&a){
            this->dimension=a.dimension;
            this->shape=a.shape;
            this->values=a.values;
            this->stride=a.stride;
        }

        Tensor(const T&a){
            dimension=0;
            shape.resize(0);
            values.resize(1);
            values[0]=a;
            this->setStride();
        }

        Tensor(const std::vector<T>&a){
            dimension=1;
            shape.resize(1);
            shape[0]=a.size();
            values=a;
            this->setStride();
        }

        template<typename...Args>T&operator()(Args...args){
            std::vector<size_t>indexes={static_cast<size_t>(args)...};
            size_t index=0;
            for(size_t i=0;i<indexes.size();++i)index+=indexes[i]*stride[i];
            return values[index];
        }

        template<typename...Args>const T&operator()(Args...args)const{
            std::vector<size_t>indexes={static_cast<size_t>(args)...};
            size_t index=0;
            for(size_t i=0;i<indexes.size();++i)index+=indexes[i]*stride[i];
            return values[index];
        }

        void print_n(std::ostream&os)const{
            os<<"{ NULL }";
        }

        void print_0d(std::ostream&os)const{
            os<<"{ "<<values[0]<<" }";
        }

        void print_1d(std::ostream&os)const{
            os<<"{ ";
            for(size_t i=0;i<shape[0];i++)os<<values[i]<<" ";
            os<<"}";
        }

        void print_2d(std::ostream&os)const{
            for(size_t i=0;i<shape[0];i++){
                if(i==0)os<<"{";
                else os<<" ";
                os<<" ";
                for(size_t j=0;j<shape[1];j++)os<<this->operator()(i,j)<<" ";
                if(i==shape[0]-1)os<<"}";
                else os<<"\n";
            }
        }

        void print_nd(std::ostream&os,size_t d,size_t r,size_t from,size_t to)const{
            if(d==dimension-1){
                os<<"{ ";
                for(size_t i=0;i<shape[d];++i){
                    os<<values[from+i*stride[d]];
                    if(i+1<shape[d])os<<" ";
                }
                os<<" }";
                return;
            }
            os << "{\n";
            for(size_t i=0;i<shape[d];++i){
                for(size_t j=0;j<r+2;++j)os<<" ";
                size_t next_from=from+i*stride[d];
                size_t next_to=next_from+stride[d];
                print_nd(os,d+1,r+2,next_from,next_to);
                if(i+1<shape[d])os<<"\n";
            }
            os<<"\n";
            for(size_t j = 0; j < r; ++j) os << " ";
            os<<"}";
        }

        friend std::ostream&operator<<(std::ostream&os,const Tensor<T>&a){
            if(a.values.size()==0)a.print_n(os);
            else if(a.dimension==0)a.print_0d(os);
            else if(a.dimension==1)a.print_1d(os);
            else if(a.dimension==2)a.print_2d(os);
            else a.print_nd(os, 0, 0, 0, a.values.size());
            return os;
        }

        static Tensor<T>matrix(std::initializer_list<std::initializer_list<T>>a){
            Tensor<T>res;
            res.dimension=2;
            res.shape.resize(2);
            res.shape[0]=a.size();
            res.shape[1]=a.begin()->size();
            res.values.reserve(res.shape[0]*res.shape[1]);
            for(auto&row:a)res.values.insert(res.values.end(),row.begin(),row.end());
            res.setStride();
            return res;
        }

        Tensor(std::initializer_list<T>a){
            dimension=1;
            shape={a.size()};
            values.assign(a.begin(),a.end());
            setStride();
        }

        Tensor(std::initializer_list<std::initializer_list<T>>a){
            dimension=2;
            shape.resize(2);
            shape[0]=a.size();
            shape[1]=a.begin()->size();
            values.reserve(shape[0]*shape[1]);
            for(auto&row:a)values.insert(values.end(),row.begin(),row.end());
            setStride();
        }

        Tensor(std::initializer_list<std::initializer_list<std::initializer_list<T>>>a){
            dimension = 3;
            shape.resize(3);
            shape[0]=a.size();
            shape[1]=a.begin()->size();
            shape[2]=a.begin()->begin()->size();
            for(auto&m:a)for(auto&row:m)values.insert(values.end(),row.begin(),row.end());
            setStride();
        }

        Tensor(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<T>>>>a){
            dimension = 4;
            shape.resize(4);
            shape[0]=a.size();
            shape[1]=a.begin()->size();
            shape[2]=a.begin()->begin()->size();
            shape[3]=a.begin()->begin()->begin()->size();
            for(auto&b:a)for(auto&m:b)for(auto&row:m)values.insert(values.end(),row.begin(),row.end());
            setStride();
        }

        Tensor<T>operator+(const Tensor<T>&a)const{
            Tensor<T>res;
            if(this->dimension==a.dimension&&this->shape==a.shape){
                res.resize(this->dimension,this->shape);
                for(size_t i=0;i<this->values.size();i++)res.values[i]=this->values[i]+a.values[i];
            }
            return res;
        }

        Tensor<T>&operator+=(const Tensor<T>&a){
            if(this->dimension==a.dimension&&this->shape==a.shape)for(size_t i=0;i<this->values.size();i++)this->values[i]+=a.values[i];
            return*this;
        }

        Tensor<T>operator-(const Tensor<T>&a)const{
            Tensor<T>res;
            if(this->dimension==a.dimension&&this->shape==a.shape){
                res.resize(this->dimension,this->shape);
                for(size_t i=0;i<this->values.size();i++)res.values[i]=this->values[i]-a.values[i];
            }
            return res;
        }

        Tensor<T>&operator-=(const Tensor<T>&a){
            if(this->dimension==a.dimension&&this->shape==a.shape)for(size_t i=0;i<this->values.size();i++)this->values[i]-=a.values[i];
            return*this;
        }

        Tensor<T>hadamard(const Tensor<T>&a)const{
            Tensor<T>res;
            if(this->dimension==a.dimension&&this->shape==a.shape){
                res.resize(this->dimension,this->shape);
                for(size_t i=0;i<this->values.size();i++)res.values[i]=this->values[i]*a.values[i];
            }
            return res;
        }

        Tensor<T>&hadamard_self(const Tensor<T>&a){
            if(this->dimension==a.dimension&&this->shape==a.shape)for(size_t i=0;i<this->values.size();i++)this->values[i]*=a.values[i];
            return*this;
        }

        Tensor<T>operator*(const T&a)const{
            Tensor<T>res(this->dimension,this->shape);
            for(size_t i=0;i<res.values.size();i++)res.values[i]=this->values[i]*a;
            return res;
        }

        Tensor<T>&operator*=(const T&a){
            for(size_t i=0;i<values.size();i++){
                values[i]*=a;
            }
            return*this;
        }

        friend Tensor<T>operator*(const T&a,const Tensor<T>&b){
            return b*a;
        }

        Tensor<T>operator*(const Tensor<T>&a)const{
            if(a.dimension==0)return this->operator*(a.values[0]);
            else if(this->dimension==0)return a*(this->values[0]);
            else return Tensor<T>();
        }

        Tensor<T>&operator*=(const Tensor<T>&a){
            if(a.dimension==0)this->operator*=(a.values[0]);
            return*this;
        }

        Tensor<T>transpose(size_t d1,size_t d2)const{
            if(d1>=dimension||d2>=dimension)return Tensor<T>();
            if(d1==d2)return*this;
            std::vector<size_t>new_shape=shape;
            std::swap(new_shape[d1],new_shape[d2]);
            Tensor<T>res(dimension,new_shape);
            for(size_t i=0;i<values.size();++i){
                std::vector<size_t>idx=unravel_index(i);
                std::swap(idx[d1],idx[d2]);
                res.values[res.ravel_index(idx)]=values[i];
            }
            return res;
        }

        Tensor<T>&transpose_self(size_t d1,size_t d2){
            if(d1==d2)return*this;
            std::vector<size_t>new_shape=shape;
            std::swap(new_shape[d1],new_shape[d2]);
            Tensor<T>res(dimension,new_shape);
            for(size_t i=0;i<values.size();++i){
                std::vector<size_t>idx=unravel_index(i);
                std::swap(idx[d1],idx[d2]);
                res.values[res.ravel_index(idx)]=values[i];
            }
            this->shape=std::move(res.shape);
            this->stride=std::move(res.stride);
            this->values=std::move(res.values);
            return*this;
        }

        Tensor<T>sum(size_t axis)const{
            std::vector<size_t>s=shape;
            s.erase(s.begin()+axis);
            Tensor<T>res(dimension-1,s);
            for(size_t i=0;i<res.values.size();++i)res.values[i]=T(0);
            for(size_t i=0;i<values.size();++i){
                std::vector<size_t>idx=unravel_index(i);
                idx.erase(idx.begin()+axis);
                size_t ri=0;
                for(size_t j=0;j<idx.size();++j)ri+=idx[j]*res.stride[j];
                res.values[ri]+=values[i];
            }
            return res;
        }

        Tensor<T>accumulate()const{
            Tensor<T>res(0,std::vector<size_t>());
            res.values[0]=std::accumulate(values.begin(),values.end(),T(0));
            return res;
        }

        Tensor<T>dot(const Tensor<T>&a)const{
            return (this->hadamard(a)).accumulate();
        }

        Tensor<T>matrixMultiplication(const Tensor<T>&b)const{
            const Tensor<T>&a=*this;
            Tensor<T>res;
            if(a.dimension==2&&b.dimension==2&&a.shape[1]==b.shape[0]){
                res.resize(2,{a.shape[0],b.shape[1]});
                for(size_t i=0;i<a.shape[0];i++)for(size_t j=0;j<b.shape[1];j++){
                    res(i,j)=T(0);
                    for(size_t k=0;k<a.shape[1];k++)res(i,j)+=a(i,k)*b(k,j);
                }
            }
            return res;
        }

        Tensor<T>ascend()const{
            std::vector<size_t>s=shape;
            s.insert(s.begin(),1);
            Tensor<T>res(dimension+1,s);
            res.values=values;
            return res;
        }

        Tensor<T>&ascend_self(){
            dimension++;
            shape.insert(shape.begin(),1);
            this->setStride();
            return*this;
        }
    };
}

#endif