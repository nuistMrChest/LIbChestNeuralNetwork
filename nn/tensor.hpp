#ifndef TENSOR_HPP
#define TENSOR_HPP

#include<vector>
#include<iostream>
#include<numeric>
#include<utility>
#include<thread>
#include<array>
#include<initializer_list>
#include<algorithm>

namespace LibCN{

    constexpr size_t MAX_DIM=8;

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

    template<Element T>struct TensorView;

    template<Element T>struct Tensor{
        size_t dimension;
        std::vector<size_t>shape;
        std::vector<size_t>stride;
        std::vector<T>values;

        size_t getDimension()const{
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
            size_t indexes[]={static_cast<size_t>(args)...};
            size_t index=0;
            for(size_t i=0;i<sizeof...(args);++i)index+=indexes[i]*stride[i];
            return values[index];
        }

        template<typename...Args>const T&operator()(Args...args)const{
            size_t indexes[]={static_cast<size_t>(args)...};
            size_t index=0;
            for(size_t i=0;i<sizeof...(args);++i)index+=indexes[i]*stride[i];
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
            for(size_t j=0;j<r;++j)os<<" ";
            os<<"}";
        }

        friend std::ostream&operator<<(std::ostream&os,const Tensor<T>&a){
            if(a.values.size()==0)a.print_n(os);
            else if(a.dimension==0)a.print_0d(os);
            else if(a.dimension==1)a.print_1d(os);
            else if(a.dimension==2)a.print_2d(os);
            else a.print_nd(os,0,0,0,a.values.size());
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

        Tensor<T>hadamard(const TensorView<T>&a)const;

        Tensor<T>&hadamard_self(const Tensor<T>&a){
            if(this->dimension==a.dimension&&this->shape==a.shape)for(size_t i=0;i<this->values.size();i++)this->values[i]*=a.values[i];
            return*this;
        }

        Tensor<T>&hadamard_self(const TensorView<T>&a)const;

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

        Tensor<T>transpose(size_t d1,size_t d2)const{
            if(d1>=dimension||d2>=dimension)return Tensor<T>();
            if(d1==d2)return*this;
            if(dimension == 2){
                Tensor<T>res(2,{shape[d2], shape[d1]});
                size_t rows=shape[0];
                size_t cols=shape[1];
                for(size_t i=0;i<rows;++i){
                    for(size_t j=0;j<cols;++j){
                        res.values[j*res.stride[0]+i*res.stride[1]]=values[i*stride[0]+j*stride[1]];
                    }
                }
                return res;
            }
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

        T accumulate()const{
            return std::accumulate(values.begin(),values.end(),T(0));
        }

        T dot(const Tensor<T>&a)const{
            return(this->hadamard(a).accumulate());
        }

        T dot(const TensorView<T>&a)const;

        static void subMatrixMultiplication(size_t f,size_t m,size_t n,size_t p,Tensor<T>&res,const Tensor<T>&a,const Tensor<T>&b){
            for(size_t i=f;i<m;i++)for(size_t k=0;k<n;++k){
                T aik=a.values[i*a.stride[0]+k*a.stride[1]];
                for(size_t j=0;j<p;++j)res.values[i*res.stride[0]+j*res.stride[1]]+=aik*b.values[k*b.stride[0]+j*b.stride[1]];
            }
        }

        Tensor<T>matrixMultiplication(const Tensor<T>&b,size_t thread_num=0)const{
            const Tensor<T>&a=*this;
            Tensor<T>res;
            if(a.dimension==2&&b.dimension==2&&a.shape[1]==b.shape[0]){
                size_t m=a.shape[0];
                size_t n=a.shape[1];
                size_t p=b.shape[1];
                if(m<((200000/p)/n)||thread_num<=0){
                    res.resize(2, {m, p});
                    for(size_t i=0;i<res.values.size();++i)res.values[i]=T(0);
                    for(size_t i=0;i<m;++i)for(size_t k=0;k<n;++k){
                        T aik=a.values[i*a.stride[0]+k*a.stride[1]];
                        for(size_t j=0;j<p;++j)res.values[i*res.stride[0]+j*res.stride[1]]+=aik*b.values[k*b.stride[0]+j*b.stride[1]];
                    }
                }
                else{
                    res.resize(2, {m, p});
                    for(size_t i=0;i<res.values.size();++i)res.values[i]=T(0);
                    size_t i=0;
                    thread_num=std::max<size_t>(1,std::min(thread_num,m));
                    size_t l=m/thread_num;
                    std::vector<std::thread>ts;
                    while(i<m){
                        if(i+l<m)ts.push_back(std::thread(subMatrixMultiplication,i,i+l,n,p,std::ref(res),std::cref(a),std::cref(b)));
                        else ts.push_back(std::thread(subMatrixMultiplication,i,m,n,p,std::ref(res),std::cref(a),std::cref(b)));
                        i+=l;
                    }
                    for(size_t i=0;i<ts.size();i++)ts[i].join();
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

    template<Element T>struct TensorView{
        Tensor<T>*ori;
        std::array<size_t,MAX_DIM>from;
        std::array<size_t,MAX_DIM>to;
        size_t value_size;
        std::array<size_t,MAX_DIM>viewed_dimension;
        std::array<size_t,MAX_DIM>token_dimension;
        std::array<bool,MAX_DIM>viewed;
        size_t dimension;
        
        TensorView(){
            ori=nullptr;
            value_size=0;
            dimension=0;
        }

            TensorView(Tensor<T>&a,std::initializer_list<size_t>f,std::initializer_list<size_t>t,std::initializer_list<size_t>s,std::initializer_list<bool>sb){
            ori=&a;
            std::copy(f.begin(),f.end(),from.begin());
            std::copy(t.begin(),t.end(),to.begin());
            std::copy(s.begin(),s.end(),token_dimension.begin());
            std::copy(sb.begin(),sb.end(),viewed.begin());
            size_t ci=0;
            for(size_t i=0;i<sb.size();i++)if(viewed[i]){
                viewed_dimension[ci]=i;
                ci++;
            }
            dimension=f.size();
        }

        template<typename...Args>T&operator()(Args...args){
            size_t indexes[]={static_cast<size_t>(args)...};
            size_t index=0;
            size_t ci=0;
            for(size_t i=0;i<ori->dimension;i++){
                if(viewed[i]){
                    index+=((indexes[ci]+from[ci])*(ori->stride[i]));
                    ci++;
                }
                else{
                    index+=(token_dimension[i]*(ori->stride[i]));
                }
            }
            return ori->values[index];
        }

        template<typename...Args>const T&operator()(Args...args)const{
            size_t indexes[]={static_cast<size_t>(args)...};
            size_t index=0;
            size_t ci=0;
            for(size_t i=0;i<ori->dimension;i++){
                if(viewed[i]){
                    index+=((indexes[ci]+from[ci])*(ori->stride[i]));
                    ci++;
                }
                else{
                    index+=(token_dimension[i]*(ori->stride[i]));
                }
            }
            return ori->values[index];
        }

        void print_n(std::ostream&os)const{
            os<<"{ NULL }";
        }

        T&scalar_ref(){
            size_t index=0;
            for(size_t i=0;i<ori->dimension;i++)if(!viewed[i]) index+=token_dimension[i]*ori->stride[i];
            return ori->values[index];
        }

        const T&scalar_ref()const{
            size_t index=0;
            for(size_t i=0;i<ori->dimension;i++)if(!viewed[i])index+=token_dimension[i]*ori->stride[i];
            return ori->values[index];
        }

        void print_0d(std::ostream&os)const{
            os<<"{ "<<scalar_ref()<<" }";
        }

        void print_1d(std::ostream&os)const{
            os<<"{ ";
            size_t len=to[0]-from[0];
            for(size_t i=0;i<len;i++)os<<this->operator()(i)<<" ";
            os<<"}";
        }

        void print_2d(std::ostream&os)const{
            size_t rows=to[0]-from[0];
            size_t cols=to[1]-from[1];
            for(size_t i=0;i<rows;i++){
                if(i==0) os<<"{";
                else os<<" ";
                os<<" ";
                for(size_t j=0;j<cols;j++)os<<this->operator()(i,j)<<" ";
                if(i==rows-1)os<<"}";
                else os<<"\n";
            }
        }

        template<size_t N>T&at_index_array(const std::array<size_t,N>&idx){
            size_t index=0;
            size_t ci=0;
            for(size_t i=0;i<ori->dimension;i++){
                if(viewed[i]){
                    index+=(idx[ci]+from[ci])*ori->stride[i];
                    ci++;
                }
                else index+=token_dimension[i]*ori->stride[i];
            }
            return ori->values[index];
        }

        template<size_t N>const T&at_index_array(const std::array<size_t,N>&idx)const{
            size_t index=0;
            size_t ci=0;
            for(size_t i=0;i<ori->dimension;i++){
                if(viewed[i]){
                    index+=(idx[ci]+from[ci])*ori->stride[i];
                    ci++;
                }
                else index+=token_dimension[i]*ori->stride[i];
            }
            return ori->values[index];
        }

        void print_nd(std::ostream&os,size_t d,size_t r,std::array<size_t,MAX_DIM>&idx)const{
            if(d==dimension-1){
                os<<"{ ";
                size_t len=to[d]-from[d];
                for(size_t i=0;i<len;i++){
                    idx[d]=i;
                    os<<at_index_array(idx);
                    if(i+1<len) os<<" ";
                }
                os<<" }";
                return;
            }

            os<<"{\n";
            size_t len=to[d]-from[d];
            for(size_t i=0;i<len;i++){
                idx[d]=i;
                for(size_t j=0;j<r+2;j++)os<<" ";
                print_nd(os,d+1,r+2,idx);
                if(i+1<len)os<<"\n";
            }
            os<<"\n";
            for(size_t j=0;j<r;j++)os<<" ";
            os<<"}";
        }

        friend std::ostream&operator<<(std::ostream&os,const TensorView<T>&a){
            if(a.ori==nullptr)a.print_n(os);
            else if(a.dimension==0)a.print_0d(os);
            else if(a.dimension==1)a.print_1d(os);
            else if(a.dimension==2)a.print_2d(os);
            else{
                std::array<size_t,MAX_DIM>idx{};
                a.print_nd(os,0,0,idx);
            }
            return os;
        }

        std::vector<size_t>getShape()const{
            std::vector<size_t>res;
            for(size_t i=0;i<from.size();i++)res.push_back(to[i]-from[i]);
            return res;
        }

        template<typename...Args>size_t getValuesIndex(Args...args)const{
            constexpr size_t N=sizeof...(Args);
            std::array<size_t,N>idx{static_cast<size_t>(args)...};
            size_t index=0;
            size_t ci=0;
            for(size_t i=0;i<ori->dimension;i++){
                if(viewed[i]){
                    index+=(idx[ci]+from[ci])*ori->stride[i];
                    ci++;
                }
                else index+=token_dimension[i]*ori->stride[i];
            }
            return index;
        }

        size_t fakeValueSize()const{
            if(ori==nullptr)return 0;
            if(dimension==0)return 1;
            size_t ans=1;
            for(size_t i=0;i<dimension;i++)ans*=to[i]-from[i];
            return ans;
        }

        size_t getValuesIndexFromFake(size_t fake_index)const{
            size_t total=fakeValueSize();
            if(dimension==0){
                size_t real_index=0;
                for(size_t i=0;i<ori->dimension;i++)if(!viewed[i]) real_index+=token_dimension[i]*ori->stride[i];
                return real_index;
            }
            std::array<size_t,MAX_DIM>idx{};
            size_t rem=fake_index;
            for(size_t d=dimension;d>0;d--){
                size_t k=d-1;
                size_t len=to[k]-from[k];
                idx[k]=rem%len;
                rem/=len;
            }
            size_t real_index=0;
            size_t ci=0;
            for(size_t i=0;i<ori->dimension;i++){
                if(viewed[i]){
                    real_index+=(idx[ci]+from[ci])*ori->stride[i];
                    ci++;
                }
                else real_index+=token_dimension[i]*ori->stride[i];
            }
            return real_index;
        }

        Tensor<T>operator+(const TensorView<T>&a)const{
            Tensor<T>res;
            if(this->getShape()==a.getShape()){
                res.resize(a.dimension,a.getShape());
                for(size_t i=0;i<res.values.size();i++)res.values[i]=ori->values[getValuesIndexFromFake(i)]+a.ori->values[a.getValuesIndexFromFake(i)];
            }
            return res;
        }

        TensorView<T>&operator+=(const TensorView<T>&a){
            if(this->getShape()==a.getShape())for(size_t i=0;i<this->fakeValueSize();i++)this->ori->values[getValuesIndexFromFake(i)]+=a.ori->values[getValuesIndexFromFake(i)];
            return*this;
        }

        Tensor<T>operator+(const Tensor<T>&a)const{
            Tensor<T>res;
            if(getShape()==a.getShape()){
                res.resize(a.dimension,a.shape());
                for(size_t i=0;i<res.values.size();i++)res.values[i]=ori->values[getValuesIndexFromFake(i)]+a.values[i];
            }
            return res;
        }

        TensorView<T>&operator+=(const Tensor<T>&a){
            if(getShape()==a.getShape())for(size_t i=0;i<this->fakeValueSize();i++)this->ori->values[getValuesIndexFromFake(i)]+=a.values[i];
            return*this;
        }

        friend Tensor<T>operator+(const Tensor<T>&a,const TensorView<T>&b){
            return b+a;
        }

        friend Tensor<T>&operator+=(Tensor<T>&a,const TensorView<T>&b){
            if(a.getShape()==b.getShape())for(size_t i=0;i<a.values.size();i++)a.values[i]+=b.ori->values[b.getValuesIndexFromFake(i)];
            return a;
        }

        Tensor<T>operator-(const TensorView<T>&a)const{
            Tensor<T>res;
            if(this->getShape()==a.getShape()){
                res.resize(a.dimension,a.getShape());
                for(size_t i=0;i<res.values.size();i++)res.values[i]=ori->values[getValuesIndexFromFake(i)]-a.ori->values[a.getValuesIndexFromFake(i)];
            }
            return res;
        }

        TensorView<T>&operator-=(const TensorView<T>&a){
            if(this->getShape()==a.getShape())for(size_t i=0;i<this->fakeValueSize();i++)this->ori->values[getValuesIndexFromFake(i)]-=a.ori->values[getValuesIndexFromFake(i)];
            return*this;
        }

        Tensor<T>operator-(const Tensor<T>&a)const{
            Tensor<T>res;
            if(getShape()==a.getShape()){
                res.resize(a.dimension,a.shape());
                for(size_t i=0;i<res.values.size();i++)res.values[i]=ori->values[getValuesIndexFromFake(i)]-a.values[i];
            }
            return res;
        }

        TensorView<T>&operator-=(const Tensor<T>&a){
            if(getShape()==a.getShape())for(size_t i=0;i<this->fakeValueSize();i++)this->ori->values[getValuesIndexFromFake(i)]-=a.values[i];
            return*this;
        }

        friend Tensor<T>operator-(const Tensor<T>&a,const TensorView<T>&b){
            Tensor<T>res;
            if(a.getShape()==b.getShape()){
                res.resize(a.dimension,a.getShape());
                for(size_t i=0;i<a.values.size();i++)res.values[i]=a.values[i]-b.ori->values[b.getValuesIndexFromFake(i)];
            }
            return res;
        }

        friend Tensor<T>&operator-=(Tensor<T>&a,const TensorView<T>&b){
            if(a.getShape()==b.getShape())for(size_t i=0;i<a.values.size();i++)a.values[i]-=b.ori->values[b.getValuesIndexFromFake(i)];
            return a;
        }

        Tensor<T>hadamard(const TensorView<T>&a)const{
            Tensor<T>res;
            if(this->getShape()==a.getShape()){
                res.resize(this->dimension,this->getShape());
                for(size_t i=0;i<this->fakeValueSize();i++)res.values[i]=this->ori->values[this->getValuesIndexFromFake(i)]*a.ori->values[getValuesIndexFromFake(i)];
            }
            return res;
        }

        Tensor<T>hadamard(const Tensor<T>&a)const{
            Tensor<T>res;
            if(this->getShape()==a.getShape()){
                res.resize(this->dimension,this->getShape());
                for(size_t i=0;i<a.values.size();i++)res.values[i]=this->ori->values[getValuesIndexFromFake(i)]*a.values[i];
            }
            return res;
        }

        TensorView<T>&hadamard_self(const TensorView<T>&a){
            if(this->getShape()==a.getShape())for(size_t i=0;i<this->ori->values.size();i++)this->ori->values[this->getValuesIndexFromFake(i)]*=a,ori->values[a.getValuesIndexFromFake(i)];
            return*this;
        }

        TensorView<T>&hadamard_self(const Tensor<T>&a){
            if(this->getShape()==a.getShape())for(size_t i=0;i<this->ori->values.size();i++)this->ori->values[this->getValuesIndexFromFake(i)]*=a.values[i];
            return*this;
        }

        Tensor<T>operator*(const T&a)const{
            Tensor<T>res(dimension,getShape());
            for(size_t i=0;i<res.values.size();i++)res.values[i]=ori->values[getValuesIndexFromFake(i)]*a;
            return res;
        }

        TensorView<T>&operator*=(const T&a){
            for(size_t i=0;i<fakeValueSize();i++)this->ori->values[getValuesIndexFromFake(i)]*=a;
            return*this;
        }

        friend Tensor<T>operator*(const T&a,const TensorView&b){
            return b*a;
        }

        Tensor<T>materialize()const{
            Tensor<T>res(dimension,getShape());
            for(size_t i=0;i<fakeValueSize();i++)res.values[i]=ori->values[getValuesIndexFromFake(i)];
            return res;
        }

        Tensor<T>transpose(size_t d1,size_t d2)const{
            return materialize().transpose_self(d1,d2);
        }

        std::vector<size_t>unravel_index(size_t index)const{
            std::vector<size_t>idx(dimension);
            for(size_t i=0;i<dimension;++i){
                idx[i]=index/ori->stride[viewed_dimension[i]];
                index%=ori->stride[viewed_dimension[i]];
            }
            return idx;
        }

        size_t ravel_index(const std::vector<size_t>&idx)const{
            size_t index=0;
            for(size_t i=0;i<idx.size();++i)index+=idx[i]*ori->stride[viewed_dimension[i]];
            return index;
        }

        Tensor<T>sum(size_t axis)const{
            std::vector<size_t>s=getShape();
            s.erase(s.begin()+axis);
            Tensor<T>res(dimension-1,s);
            for(size_t i=0;i<res.values.size();i++)res.values[i]=T(0);
            for(size_t i=0;i<this->fakeValueSize();i++){
                std::vector<size_t>idx=unravel_index(i);
                idx.erase(idx.begin()+axis);
                size_t ri=0;
                for(size_t j=0;j<idx.size();j++)ri+=idx[j]*res.stride[j];
                res.values[ri]+=ori->values[getValuesIndexFromFake(i)];
            }
            return res;
        }

        T accumulate()const{
            T res(0);
            for(size_t i=0;i<fakeValueSize();i++)res+=ori->values[getValuesIndexFromFake(i)];
            return res;
        }

        T dot(const TensorView<T>&a)const{
            T res(0);
            if(this->getShape()==a.getShape())for(size_t i=0;i<this->fakeValueSize();i++)res+=this->ori->values[this->getValuesIndexFromFake(i)]*a.ori->values[a.getValuesIndexFromFake(i)];
            return res;
        }

        T dot(const Tensor<T>&a)const{
            T res(0);
            if(this->getShape()==a.getShape())for(size_t i=0;i<a.values.size();i++)res+=this->ori->values[this->getValuesIndexFromFake(i)]*a.values[i];
            return res;
        }
    };

    template<Element T>Tensor<T>Tensor<T>::hadamard(const TensorView<T>&a)const{
        Tensor<T>res;
        if(this->getShape()==a.getShape()){
            res.resize(this->dimension,this->getShape());
            for(size_t i=0;i<this->values.size();i++)res.values[i]=this->values[i]*a.ori->values[a.getValuesIndexFromFake(i)];
        }
        return res;
    }

    template<Element T>Tensor<T>&Tensor<T>::hadamard_self(const TensorView<T>&a)const{
        if(this->getShape()==a.getShape())for(size_t i=0;i<this-<values.size();i++)this->values[i]*=a.ori->values[a.getValuesIndexFromFake(i)];
        return*this;
    }

    template<Element T>T Tensor<T>::dot(const TensorView<T>&a)const{
        T res(0);
        if(this->getShape()==a.getShape)for(size_t i=0;i<this->values.size();i++)res+=this->values[i]*a.ori->values[a.getValuesIndexFromFake(i)];
        return res;
    }
}

#endif