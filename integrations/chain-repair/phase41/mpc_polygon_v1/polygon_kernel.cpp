#include <cstdint>
#include <cmath>
#include <limits>
extern "C" int polygon_kernel_abi(){return 1;}
extern "C" void polygon_csr(const double* xy,const double* axes,const double* lo,const double* hi,const double* body,const double* centers,const int64_t* offsets,const int64_t* ids,int64_t n,int64_t m,int64_t a,int use_ids,double margin,double* out){
 for(int64_t i=0;i<n;++i){
  double best=std::numeric_limits<double>::infinity();
  const int64_t begin=use_ids?offsets[i]:0,end=use_ids?offsets[i+1]:m;
  for(int64_t k=begin;k<end;++k){
   int64_t j=use_ids?ids[k]:k;
   double dx=centers[2*j]-xy[2*i],dy=centers[2*j+1]-xy[2*i+1];
   double gap=-std::numeric_limits<double>::infinity();
   for(int64_t h=0;h<a;++h){
    int64_t q=i*a+h;
    double delta=dx*axes[2*q]+dy*axes[2*q+1];
    double left=delta+lo[q]-body[q],right=-delta-hi[q]-body[q];
    double v=left>right?left:right;if(v>gap)gap=v;
   }
   if(gap<best)best=gap;
  }
  out[i]=best-margin;
 }
}
