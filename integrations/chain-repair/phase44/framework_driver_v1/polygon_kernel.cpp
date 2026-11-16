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

inline double nmax(double a,double b){return std::isnan(a)||std::isnan(b)?std::numeric_limits<double>::quiet_NaN():(a>b?a:b);}
inline double nmin(double a,double b){return std::isnan(a)||std::isnan(b)?std::numeric_limits<double>::quiet_NaN():(a<b?a:b);}

inline double norm2(double x,double y){return std::sqrt(x*x+y*y);}
inline double sdf(double x,double y,double w,double l){
 double a=std::abs(x)-w,b=std::abs(y)-l;
 return norm2(nmax(a,0.0),nmax(b,0.0))+nmin(nmax(a,b),0.0);
}

extern "C" void occupancy_groups(const double* states,const double* points,const double* er,const double* ef,const int64_t* starts,const int64_t* ends,int64_t n,double w,double l,double margin,double radius,double* out){
 for(int64_t i=0;i<n;++i){
  double best=std::numeric_limits<double>::infinity();
  for(int64_t j=starts[i];j<ends[i];++j){
   double dx=points[j*2]-states[i*5],dy=points[j*2+1]-states[i*5+1];
   double x=dx*er[i*2]+dy*er[i*2+1],y=dx*ef[i*2]+dy*ef[i*2+1];
   best=nmin(best,sdf(x,y,w,l));
  }
  out[i]=best-margin-radius;
 }
}
