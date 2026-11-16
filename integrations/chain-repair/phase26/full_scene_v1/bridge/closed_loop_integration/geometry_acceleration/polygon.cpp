#include <cmath>
#include <cstdint>
#include <limits>
extern "C" int edited_polygon_abi(){return 1;}
// Projection, trigonometry and broadphase remain NumPy/SciPy operations.
// CSR lists contain the exact original KD-tree candidate sets.
extern "C" void projected_polygon_gap(const double* states,const double* centers,
 const double* axes,const double* lo,const double* hi,const double* body,
 const std::int64_t* starts,const std::int64_t* ids,std::int64_t n,
 std::int64_t axis_count,double margin,double* out){
 for(std::int64_t i=0;i<n;i++){
  double best=std::numeric_limits<double>::infinity();
  for(std::int64_t k=starts[i];k<starts[i+1];k++){
   auto j=ids[k];double dx=centers[2*j]-states[2*i],dy=centers[2*j+1]-states[2*i+1];
   double gap=-std::numeric_limits<double>::infinity();
   for(std::int64_t a=0;a<axis_count;a++){
    auto index=i*axis_count+a;double delta=dx*axes[2*index]+dy*axes[2*index+1];
    double left=delta+lo[index]-body[index];double right=-delta-hi[index]-body[index];
    double value=left>right?left:right;if(value>gap)gap=value;
   }
   if(gap<best)best=gap;
  }
  out[i]=best-margin;
 }
}
