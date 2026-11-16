#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>
inline double nmax(double a,double b){return std::isnan(a)||std::isnan(b)?std::numeric_limits<double>::quiet_NaN():(a>b?a:b);}
inline double nmin(double a,double b){return std::isnan(a)||std::isnan(b)?std::numeric_limits<double>::quiet_NaN():(a<b?a:b);}
extern "C" int mpc_kernel_abi(){return 1;}
inline double norm2(double x,double y){return std::sqrt(x*x+y*y);}
inline double sdf(double x,double y,double w,double l){
 double a=std::abs(x)-w,b=std::abs(y)-l;
 return norm2(nmax(a,0.0),nmax(b,0.0))+nmin(nmax(a,b),0.0);
}
extern "C" void rectangle_sat(const double* states,const double* boxes,const double* times,
 const double* er,const double* ef,const double* br,const double* bf,
 std::int64_t n,std::int64_t m,double w,double l,double* out){
 for(std::int64_t i=0;i<n;i++)for(std::int64_t j=0;j<m;j++){
  double dx=boxes[j*9]+times[i]*boxes[j*9+7]-states[i*5];
  double dy=boxes[j*9+1]+times[i]*boxes[j*9+8]-states[i*5+1];
  double best=-std::numeric_limits<double>::infinity();
  for(int k=0;k<4;k++){
   const double* axis=k==0?er+i*2:k==1?ef+i*2:k==2?br+j*2:bf+j*2;
   double x=axis[0],y=axis[1];
   double projection=std::abs(dx*x+dy*y);
   double ego=w*std::abs(er[i*2]*x+er[i*2+1]*y)+l*std::abs(ef[i*2]*x+ef[i*2+1]*y);
   double other=boxes[j*9+3]/2*std::abs(br[j*2]*x+br[j*2+1]*y)+boxes[j*9+4]/2*std::abs(bf[j*2]*x+bf[j*2+1]*y);
   best=nmax(best,projection-ego-other);
  }
  out[i*m+j]=best;
 }
}
extern "C" void segment_envelope(const double* start,const double* direction,const double* radii,
 std::int64_t n,std::int64_t m,double w,double l,double* out){
 for(std::int64_t i=0;i<n;i++)for(std::int64_t j=0;j<m;j++){
  std::int64_t k=(i*m+j)*2;double x=start[k],y=start[k+1],vx=direction[k],vy=direction[k+1];
  double distance=nmin(sdf(x,y,w,l),sdf(x+vx,y+vy,w,l));
  double denom=nmax(vx*vx+vy*vy,1e-20);
  for(int c=0;c<4;c++){
   double cx=(c<2?-w:w),cy=(c%2==0?-l:l);
   double dx=cx-x,dy=cy-y;double f=nmin(1.0,nmax(0.0,(dx*vx+dy*vy)/denom));
   distance=nmin(distance,norm2(dx-f*vx,dy-f*vy));
  }
  bool px=std::abs(vx)<1e-12,py=std::abs(vy)<1e-12;
  double sx=px?1.0:vx,sy=py?1.0:vy;
  double ax=(-w-x)/sx,bx=(w-x)/sx,ay=(-l-y)/sy,by=(l-y)/sy;
  double enter=nmax(px?-INFINITY:nmin(ax,bx),py?-INFINITY:nmin(ay,by));
  double leave=nmin(px?INFINITY:nmax(ax,bx),py?INFINITY:nmax(ay,by));
  bool intersects=nmax(enter,0.0)<=nmin(leave,1.0)&&!(px&&std::abs(x)>w)&&!(py&&std::abs(y)>l);
  if(intersects)distance=nmin(distance,0.0);
  out[i*m+j]=distance-radii[j];
 }
}
extern "C" void point_occupancy(const double* states,const double* points,const double* er,const double* ef,
 std::int64_t n,std::int64_t m,double w,double l,double margin,double radius,double* out){
 for(std::int64_t i=0;i<n;i++){
  double best=std::numeric_limits<double>::infinity();
  for(std::int64_t j=0;j<m;j++){
   double dx=points[j*2]-states[i*5],dy=points[j*2+1]-states[i*5+1];
   double x=dx*er[i*2]+dy*er[i*2+1],y=dx*ef[i*2]+dy*ef[i*2+1];
   best=nmin(best,sdf(x,y,w,l));
  }
  out[i]=best-margin-radius;
 }
}
