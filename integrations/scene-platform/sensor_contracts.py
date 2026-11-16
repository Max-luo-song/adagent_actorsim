"""Optional diagnostic sensors. Pure JSON validation; default is zero work."""
import copy,math
DEFAULT=dict(enabled=False,provider='gaussian_depth_scan_v1',record_every_n_steps=1,
    sensor_to_ego=[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,1.8],[0.,0.,0.,1.]],
    min_range_m=.5,max_range_m=100.,minimum_opacity=.9,cube_resolution=128,
    channels=32,azimuth_samples=720,elevation_min_deg=-25.,elevation_max_deg=15.,camera_pixel_stride=8)
PROVIDERS=['camera_depth_pointcloud_v1','gaussian_depth_scan_v1']
SCHEMA=dict(type='object',additionalProperties=False,properties={
    'enabled':dict(type='boolean',default=False),'provider':dict(enum=PROVIDERS),
    'record_every_n_steps':dict(type='integer',minimum=1,maximum=188),
    'sensor_to_ego':dict(type='array',minItems=4,maxItems=4,items=dict(type='array',minItems=4,maxItems=4,items=dict(type='number'))),
    **{k:dict(type='number') for k in ['min_range_m','max_range_m','minimum_opacity','elevation_min_deg','elevation_max_deg']},
    **{k:dict(type='integer') for k in ['cube_resolution','channels','azimuth_samples','camera_pixel_stride']}})

def configuration(value=None):
    if value is None:value={}
    if not isinstance(value,dict) or set(value)-set(DEFAULT):raise ValueError('Unknown sensor configuration field')
    c=copy.deepcopy(DEFAULT);c.update(copy.deepcopy(value))
    if type(c['enabled']) is not bool or c['provider'] not in PROVIDERS:raise ValueError('Unsupported sensor switch/provider')
    for k,lo,hi in [('record_every_n_steps',1,188),('cube_resolution',32,512),('channels',1,128),('azimuth_samples',16,2048),('camera_pixel_stride',1,64)]:
        if type(c[k]) is not int or not lo<=c[k]<=hi:raise ValueError('Invalid sensor parameter '+k)
    for k in ['min_range_m','max_range_m','minimum_opacity','elevation_min_deg','elevation_max_deg']:
        if type(c[k]) not in (float,int) or not math.isfinite(c[k]):raise ValueError('Nonfinite/invalid sensor parameter '+k)
    if not 0<c['min_range_m']<c['max_range_m']<=200 or not 0<c['minimum_opacity']<=1:raise ValueError('Invalid range/opacity')
    if not -89<=c['elevation_min_deg']<=c['elevation_max_deg']<=89:raise ValueError('Invalid beam elevation')
    m=c['sensor_to_ego']
    if not isinstance(m,list) or len(m)!=4 or any(not isinstance(row,list) or len(row)!=4 for row in m):raise ValueError('Expected 4x4 sensor_to_ego')
    if any(type(x) not in (int,float) or not math.isfinite(x) for row in m for x in row):raise ValueError('Invalid sensor pose')
    if any(abs(m[3][j]-[0,0,0,1][j])>1e-6 for j in range(4)):raise ValueError('Invalid homogeneous row')
    r=[row[:3] for row in m[:3]]
    if any(abs(sum(r[k][i]*r[k][j] for k in range(3))-(i==j))>1e-5 for i in range(3) for j in range(3)):raise ValueError('Nonrigid sensor rotation')
    det=sum(r[0][i]*(r[1][(i+1)%3]*r[2][(i+2)%3]-r[1][(i+2)%3]*r[2][(i+1)%3]) for i in range(3))
    if abs(det-1)>1e-5 or max(abs(row[3]) for row in m[:3])>10:raise ValueError('Invalid sensor mounting pose')
    return c

def capabilities():
    return dict(default_enabled=False,providers=PROVIDERS,consumer='diagnostic recorder only',
        source='current edited Gaussian scene',units='metres, seconds, radians; sensor x-forward/y-left/z-up',
        gaussian_scan=dict(horizontal_fov_deg=360,independent_mount=True,depth_semantics='expected camera-Z sampled on configured beams; approximate, not guaranteed first return',
            physical_lidar_certified=False,intensity=False,motion_distortion=False),
        camera_pointcloud=dict(camera_FOV_only=True,pseudo_pointcloud=True),radar=dict(available=False),
        feeds_UniAD=False,feeds_controller=False)
