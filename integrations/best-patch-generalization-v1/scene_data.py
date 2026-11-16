from source_loader import load
_source=load('_scene_data_generic','scene_data.py')
for _k,_v in vars(_source).items():
 if not _k.startswith('__'):globals()[_k]=_v
