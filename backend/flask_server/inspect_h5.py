import h5py
import json

path = r'd:/Major Project/IPD-main/backend/flask_server/models/deepfake_detection_model.h5'
f = h5py.File(path, 'r')

print('=== ROOT ATTRS ===')
for k, v in f.attrs.items():
    try:
        val = v.decode() if isinstance(v, bytes) else str(v)
        if k == 'model_config':
            cfg = json.loads(val)
            print(k, '->', json.dumps(cfg, indent=2)[:6000])
        else:
            print(k, '->', val[:200])
    except Exception as e:
        print(k, '-> (err)', e)

print()
print('=== WEIGHTS ===')
def walk(g, prefix=''):
    for k in g.keys():
        v = g[k]
        if hasattr(v, 'shape'):
            print(prefix + k, '->', v.shape, v.dtype)
        else:
            walk(v, prefix + k + '/')
walk(f)
