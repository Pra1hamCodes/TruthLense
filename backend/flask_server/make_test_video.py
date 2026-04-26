import os, glob, cv2
import sys
label = sys.argv[1] if len(sys.argv) > 1 else 'Real'
src = rf'd:/Major Project/Dataset/Test/{label}'
out = rf'd:/Major Project/IPD-main/backend/flask_server/uploads/_test_{label.lower()}.mp4'
files = sorted(glob.glob(os.path.join(src, '*.jpg')))[:60]
if not files:
    raise SystemExit('no frames')
frame = cv2.imread(files[0])
h, w = frame.shape[:2]
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))
for fp in files:
    img = cv2.imread(fp)
    if img is None: continue
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h))
    vw.write(img)
vw.release()
print('wrote', out, 'frames:', len(files))
