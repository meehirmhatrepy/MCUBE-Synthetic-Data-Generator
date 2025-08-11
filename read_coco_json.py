import json
import os
import numpy as np
from sanitize_polygon import sanitize_polygon
def read_coco_json(json_path, image_filename, img_w, img_h):
    with open(json_path, 'r') as f:
        jd = json.load(f)
    images = {im['file_name']: im for im in jd.get('images', [])}
    if image_filename not in images:
        bname = os.path.basename(image_filename)
        if bname in images:
            img_entry = images[bname]
        else:
            raise ValueError('Image filename not in COCO JSON')
    else:
        img_entry = images[image_filename]
    img_id = img_entry['id']
    instances = []
    for ann in jd.get('annotations', []):
        if ann.get('image_id') != img_id:
            continue
        seg = ann.get('segmentation')
        if seg is None:
            continue
        if isinstance(seg, list) and len(seg) > 0:
            poly_flat = seg[0]
            coords = np.array(poly_flat, dtype=np.float32).reshape(-1,2)
            coords = sanitize_polygon(coords, img_w, img_h)
            instances.append({'class': ann.get('category_id', 0), 'poly': coords})
        else:
            # RLE not handled
            continue
    return instances