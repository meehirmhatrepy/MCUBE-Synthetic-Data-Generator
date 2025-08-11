import json
import os
import numpy as np
import cv2
def write_coco_json(out_path, image_filename, width, height, instances):
    images = [{
        'id': 1,
        'file_name': os.path.basename(image_filename),
        'width': width,
        'height': height
    }]
    annos = []
    cat_ids = set([int(inst['class']) for inst in instances])
    categories = [{'id': cid, 'name': str(cid)} for cid in sorted(cat_ids)]
    aid = 1
    for inst in instances:
        poly = np.array(inst['poly'], dtype=np.float32)
        xs = poly[:,0]; ys = poly[:,1]
        x1 = float(xs.min()); y1 = float(ys.min()); w = float(xs.max() - xs.min()); h = float(ys.max() - ys.min())
        segmentation = [poly.flatten().tolist()]
        area = float(abs(cv2.contourArea(poly)))
        annos.append({
            'id': aid,
            'image_id': 1,
            'category_id': int(inst['class']),
            'segmentation': segmentation,
            'bbox': [x1, y1, w, h],
            'area': area,
            'iscrowd': 0
        })
        aid += 1
    out = {'images': images, 'annotations': annos, 'categories': categories}
    with open(out_path, 'w') as f:
        json.dump(out, f)