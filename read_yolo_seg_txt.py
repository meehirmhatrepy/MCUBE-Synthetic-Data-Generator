from sanitize_polygon import sanitize_polygon

def read_yolo_seg_txt(txt_path, img_w, img_h):
    instances = []
    with open(txt_path, 'r') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                cls = int(parts[0])
            except Exception:
                print(f"[read_yolo] failed parsing class on line {line_num}: {line}")
                continue
            if len(parts) < 5:
                continue
            try:
                xc = float(parts[1]); yc = float(parts[2]); w = float(parts[3]); h = float(parts[4])
            except Exception:
                print(f"[read_yolo] failed parsing bbox on line {line_num}")
                continue
            poly = []
            if len(parts) > 5:
                try:
                    coords = list(map(float, parts[5:]))
                    for i in range(0, len(coords), 2):
                        nx = coords[i]; ny = coords[i+1]
                        x = nx * img_w
                        y = ny * img_h
                        poly.append((x, y))
                except Exception:
                    print(f"[read_yolo] malformed polygon coords on line {line_num}")
                    # fall back to bbox
                    poly = None
            if poly is None or len(poly) == 0:
                bw = w * img_w; bh = h * img_h
                cx = xc * img_w; cy = yc * img_h
                x1 = cx - bw/2; y1 = cy - bh/2
                x2 = cx + bw/2; y2 = cy + bh/2
                poly = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
            poly = sanitize_polygon(poly, img_w, img_h)
            instances.append({'class': cls, 'poly': poly})
    return instances
