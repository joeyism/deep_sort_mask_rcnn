

def gen_xywh_from_box(box):
    x = int(box[1])  
    y = int(box[0])  
    w = int(box[3]-box[1])
    h = int(box[2]-box[0])
    if x < 0 :
        w = w + x
        x = 0
    if y < 0 :
        h = h + y
        y = 0 
    return x, y, w, h 
