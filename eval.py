import os
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from utils.detect_config import FRCNN
from utils.eval_args import *


if __name__ == "__main__":
    image_ids = [image_id.split() for image_id in open(eval_f).readlines()]

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        frcnn = FRCNN(confidence=confidence, nms_iou=nms_iou)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = image_id[0]
            image_id = image_id[0].split('/')[-1][:-4]
            image = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
            frcnn.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")

    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            image_path = image_id[0]
            objs = image_id[1:]
            image_id = image_id[0].split('/')[-1][:-4]
            w, h = Image.open(image_path).size
            with open(os.path.join(map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
                for obj in objs:
                    difficult_flag = False
                    # if obj.find('difficult') != None:
                    #     difficult = obj.find('difficult').text
                    #     if int(difficult) == 1:
                    #         difficult_flag = True
                    obj_name = class_names[int(obj.split(',')[-1])]
                    if obj_name not in class_names:
                        print('error class name')
                        continue
                    bndbox = obj.split(',')[:-1]
                    left = int(bndbox[0])
                    top = int(bndbox[1])
                    right = int(bndbox[2])
                    bottom = int(bndbox[3])

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, score_threhold=score_threhold, path=map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names=class_names, path=map_out_path)
        print("Get map done.")
