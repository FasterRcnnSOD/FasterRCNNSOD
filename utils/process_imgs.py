from PIL import Image
import pickle
import numpy as np


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


if __name__ == '__main__':
    img_f = open('datasets/COCO_VOC/train.txt', 'r')
    file_lst = img_f.readlines()
    img_f.close()

    for i, line in enumerate(file_lst):
        img_path = line.split()[0]
        img_type = img_path[-4:]
        save_path = img_path.replace(img_type, '.pkl')

        print(f'{i+1}: {img_path} & {save_path}')
        image = Image.open(img_path)
        image = cvtColor(image)
        with open(save_path, 'wb') as f:
            pickle.dump(image, f)