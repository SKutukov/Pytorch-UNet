from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2

coco = COCO('/HDD/seg+skelet/lifeguard.json')
img_dir = '/HDD/seg+skelet/coco_datasets/lifeguard/'
mask_dir = "/HDD/seg+skelet/dataset/masks/"
imgs_dir = "/HDD/seg+skelet/dataset/imgs/"

# cat_ids = coco.getCatIds()
cat_ids = [1, 16, 17]
map_cat = {
    16: 1,  # underwater
    17: 2  # upperwater
}
for image_id in coco.getImgIds(catIds=cat_ids):
    img = coco.imgs[image_id]
    image = np.array(Image.open(os.path.join(img_dir, img['file_name'])))
    anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(anns_ids)

    if len(anns) > 0:
        anns_img = np.zeros((img['height'], img['width']))
        per_ans = []
        for ann in anns:
            if ann['category_id'] != 1:
                anns_img = np.maximum(anns_img, coco.annToMask(ann)*map_cat[ann['category_id']])
            else:
                per_ans.append(ann)

        for i, ann in enumerate(per_ans):
            x_min = ann['bbox'][0]
            y_min = ann['bbox'][1]
            x_max = x_min + ann['bbox'][2]  # x_0 + w
            y_max = y_min + ann['bbox'][3]  # y_0 + h
            cv2.imwrite(os.path.join(mask_dir,
                                     f"{img['file_name'].split('.')[0]}_{i}_mask.png"),
                        anns_img[y_min:y_max, x_min:x_max])
            cv2.imwrite(os.path.join(imgs_dir,
                                     f"{img['file_name'].split('.')[0]}_{i}.png"),
                        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[y_min:y_max, x_min:x_max])

# mask = coco.annToMask(anns[0])
# for i in range(len(anns)):
#     mask += coco.annToMask(anns[i])
#
# plt.imshow(mask)
#
# plt.savefig("/home/skutukov/HDD/seg+skelet/temp.png", transparent=True)
