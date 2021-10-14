import cv2
import os, shutil
import albumentations as A
from albumentations.pytorch import ToTensorV2
import csv
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import  StratifiedShuffleSplit



class preProcessing():

    def __init__(self):
        self.settings_location = '/home/bbejczy/repos/GALIROOT/config/preprocessing.json'
        self.settings = self.open_settings()
        self.img_list = self.list_files(self.settings["directory"],self.settings["image_format"])
        self.ann_list = []
        self.p_d_list = []
        self.p_rgb_list = []
        self.l515_d = self.settings["directory"]+"l515_d/"
        self.l515_rgb = self.settings["directory"]+"l515_rgb/"

    def open_settings(self):
        with open(self.settings_location) as j:
            settings = json.load(j)
        return settings

    @staticmethod
    def list_files(directory, fileformat):
        img_list = []
        for root, dirs, files in os.walk(directory):
            for name in files:
                if name.endswith(fileformat):
                    # print(root)
                    # print(name)
                    img_list.append(os.path.join(root, name))
        return img_list

    def sort_files(self, list_again=True):

        os.chdir(self.settings["directory"]) # FIXME: not universal
        try:
            os.mkdir('l515_d')
            os.mkdir('l515_rgb')
        except FileExistsError:
            print("Directories already exist.")
            pass

        for file in self.img_list:

            if "depth" in file:

                try: 
                    shutil.copyfile(file, self.l515_d+file.split('/')[-1])
                except shutil.Error:
                    print(f"{file.split('/')[-1]} already exists")
                    pass

            elif "rgb" in file:

                try: 
                    shutil.copyfile(file, self.l515_rgb+file.split('/')[-1])
                except shutil.Error:
                    print(f"{file.split('/')[-1]} already exists")
                    pass

            else:

                raise Exception("Image could not be cathegorized. Check file naming.")

        if list_again:
            self.p_d_list = self.list_files(self.l515_d, self.settings["image_format"])
            self.p_rgb_list = self.list_files(self.l515_rgb, self.settings["image_format"])

    @staticmethod
    def clean_json(directory,fileformat,removable_text = ".png"): # Supervisely leaves the .png in the image name

        for root, dirs, files in os.walk(directory):
            for name in files:
                if name.endswith(fileformat):
                    with open(directory+name) as j:
                        annotation = json.load(j)
                        print(type(annotation['objects']))

                    # remove empty annotation files
                    if annotation['objects']:
                        new_name = name.replace(removable_text,'')
                        os.rename(directory+name, directory+new_name)
                    else:
                        os.remove(directory+name)

        print("{} annotation files renamed and kept".format(len(files)))

    @staticmethod
    def vis_keypoints(image, keypoints, color=(0, 255, 0), diameter=10):

        image = image.copy()

        for (x, y) in keypoints:
            cv2.circle(image, (int(x), int(y)), diameter, (0, 255, 0), -1)

        cv2.imshow('image',image)
        cv2.waitKey(1000)


    def augmentations(self, img_dir, ann_dir):


        img_list = self.list_files(img_dir,self.settings["image_format"])
        ann_list = self.list_files(ann_dir, self.settings["annotation_format"])

        # read ROI from json
        x = self.settings['ROI']['x']
        y = self.settings['ROI']['y']

        # define augmentation pipeline
        transform = A.Compose([
            A.Normalize(mean=0, std=1),
            A.Crop(x_min=x[0], y_min=y[0], x_max=x[1], y_max=y[1]),
            A.Resize(height = self.settings['size']['height'], width = self.settings['size']['width']),
            # ToTensorV2()
        ],
        keypoint_params=A.KeypointParams(format='xy')
        )

        count = 0

        for img_name in img_list:

            img_name_stem=str(Path(img_name).stem)

            for ann_name in ann_list:

                ann_name_stem=str(Path(ann_name).stem)

        
                if img_name_stem == ann_name_stem:

                    with open(ann_name) as j:
                       ann_file = json.load(j)
                    kp_raw = ann_file['objects'][0]['points']['exterior'][0]
                    keypoints = [(kp_raw[0],kp_raw[1])] # convert from list to tuple

                    count += 1

                    img = cv2.imread(img_name)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    transformed = transform(image=img, keypoints=keypoints)

                    print(transformed['keypoints'])
                    
                    self.vis_keypoints(transformed['image'], transformed['keypoints'])

    #     print(count)

    @staticmethod
    def train_test_split(img_dir, ann_dir, train_size = 0.8):

        img_list = preProcessing.list_files(img_dir, '.png')
        ann_list = preProcessing.list_files(ann_dir, '.json')
        name_list =[]
        for name in ann_list:
            name_stem = Path(name).stem
            # print(name_stem)
            name_list.append(name_stem)







def main():

    # Unit test
    pp = preProcessing()
    # pp.sort_files()
    # preProcessing.clean_json(pp.settings['annotations'], pp.settings['annotation_format'])
    # pp.ann_list = preProcessing.list_files(pp.settings['annotations'],pp.settings['annotation_format'])


    # print(len(pp.list_files()))
    # print(pp.p_d_list[0])
    # print(pp.p_rgb_list[0])
    # pp.augmentations(pp.l515_rgb, pp.settings['annotations'])
    # pp.augmentations('/home/bbejczy/repos/GALIROOT/data/test_set/img/', pp.settings['annotations'])
    # pp.annotations = preProcessing.list_files(pp.settings['annotations'],pp.settings['annotation_format'])
    # print(pp.ann_list)

    preProcessing.train_test_split('/home/bbejczy/repos/GALIROOT/data/l515_lab_1410 (copy)/img/','/home/bbejczy/repos/GALIROOT/data/l515_lab_1410 (copy)/ann/', train_size=0.8)



if __name__ == "__main__":
    main()
