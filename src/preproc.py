import cv2
import os
import numpy as np
import csv
import albumentations as albu

class preProc():

    def __init__(self, directory):
        self.img_list = []
        self.name_list = []
        self.directory = os.path.join(directory,"l515_imgs")
        self.depth_dir = os.path.join(directory,"depth_imgs")
        self.rgb_dir = os.path.join(directory,"rgb_imgs")
        self.img_count = np.zeros([1,2])


    def list_files(self, dir, fileformat = ".png"):
        for root, dirs, files in os.walk(dir):
            for name in files:
                if name.endswith(fileformat):
                    self.img_list.append(os.path.join(root, name))
                    self.name_list.append(name)
                    self.crop_roi(root,name) # crop the images to ROI


    def crop_roi(self,root,name):
        raw = cv2.imread(os.path.join(root, name))
        roi = raw[365:1000 , 345:1120] 
        self.sort_imgs(name,roi) # sort images to /depth nd /rgb


    def sort_imgs(self,name,img):
            if "depth" in name:
                new_filename = os.path.join(self.depth_dir,name)
                cv2.imwrite(new_filename,img)
            elif "rgb" in name:
                new_filename = os.path.join(self.rgb_dir,name)
                cv2.imwrite(new_filename,img)
            else:
                raise Exception("Image could not be cathegorized. Check file naming.")
    
    def clean_json(self,dir,fileformat = ".json"):
        for root, dirs, files in os.walk(dir):
            for name in files:
                if name.endswith(fileformat):

                    new_name = name.replace('.png','')
                    os.rename(dir+name, dir+new_name)
            print("{} files renamed".format(len(files)))



            

def main():
    directory = "/home/bbejczy/repos/GALIROOT/data/"
    annotations = '/home/bbejczy/repos/GALIROOT/data/20200809_skaevinge_labelled/ann_p/'
    pp = preProc(directory)
    # pp.list_files(pp.directory)  
    pp.clean_json(annotations)

if __name__ == "__main__":
    main()
