import utils
import random
import json
import numpy as np
from pathlib import Path
import os, shutil


class preProcessing():

    @staticmethod
    def sort_files(directory, fileformat, d_folder, rgb_folder): # TODO: general sorting solution based on dictionary{targetfolder, file}

        '''
        To sort images batches to depth and rgb images by naming.
        '''

        os.chdir(directory)
        os.makedirs(d_folder, exist_ok=True)
        d_path = os.path.join(directory, d_folder)
        os.makedirs(rgb_folder, exist_ok=True)
        rgb_path = os.path.join(directory, rgb_folder)

        path_list, name_list = preProcessing(directory, fileformat)

        for path in path_list:

            name = Path(path).stem

            if "depth" in name:
                try: 
                    shutil.copyfile(path, os.path.join(d_path,name))
                except shutil.Error:
                    print(f'{name} already exists in dire')
                    pass
            elif "rgb" in name:
                try: 
                    shutil.copyfile(path, os.path.join(rgb_path,name))
                except shutil.Error:
                    print(f'{name} already exists in dire')
                    pass
            else:
                raise Exception(f"Image {name} not be cathegorized. Check file naming.")

    @staticmethod
    def clean_json(directory, fileformat, removable_text = ".png"):

        '''
        - Remove .png from json names ( Supervisely specific issue )
        - Remove json files with no annotation information ( Supervisely specific format )
        '''

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
    def train_test_split(img_dir, ann_dir, train_size = 0.85): #FIXME: could be more efficient

        '''
        Split images and annotations into a training and test set. The training set remains in the same folder and a separate test subfoler will be created for images and annotations.
        '''

        img_list, __ = sorted(utils.list_files(img_dir, '.png'))
        ann_list, __ = sorted(utils.list_files(ann_dir, '.json'))

        test_path_img = str(Path(img_list[0]).parents[0])+'/test/'
        test_path_ann = str(Path(ann_list[0]).parents[0])+'/test/'

        os.makedirs(test_path_img, exist_ok=True)
        os.makedirs(test_path_ann, exist_ok=True)

        num_test_sample = int(np.round(len(img_list)*(1-train_size)))
        test_set_idx = random.sample(range(len(img_list)), num_test_sample)

        for idx in test_set_idx:
            shutil.move(img_list[idx], test_path_img+str(Path(img_list[idx]).name))
            shutil.move(ann_list[idx], test_path_ann+str(Path(ann_list[idx]).name))



def main():

    # test functions

    settings = utils.open_config('/home/bbejczy/repos/GALIROOT/config/preprocessing.json')
    
    images, names = utils.list_files(settings['directory'], settings['image_format'])

    print(names[0])

    preProcessing.train_test_split(settings['directory'], settings['annotations'])



if __name__ == "__main__":
    main()
