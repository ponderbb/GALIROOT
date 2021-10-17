import albumentations as A

class Aug()
def __init__(self):
    self.settings = 

def vis_keypoints(image, keypoints, color=(0, 255, 0), diameter=10):

    image = image.copy()

    for (x, y) in keypoints:
        cv2.circle(image, (int(x), int(y)), diameter, (0, 255, 0), -1)

    cv2.imshow('image',image)
    cv2.waitKey(3000)


def augmentations(self, img_dir, ann_dir): # TODO: remove this from here, just for show


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


def main():

    augmentations()



if __name__ == "__main__":
    main()