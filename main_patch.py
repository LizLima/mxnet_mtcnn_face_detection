# coding: utf-8
from turtle import width
from cv2 import circle
import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import os
import time
import math

detector = MtcnnDetector(model_folder='model', ctx=mx.cpu(0), num_worker = 4 , accurate_landmark = False)

# Crop image from dataset => image 224 x 224
def croped_image() :
    print("start crop image")
    path_result = "" # "/media/liz/TOSHIBA EXT1/Data/helen_eye_dataset_crop/train_4"
    path_dataset = "" # "/media/liz/TOSHIBA EXT1/Data/helen_eye_dataset/train_4/train_4"


    items = os.listdir(path_dataset) 
    count = 1
    for path in items:
        img = cv2.imread(os.path.join(path_dataset, path))
        print(os.path.join(path_dataset, path))
        # run detector
        results = detector.detect_face(img)

        if results is not None:

                total_boxes = results[0]
                points = results[1]
                
                # extract aligned face chips
                # chips = detector.extract_image_chips(img, points, 144, 0.37)
                # for i, chip in enumerate(chips):
                #     cv2.imshow('chip_'+str(i), chip)
                #     cv2.imwrite('chip_'+str(i)+'.png', chip)

                draw = img.copy()
                cropped = img.copy()
                
                name_folder = os.path.join(path_result, str(count).zfill(3) )
                
                if not os.path.exists(name_folder):
                    os.makedirs(name_folder)

                count_item = 1
                for b in total_boxes:
                    # get point middle
                    x = int(int(b[0] + int(b[2])) / 2)
                    y = int(int(b[1] + int(b[3])) / 2)
                    cv2.circle(draw, (x, y), 1, (0, 255, 0), 2)

                    # get distance
                    x_dist = abs(int(b[0] - int(b[2])))
                    y_dist = abs(int(b[1] - int(b[3])))
                   
                    mid_length = int(max(x_dist, y_dist) / 2)
                    # cv2.line(draw, (int(b[0]), int(b[1])), (int(b[0]) + x_dist, int(b[1])), (255, 0, 0), 3)
                    # cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))
                    # Get new boxes
                    x_init = x - mid_length
                    y_init = y - mid_length

                    x_end = x + mid_length
                    y_end = y + mid_length
                    
                    # cv2.line(draw, (x_init, y_init), (x, y), (124, 59, 232), 2)
                    # cv2.line(draw, (x_end, y_end ), (x, y), (232, 213, 59), 2)
                    cv2.rectangle(draw, (x_init,y_init), (x_end, y_end), (255, 255, 255), 3)
                    
                   
                    try:
                        crop_img = cropped[ y_init:y_end, x_init:x_end]
                        cv2.imwrite(os.path.join(name_folder, str(count_item).zfill(2) + ".png" ), crop_img)
                        cv2.imwrite(os.path.join(name_folder, "original.png" ), img)
                        # cv2.imshow("cropped", crop_img)
                        # cv2.waitKey(0)
                        count_item += 1
                    except:
                        print(path)
                    

                    
                
                
                count += 1

                # for p in points:
                #     for i in range(5):
                #         cv2.circle(draw, (int(p[i]), int(p[i + 5])), 1, (0, 0, 255), 2)
                
                
                # cv2.imshow("detection result", draw)
                # cv2.waitKey(0)


#get patch from copped image
def get_shape_quare(size ,point_x, point_y, width, height):
    mid_width = int(width / 2)
    mid_height = int(height / 2)  

    x_init = max(point_x - mid_width , 0 )
    y_init = max(point_y - mid_height , 0 )

    x_end = x_init + width
    y_end = y_init + height

    return x_init, y_init, x_end, y_end


def getpatch() :
    print("get patch")
    path_dataset = "/media/liz/TOSHIBA EXT1/Data/helen_eye_dataset_crop/train_4"

    error_path = []
    directory = os.listdir(path_dataset) 
    for folder in directory:
        img_path = os.path.join(path_dataset, folder, "01.png")
       
        if os.path.exists(img_path):
            img_real = cv2.imread(img_path)
            img = cv2.resize(img_real, (224, 224), interpolation= cv2.INTER_LINEAR)

            draw = img.copy()
            cropped = img.copy()

            results = detector.detect_face(img)
           
            if (results is not  None):
                points = results[1]
          
                for p in points:
                    if len(p) == 10:
                        for i in range(5):
                            cv2.circle(draw, (int(p[i]), int(p[i + 5])), 1, (0, 0, 255), 2)
                        
                        # Left eye
                        x_init, y_init, x_end, y_end  = get_shape_quare(224, int(p[0]), int( p[5]), 40, 40)
                        # cv2.rectangle(draw, (x_init,y_init), (x_end, y_end), (255, 255, 255), 3)

                        # Right eye
                        x_init_r, y_init_r, x_end_r, y_end_r  = get_shape_quare(224, int(p[1]), int( p[6]), 40, 40)
                        # cv2.rectangle(draw, (x_init_r,y_init_r), (x_end_r, y_end_r), (255, 255, 255), 3)

                        # Nose
                        x_init_nose, y_init_nose, x_end_nose, y_end_nose  = get_shape_quare(224, int(p[2]), int( p[7]), 32, 48)
                        # cv2.rectangle(draw, (x_init_nose,y_init_nose), (x_end_nose, y_end_nose), (255, 255, 255), 3)

                        # Mouth
                        center_x = int((p[3] + p[4]) / 2)
                        center_y = int((p[8] + p[9]) / 2)
                        x_init_mouth, y_init_mouth, x_end_mouth, y_end_mouth  = get_shape_quare(224, center_x, center_y, 54, 32)
                        # cv2.rectangle(draw, (x_init_mouth,y_init_mouth), (x_end_mouth, y_end_mouth), (255, 255, 255), 3)

                        try:
                            crop_img = cropped[ y_init:y_end, x_init:x_end]
                            cv2.imwrite(os.path.join(path_dataset, folder,  "eye-left-01.png" ), crop_img)

                            crop_img = cropped[ y_init_r:y_end_r, x_init_r:x_end_r]
                            cv2.imwrite(os.path.join(path_dataset, folder,  "eye-right-01.png" ), crop_img)

                            crop_img = cropped[ y_init_nose:y_end_nose, x_init_nose:x_end_nose]
                            cv2.imwrite(os.path.join(path_dataset, folder,  "nose_01.png" ), crop_img)

                            crop_img = cropped[ y_init_mouth:y_end_mouth, x_init_mouth:x_end_mouth]
                            cv2.imwrite(os.path.join(path_dataset, folder,  "mouth_01.png" ), crop_img)
                            
                        except:
                            error_path.append(img_path)
                        
                        # cv2.imshow("detection result", draw)
                        # cv2.waitKey(0)
                    else:
                        error_path.append(img_path)

            else:
                error_path.append(img_path)
        else:
            error_path.append(img_path)
    
    print(error_path)

def main() :
    # crop image widthout rezise to 224 x 224
    getpatch()
    
if __name__ == "__main__":
    main()