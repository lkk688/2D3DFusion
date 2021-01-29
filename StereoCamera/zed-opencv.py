import sys
import numpy as np
import pyzed.sl as sl
import cv2
import math
from os import makedirs
from os.path import exists, join
import shutil
import json


help_string = "[s] Save side by side image [d] Save Depth, [n] Change Depth format, [p] Save Point Cloud, [m] Change Point Cloud format, [q] Quit"
prefix_point_cloud = "Cloud"
prefix_depth = "Depth"
path = "./Dataset0123a"

count_save = 0
mode_point_cloud = 0
mode_depth = 0
point_cloud_format_ext = ".ply"
depth_format_ext = ".png"

def make_clean_folder(path_folder):
    if not exists(path_folder):
        makedirs(path_folder)
    else:
        user_input = input("%s not empty. Overwrite? (y/n) : " % path_folder)
        if user_input.lower() == 'y':
            shutil.rmtree(path_folder)
            makedirs(path_folder)
        else:
            exit()

def point_cloud_format_name(): 
    global mode_point_cloud
    if mode_point_cloud > 3:
        mode_point_cloud = 0
    switcher = {
        0: ".xyz",
        1: ".pcd",
        2: ".ply",
        3: ".vtk",
    }
    return switcher.get(mode_point_cloud, "nothing") 
  
def depth_format_name(): 
    global mode_depth
    if mode_depth > 2:
        mode_depth = 0
    switcher = {
        0: ".png",
        1: ".pfm",
        2: ".pgm",
    }
    return switcher.get(mode_depth, "nothing") 

def save_point_cloud(zed, filename) :
    print("Saving Point Cloud...")
    tmp = sl.Mat()
    zed.retrieve_measure(tmp, sl.MEASURE.DEPTH)
    saved = (tmp.write(filename + depth_format_ext) == sl.ERROR_CODE.SUCCESS)
    if saved :
        print("Done")
    else :
        print("Failed... Please check that you have permissions to write on disk")

def save_depth(zed, filename) :
    print("Saving Depth Map...")
    tmp = sl.Mat()
    zed.retrieve_measure(tmp, sl.MEASURE.XYZRGBA)
    saved = (tmp.write(filename + point_cloud_format_ext) == sl.ERROR_CODE.SUCCESS)
    if saved :
        print("Done")
    else :
        print("Failed... Please check that you have permissions to write on disk")

def save_sbs_image(zed, filename) :

    image_sl_left = sl.Mat()
    zed.retrieve_image(image_sl_left, sl.VIEW.LEFT)
    image_cv_left = image_sl_left.get_data()

    image_sl_right = sl.Mat()
    zed.retrieve_image(image_sl_right, sl.VIEW.RIGHT)
    image_cv_right = image_sl_right.get_data()

    sbs_image = np.concatenate((image_cv_left, image_cv_right), axis=1)

    cv2.imwrite(filename, sbs_image)

def save_all(zed, count_save):

    #foldername=path + str(count_save) + "_"

    image_sl_left = sl.Mat()
    image_sl_right = sl.Mat()
    depth = sl.Mat()
    #pointcloud = sl.Mat()

    zed.retrieve_image(image_sl_left, sl.VIEW.LEFT)
    zed.retrieve_image(image_sl_right, sl.VIEW.RIGHT)
    #zed.retrieve_measure(pointcloud, sl.MEASURE.XYZRGBA) #Point Cloud is aligned on the left image
    zed.retrieve_measure(depth, sl.MEASURE.DEPTH) #Depth is aligned on the left image.

    print("Saving Depth Map...")
    #depthfilename=path + prefix_depth + 'Depth/' +str(count_save)+ depth_format_ext #point_cloud_format_ext
    path_depth = join(path, "depth")
    if not exists(path_depth):
        makedirs(path_depth)
    depthfilename="%s/%06d%s" % (path_depth, count_save, depth_format_ext)
    print(depthfilename)
    saved = (depth.write(depthfilename) == sl.ERROR_CODE.SUCCESS)
    if saved :
        print("Depth Written Done")
    else :
        print("Failed... Please check that you have permissions to write on disk")
    
    # print("Saving Point Cloud...")
    # pointcloudfilename=foldername + prefix_point_cloud + point_cloud_format_ext #depth_format_ext
    # saved = (pointcloud.write(pointcloudfilename) == sl.ERROR_CODE.SUCCESS)
    # if saved :
    #     print("Pointcloud Written Done")
    # else :
    #     print("Failed... Please check that you have permissions to write on disk")

    print("Saving left and right image...")
    image_cv_left = image_sl_left.get_data()
    image_cv_right = image_sl_right.get_data()
    image2_path = join(path, "image2")
    if not exists(image2_path):
        makedirs(image2_path)
    leftimagefilename="%s/%06d%s" % (image2_path, count_save, ".jpg")
    image3_path = join(path, "image3")
    if not exists(image3_path):
        makedirs(image3_path)
    rightimagefilename="%s/%06d%s" % (image3_path, count_save, ".jpg")
    print(leftimagefilename)
    leftwriteStatus = cv2.imwrite(leftimagefilename, image_cv_left) 
    rightwriteStatus = cv2.imwrite(rightimagefilename, image_cv_right) 
    if leftwriteStatus is True and rightwriteStatus is True:
        print("Image written")
    else:
        print("problems in image written") # or raise exception, handle problem, etc.

def save_all_backup(zed, count_save):

    foldername=path + str(count_save) + "_"

    image_sl_left = sl.Mat()
    image_sl_right = sl.Mat()
    depth = sl.Mat()
    pointcloud = sl.Mat()

    zed.retrieve_image(image_sl_left, sl.VIEW.LEFT)
    zed.retrieve_image(image_sl_right, sl.VIEW.RIGHT)
    zed.retrieve_measure(pointcloud, sl.MEASURE.XYZRGBA) #Point Cloud is aligned on the left image
    zed.retrieve_measure(depth, sl.MEASURE.DEPTH) #Depth is aligned on the left image.

    print("Saving Depth Map...")
    depthfilename=foldername + prefix_depth + depth_format_ext #point_cloud_format_ext
    saved = (depth.write(depthfilename) == sl.ERROR_CODE.SUCCESS)
    if saved :
        print("Depth Written Done")
    else :
        print("Failed... Please check that you have permissions to write on disk")
    
    print("Saving Point Cloud...")
    pointcloudfilename=foldername + prefix_point_cloud + point_cloud_format_ext #depth_format_ext
    saved = (pointcloud.write(pointcloudfilename) == sl.ERROR_CODE.SUCCESS)
    if saved :
        print("Pointcloud Written Done")
    else :
        print("Failed... Please check that you have permissions to write on disk")

    print("Saving left and right image...")
    image_cv_left = image_sl_left.get_data()
    image_cv_right = image_sl_right.get_data()
    leftwriteStatus = cv2.imwrite(foldername + "ZED_image" + "_left.jpg", image_cv_left) 
    rightwriteStatus = cv2.imwrite(foldername + "ZED_image" + "_right.jpg", image_cv_right) 
    if leftwriteStatus is True and rightwriteStatus is True:
        print("Image written")
    else:
        print("problems in image written") # or raise exception, handle problem, etc.
    
    

def process_key_event(zed, key) :
    global mode_depth
    global mode_point_cloud
    global count_save
    global depth_format_ext
    global point_cloud_format_ext

    if key == 100 or key == 68:
        save_depth(zed, path + prefix_depth + str(count_save))
        count_save += 1
    elif key == 110 or key == 78:
        mode_depth += 1
        depth_format_ext = depth_format_name()
        print("Depth format: ", depth_format_ext)
    elif key == 112 or key == 80:
        save_point_cloud(zed, path + prefix_point_cloud + str(count_save))
        count_save += 1
    elif key == 109 or key == 77:
        mode_point_cloud += 1
        point_cloud_format_ext = point_cloud_format_name()
        print("Point Cloud format: ", point_cloud_format_ext)
    elif key == 104 or key == 72:
        print(help_string)
    elif key == 115:
        #save_sbs_image(zed, "ZED_image" + str(count_save) + ".png")
        save_all(zed, count_save)
        count_save += 1
    else:
        a = 0

def print_help() :
    print(" Press 's' to save Side by side images")
    print(" Press 'p' to save Point Cloud")
    print(" Press 'd' to save Depth image")
    print(" Press 'm' to switch Point Cloud format")
    print(" Press 'n' to switch Depth format")


def main() :

    # Create a ZED camera object
    zed = sl.Camera()

    # Set configuration parameters
    input_type = sl.InputType()
    if len(sys.argv) >= 2 :
        input_type.set_from_svo_file(sys.argv[1])
    init = sl.InitParameters(input_t=input_type)
    init.camera_resolution = sl.RESOLUTION.HD1080 #sl.RESOLUTION.HD720
    init.depth_mode = sl.DEPTH_MODE.ULTRA #sl.DEPTH_MODE.PERFORMANCE, ref:https://www.stereolabs.com/docs/depth-sensing/
    init.coordinate_units = sl.UNIT.MILLIMETER
    #init.coordinate_units = sl.UNIT.METER
    init.depth_minimum_distance = 0.15 # Set the minimum depth perception distance to 15cm


    # Open the camera
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        zed.close()
        exit(1)
    
    # Set runtime parameters after opening the camera
    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD #sl.SENSING_MODE.FILL

    # Prepare new image size to retrieve half-resolution images
    image_size = zed.get_camera_information().camera_resolution
    #image_size.width = image_size.width /2
    #image_size.height = image_size.height /2
    print("Image size half width: ", image_size.width)
    print("Image size half height: ", image_size.height)
    make_clean_folder(path)
    
    #Read camera parameters
    calibration_params = zed.get_camera_information().calibration_parameters
    # Focal length of the left eye in pixels
    focal_left_x = calibration_params.left_cam.fx
    # First radial distortion coefficient
    k1 = calibration_params.left_cam.disto[0]
    # Translation between left and right eye on z-axis
    #tz = calibration_params.T.z
    # Horizontal field of view of the left eye in degrees
    h_fov = calibration_params.left_cam.h_fov
    print(calibration_params)
    print("Focal length of the left eye in pixels, fx:", focal_left_x)
    print("Focal length of the left eye in pixels, fy:", calibration_params.left_cam.fy)
    print("Principal points: cx:", calibration_params.left_cam.cx) #Principal points: cx, cy.
    print("Principal points: cy:", calibration_params.left_cam.cy)
    print("First radial distortion coefficient, k1:", k1)
    print("First radial distortion coefficient, k2:", calibration_params.left_cam.disto[1])
    #print("Translation between left and right eye on z-axis:", tz)
    print("Horizontal field of view of the left eye in degrees:", h_fov)
    jsonfilename=join(path, "zed_intrinsics.json")
    with open(jsonfilename, 'w') as outfile:
        obj = json.dump(
            {
                'width':
                    image_size.width,
                'height':
                    image_size.height,
                'intrinsic_matrix': [
                    calibration_params.left_cam.fx, 0, 0, 0, calibration_params.left_cam.fy, 0, calibration_params.left_cam.cx,
                    calibration_params.left_cam.cy, 1
                ]
            },
            outfile,
            indent=4)

    # Display help in console
    print_help()

    # Declare your sl.Mat matrices
    image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    depth_image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    point_cloud = sl.Mat()

    key = ' '
    while key != 113 :
        err = zed.grab(runtime) #grab() to grab a new image
        if err == sl.ERROR_CODE.SUCCESS :
            # Retrieve the left image, depth image in the half-resolution
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            #zed.retrieve_image(image_zed_right, sl.VIEW.LEFT, sl.MEM.CPU, image_size)

            zed.retrieve_image(depth_image_zed, sl.VIEW.DEPTH, sl.MEM.CPU, image_size)
            # Retrieve the RGBA point cloud in half resolution
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, image_size)

            # The depth matrix stores 32-bit floating-point values which represent depth (Z) for each (X,Y) pixel
            depth_value = depth_image_zed.get_value(image_size.width/2,image_size.height/2)
            #print("Depth value at center point:", depth_value) #array([ 70,  70,  70, 255])
            # Get the 3D point cloud values for pixel (i,j)
            point3D = point_cloud.get_value(image_size.width/2,image_size.height/2)
            #print(point3D) #SUCCESS, array, NaN value
            #The point cloud stores its data on 4 channels using 32-bit float for each channel.
            x = point3D[1][0]
            y = point3D[1][1]
            z = point3D[1][2]
            color = point3D[1][3] #The last float is used to store color information, where R, G, B, and alpha channels (4 x 8-bit) are concatenated into a single 32-bit float
            distance = math.sqrt(x*x + y*y + z*z)
            #print("Distance to the center point,", distance)

            # To recover data from sl.Mat to use it with opencv, use the get_data() method
            # It returns a numpy array that can be used as a matrix with opencv
            image_ocv = image_zed.get_data()
            #image_ocv_right = image_zed_right.get_data()
            depth_image_ocv = depth_image_zed.get_data()
            #print("Depth numpy shape:", depth_image_ocv.shape)#540, 960, 4, 1080, 1920
            # Print the depth value at the center of the image
            #print("len:", len(depth_image_ocv)) #1080
            #print("len0:", len(depth_image_ocv[0])) #1920
            #print(depth_image_ocv[int(len(depth_image_ocv)/2)][int(len(depth_image_ocv[0])/2)]) #[218 218 218 255]

            cv2.imshow("Image", image_ocv)
            cv2.imshow("Depth", depth_image_ocv)

            key = cv2.waitKey(10)

            process_key_event(zed, key)

    cv2.destroyAllWindows()
    zed.close()

    print("\nFINISH")

if __name__ == "__main__":
    main()
