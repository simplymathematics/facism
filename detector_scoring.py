import pandas as pd 
import numpy
import matplotlib.pyplot as plt 
import glob
from pyspark.sql import SparkSession
# create sparksession
spark = SparkSession \
    .builder \
    .appName("Pysparkexample") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
import cv2
from PIL import Image
import os
from time import time
import numpy as np
import copy
from matplotlib import pyplot as plt


def read_image(filename):
    img_BGR = cv2.imread(filename)
    img_BGR = img_BGR.astype(np.uint8)
    return img_BGR


def grayscale(img_BGR):
    img_grayscale = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)
    return img_grayscale


def process(filename, faceCascade, scale_factor=1.1, min_Neighbors=5, min_Size=(30,30)):
    img_BGR = read_image(filename)
    img_grayscale = grayscale(img_BGR)
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(img_grayscale, scaleFactor=scale_factor, minNeighbors=min_Neighbors, minSize=min_Size)
    return faces

def find_these_files(directory):
    roots = set()
    dirs = set()
    files = set()
    for(root_folder, directory_list, file_list) in os.walk(directory):
        roots.add(root_folder)
        for directory in directory_list:
            dirs.add(directory)
        for file in file_list:
            abspath = os.path.join(root_folder, file)
            files.add(abspath)
    return list(files)


def test_haar_cascades(df, cascades):
    results = pd.DataFrame()
    i = 0
    j = 0
    t0 = time()

    results = []

    for cascade in cascades:
        i = i + 1
        j = 0
        print("-----------------------------------------")
        print(i, " of ", len(cascades), " in outer loop.")
        print("-----------------------------------------")
        cascade_classifier = cv2.CascadeClassifier(cascade)
        for path in df['Path']:
            j = j + 1
            if j == 1 or j % 10 == 0:
                print("\t", j, " of ", len(df['Path']), " in inner loop")
                pass
            try:
                result = process(path, cascade_classifier)
                result = len(result)
                tmp = [result, cascade, path]
                results.append(tmp)
            except:
                pass
    
    results = np.array(results)
    results = pd.DataFrame(results)
    results = results.rename(columns = {0: "Faces", 
                              1: "Cascade", 
                              2: "Path"})
    print("\t", j, " of ", len(df['Path']), " in inner loop")
    return(results)



def find_faces(directory, cascade):
    paths = find_these_files(directory)
    files = find_these_files("lfw-subset/raw/")
    df = create_data_frame_lfw(files)
#     df = pd.DataFrame({ "Path" : paths})
    results = test_haar_cascades(df, [cascade])
    return results

def create_data_frame_lfw(files, delims = "/."):
    dataframe = []
    for file in files:
        fullpath = copy.deepcopy(file)
        file = file.replace("../", "") #drop relative path
        for delim in delims:
            file = file.replace(delim,", ") # Change all delims to commas
        file = file.replace(" , " ,"") # Drop extra commas
        file = file.replace('\'', "") # Drop back slashes
        file = file.strip()
        file = file.split(",") # Creates a list from the string
        file.append(str(fullpath))
        dataframe.append(file)
        
    dataframe = pd.DataFrame(dataframe, columns = [ "Root", "Folder", "Identity", "File", "Type", "Path"])
#     dataframe = dataframe.drop("Blank", axis = 1)
    return(dataframe)

def segment_otsu(image_grayscale, img_BGR):
    threshold_value, threshold_image = cv2.threshold(image_grayscale, 0, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    threshold_image_binary = 1- threshold_image/255
    threshold_image_binary = np.repeat(threshold_image_binary[:, :, np.newaxis], 3, axis=2)
    img_face_only = np.multiply(threshold_image_binary, img_BGR)
    img_face_only = img_face_only.astype(np.uint8)
    return img_face_only

def skin_tone_estimator(img_face_only, verbose = False):
    img_HSV = cv2.cvtColor(img_face_only, cv2.COLOR_BGR2HSV)
    img_YCrCb = cv2.cvtColor(img_face_only, cv2.COLOR_BGR2YCrCb)

    # aggregate skin pixels
    blue = []
    green = []
    red = []

    height, width, channels = img_face_only.shape

    for i in range (height):
        for j in range (width):
            if((img_HSV.item(i, j, 0) <= 170) and (140 <= img_YCrCb.item(i, j, 1) <= 170) and (90 <= img_YCrCb.item(i, j, 2) <= 120)):
                blue.append(img_face_only[i, j].item(0))
                green.append(img_face_only[i, j].item(1))
                red.append(img_face_only[i, j].item(2))
            else:
                img_face_only[i, j] = [0, 0, 0]

    # determine mean skin tone estimate
    skin_tone_estimate_BGR = [np.mean(blue), np.mean(green), np.mean(red)]
    return skin_tone_estimate_BGR

def skin_tone_process(filename, verbose = False):
    img_BGR = read_image(filename)
    try:
        img_grayscale = grayscale(img_BGR)
    except:
        print("grayscale didn't work")
    try:
        img_face_only = segment_otsu(img_grayscale, img_BGR)
    except:
        print("segment broken")
    try:
        skin_tone_estimate_BGR = skin_tone_estimator(img_face_only, verbose)
    except:
        print("skin_tone_broken @ ", filename)
    relative_luminance = .2126 * skin_tone_estimate_BGR[2] + .7152 * skin_tone_estimate_BGR[1] + .0722 * skin_tone_estimate_BGR[0]
    return relative_luminance
def luminance_iterator(paths, verbose = False):
    lums = []
    i = 0
    for path in paths:
        try:
            lums.append(skin_tone_process(path))
        except:
            print("Numerical Error. Skipping this image.")
            lums.append("NaN")
        i = i + 1
        if verbose == True and i % 1000 == 0:
            print(round(i/len(paths)*100,1) , "% Complete.")
    return lums

def calculate_luminance(df): 
    paths = df['Path']
    lums= luminance_iterator(paths, True)
    df['Luminance'] = lums
    return df
def bin_luminance(df, no_bins = 7, bin_labels = ["Least Reflective", "Much Less Reflective", 
                                       "Less Reflective", "Average Reflectivity", 
                                       "Somewhat More Reflective", "More Reflective", 
                                       "Most Reflective"]):
    labelled, bins = pd.cut(df.Luminance, no_bins, 
                              labels = bin_labels,
                              retbins = True, 
                              precision = 0)
    dummy = pd.get_dummies(labelled)
    df = pd.concat([df, dummy], axis = 1)
    return df, bin_labels

def clean_and_save(df, file_location = 'dataset.csv', no_bins = 3, bin_labels = ["Less Reflective", "Average Reflectivity", "More Reflective"]):    
    print("Saving ", file_location,)
    df.dropna(axis = 'rows', how =  'any',  inplace = True)
    df = df.loc[df.Luminance.notnull()]
    df = df[df.Luminance != 'NA']
    df = df.reset_index()
    df = df.drop('index', axis = 'columns')
    try:
        df = df.drop(['level_0', 'index'], 1)
    except:
        pass
    try:
        df = bin_pose(df, bin_labels)
    except:
        pass
    df, clean_and_save = bin_luminance(df, no_bins, bin_labels)
    df.to_csv(file_location)
    return df, bin_labels


def luminance(files):
    files = find_these_files("lfw-subset/raw/")
    df = create_data_frame_lfw(files)
    df = calculate_luminance(df)
    df, bin_labels = clean_and_save(df, 'results.csv')
    return df, bin_labels

def main(directory, cascade, filepath):
        files = find_these_files(directory)
        df, bin_labels = luminance(files)
        results = find_faces(directory, cascade)
        merged = df.merge(results, how = 'left', on = 'Path')
        merged.Faces = merged.Faces.astype(str).str.replace("2", "0")
        merged.Faces = merged.Faces.astype(int)
        merged.to_csv()
        accuracy = sum(merged.Faces)/len(merged.Faces)* 100
        accuracy = round(accuracy, 2)
        print("Accuracy of Detector is", accuracy, "%")
        print("Final Results Saved at", filepath)
        return merged



if __name__ == '__main__':
    


    directory = "./lfw-subset/raw/"
    cascade = 'haarcascade_frontalface_default.xml'
    filepath = "results.csv"
    df = main(directory = directory, cascade = cascade, filepath = filepath)
    