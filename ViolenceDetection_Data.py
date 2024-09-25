#importing libraries
import numpy as np
import tensorflow as tf
import cv2
import os
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
from sklearn.preprocessing import LabelEncoder

#data collection
def data_collection(datasetfolder):
    path = datasetfolder
    print(f'Dataset Folder Identified Successfully at',path,'✅')

#Exploratory Data Analysis
def eda(datasetfolder):
    print(f'dataset name:-',os.path.basename(datasetfolder))
    labels = os.listdir(datasetfolder)
    print(f'Total class labels:-', len(os.listdir(datasetfolder)))
    print(f'Class Names:-',labels)
    videos_count = {}
    for label in labels:
        label_path = os.path.join(datasetfolder,label)
        videos_count[label] = len(os.listdir(label_path))
    print(f'total number of videos in each class:-',videos_count)
    plt.figure(figsize=(10,5))
    plt.bar(videos_count.keys(),videos_count.values())
    plt.xticks(rotation=False)
    plt.xlabel('Labels')
    plt.ylabel('Number of videos')
    plt.show(block=True)
    plt.figure(figsize=(20,10))
    random_range = random.sample(range(len(labels)), 2)
    for counter, random_index in enumerate(random_range, 1):
        selected_class_name = labels[random_index]
        video_files_names_list = os.listdir(datasetfolder + f'/{selected_class_name}')
        selected_video_file_name = random.choice(video_files_names_list)
        video_reader = cv2.VideoCapture(datasetfolder + f'/{selected_class_name}/{selected_video_file_name}')
        _, bgr_frame = video_reader.read()
        video_reader.release()
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        cv2.putText(rgb_frame, selected_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        plt.subplot(1, 2, counter)
        plt.imshow(rgb_frame)
        plt.axis('off')
    plt.show()

#Data Preprocessing
def datapreprocessing(datasetfolder):
    print(f'Data preprocessing in progress...')
    img_height = 64
    img_width = 64
    targets = os.listdir(datasetfolder)
    mdl_output_size = len(targets)

    seed_constant = 23
    preprocessed_data_file_path = r"preprocessed_data.pkl"
    load_preprocessed_data_from_file = input('Do you want to load the preprocessed data from file (yes/no)?\n')
    if load_preprocessed_data_from_file == 'no':
        print(f'Data Extraction Started')
        features,labels = create_dataset(datasetfolder, targets, img_height, img_width)
        print(f'Feature Extraction Completed✅')
        print(f'DataFrame Created Successfully✅')
        print(f'preprocessed features data shapes and types are:-\n', 'features:-',np.shape(features),type(features))
        #Label Encoded Used Cause One Hot Encoder won't accept integer values
        labelencoder = LabelEncoder()
        labels = labelencoder.fit_transform(labels)
        print('labels preprocessing is completed✅')
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, shuffle=True, random_state=seed_constant)
        print(f'Data Splitting for training and testing Completed✅')
        with open(preprocessed_data_file_path,'wb') as file:
            pickle.dump((x_train, x_test,y_train,y_test, mdl_output_size), file)
        print('PreProcessed data saved to file Successfully✅')
        print(f'preprocessed data shapes are:-\n'+'x_train shape-',np.shape(x_train),' y_train shape-',np.shape(y_train),'\nx_test shape-',np.shape(x_test),' y_test shape-',np.shape(y_test))
        preprocessed_data_status = True
        y_train = y_train.tolist()
        y_test = y_test.tolist()
        x_train = tf.convert_to_tensor(x_train)
        x_test = tf.convert_to_tensor(x_test)
        y_test = tf.convert_to_tensor(y_test)
        y_train = tf.convert_to_tensor(y_train)
        return x_train, x_test, y_train, y_test, mdl_output_size, preprocessed_data_status,img_height,img_width
    elif load_preprocessed_data_from_file == 'yes':
        with open(preprocessed_data_file_path,'rb') as file:
            x_train,x_test,y_train,y_test,mdl_output_size = pickle.load(file)
            print("Data Loaded Successfully from file",os.path.basename(preprocessed_data_file_path))
        print(f'preprocessed data shapes and types are:-\n', 'x_train shape-', np.shape(x_train),type(x_train), ' y_train shape-',np.shape(y_train),type(y_train), '\nx_test shape-', np.shape(x_test),type(x_test), ' y_test shape-', np.shape(y_test),type(y_test))
        preprocessed_data_status = True
        y_train = y_train.tolist()
        y_test = y_test.tolist()
        x_train = tf.convert_to_tensor(x_train)
        x_test = tf.convert_to_tensor(x_test)
        y_test = tf.convert_to_tensor(y_test)
        y_train = tf.convert_to_tensor(y_train)
        return x_train,x_test,y_train,y_test,mdl_output_size,preprocessed_data_status,img_height,img_width
    else:
        print("Option Chosen Not found")
        preprocessed_data_status = False
        return preprocessed_data_status

#function for Frame Extraction
def frame_extraction(video_path, img_height, img_width):
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    while True:
        success,frame = video_reader.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (img_height, img_width))
        normalized_frame = resized_frame/255
        frames_list.append(normalized_frame)
    video_reader.release()
    return frames_list
#function to create dataset from the extracted frames
def create_dataset(datasetfolder,targets,img_height,img_width):
    temp_features = []
    features = []
    labels = []
    #total frames for 1 class is 1150*5*30=1,72,500
    max_images_per_label = 125000
    for class_name in targets:
        print(f'extracting the data of class:{class_name}')
        files_list = os.listdir(os.path.join(datasetfolder,class_name))
        for file_name in tqdm(files_list):
            video_file_path  = os.path.join(datasetfolder,class_name,file_name)
            frames = frame_extraction(video_file_path,img_height,img_width)
            temp_features.extend(frames)
        features.extend(random.sample(temp_features,max_images_per_label))
        labels.extend([class_name]*max_images_per_label)
        temp_features.clear()
    #features = tf.convert_to_tensor(features)
    #labels = tf.convert_to_tensor(labels)
    #print(features.numpy(),labels.numpy())
    print("after appending features and labels to list the shapes of features and labels are:-",np.shape(features),type(features),np.shape(labels),type(labels))
    return features,labels
