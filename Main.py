#deleted all packages
#installed keras 2.15.0 cause it is allowing to call keras.models instead of keras.api.models or keras.src.models
#imported load_model method from keras.models directly as the simplecnn model is also called from keras.models
#numpy 1.26.4 is installed automatically with keras (numpy 1.26.x is only compatible with keras)
#installed tensorflow 2.15..0 as it is compatible with using keras 2.15.0 and numpy 1.26.4
#installed opencv-python for cv2
#installed matplotlib for pyplot
#installed scikit-learn instead of sklearn as sklearn is not working and we need train_test_split method for splitting from sklearn.preprocessing
#installed tqdm as we need it to visualize the progress bar
#Use pip only for installations as it will check for version compatibilities and installs compatible versions automatically if using any other methods for installing packages then there are issues with compatibilies
#compatibility between tensorflow and keras and numpy is very important for model training

import Models
import ViolenceDetection_Data as data
import Models as models
import sys

if sys.version == "3.11.9 (tags/v3.11.9:de54cf5, Apr  2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]":
    print("Project Violence Detection setup is successfull"+" 🟢")
else:
    print("Project Violence Detection setup is Failed"+" 🔴")
    exit()

#main function
if __name__=='__main__':
    dataset_path = r'C:\Users\swarn\Documents\Projects\DeepLearning_DataSets\Real Life Violence Dataset'
    data.data_collection(dataset_path)
    data.eda(dataset_path)
    features_train,features_test,labels_train,labels_test,model_output_size,preprocess_data_status,image_height,image_width=data.datapreprocessing(dataset_path)
    model=models.simplecnnmodel(preprocess_data_status,image_height,image_width)
    Models.model_fit(model,features_train,labels_train,features_test,labels_test)
    Models.savemodel(model,features_train,features_test,labels_train,labels_test)
    Models.saved_model_evaluating(features_test,labels_test)
    Models.predict(video_file_path=r"C:\Users\swarn\Documents\Projects\DeepLearning_DataSets\RandomTestingData_ForModels\vid2.mp4",output_video_file_path=r'C:\Users\swarn\Documents\Projects\PycharmProjects\ViolenceDetectionSystem\Output\result.mp4',moving_avg_window_size=1)
    print(f'Project Executed Successfully🟢')