from collections import deque
import cv2
from keras.models import load_model
from keras import layers,models
import keras
import numpy as np


#Simple CNN Model Building
def simplecnnmodel(data_status, img_height, img_width):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(img_height, img_width, 3)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.GlobalMaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    #model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    print("Model Built is Successfull✅")
    model.summary()
    if data_status:
        model.compile(optimizer='adam',loss=keras.losses.BinaryCrossentropy(),metrics=['accuracy'])
        print('Model Ready for training✅')
        return model
    else:
        print('Data Not Available to train the model❌')
        return

#method for fitting the model
def model_fit(model,x_train,y_train,x_test,y_test):
    status = input('Do you want to train the model? (yes/no)')
    if status == 'yes':
        print('Model training in progress...')
        early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min',restore_best_weights=True)
        model.fit(x_train, y_train, batch_size=4, shuffle=True, epochs=10, validation_data=(x_test, y_test),callbacks=[early_stopping_callback])
        model_evaluation = model.evaluate(x_test, y_test)
        print(f'Model Training is Done✅','\n',f'model evaluation:-{model_evaluation}')

    else:
        return

#method for saving the model
def savemodel(model,x_train,y_train,x_test,y_test):
    status = input('Do you want to save model to file? (yes/no)')
    if status == 'yes':
        model.save("simpleCnnModel.h5")
        print('Model Saved Successfully✅')
    elif status == 'no':
        return
    else:
        print('option choosen not found❌')
        return

#method for predicting
def predict(video_file_path,output_video_file_path,moving_avg_window_size):
    model_new=load_model('simpleCnnModel.h5')
    print('Model Loading Successfull✅')
    model_new.summary()
    predict_prob_dequee = deque(maxlen=moving_avg_window_size)
    video_reader = cv2.VideoCapture(video_file_path)
    video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(output_video_file_path,cv2.VideoWriter.fourcc('m','p','4','v'),30,(video_width,video_height))
    while True:
        status, frame = video_reader.read()
        if not status:
            print("Video Ended")
            break
        resized_frame = cv2.resize(frame, (64, 64))
        normalized_frame = resized_frame / 255
        predicted_labels_probabilities = model_new.predict(np.expand_dims(normalized_frame, axis=0))[0]
        predict_prob_dequee.append(predicted_labels_probabilities)
        if len(predict_prob_dequee) == moving_avg_window_size:
            predict_prob_np = np.array(predict_prob_dequee)
            predict_prob_avg = predict_prob_np.mean(axis=0)
            predicted_label = np.argmax(predict_prob_avg)
            if predicted_label==0:
                final_prediction = "Violence"
            else:
                final_prediction = "Non Violence"
            print(final_prediction)
            cv2.putText(frame,final_prediction,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        video_writer.write(frame)
    print("prediction Done✅")
    video_writer.release()
    video_reader.release()
print("prediction done")

def saved_model_evaluating(x_test,y_test):
    model= load_model('simpleCnnModel.h5')
    print(model.evaluate(x_test,y_test))