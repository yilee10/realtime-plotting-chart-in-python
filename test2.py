import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import math
import random
import winsound as sd


pi = 3.14159265358979
num_data_points = 50#*4 #np.linspace()기본이 50개
sin_x = []
sin_y = []
start = -2*pi#*4
finsih = 2*pi#*4
for i in range(num_data_points): # 1부터 num_data_points까지 
    now_x = start +(finsih-start)*(i/(num_data_points)) # start포함  finsih 포함X => 50개 점 이므로 num_data_points #start,finsih둘 다 포함이면 => 50개 점 이므로 num_data_points-1
    sin_x.append(now_x)
    now_y = math.sin(now_x) #노이즈 없는 sin그래프 추가
    sin_y.append(now_y)


#(batchsize,한번에 볼 시점,한 시점의 feature) (8,10,1)
################################################## 그래프 초기화 ###################

time_tar_pred = [] # 실제값,예측값 그래프 x축인 시점
val_target = [] #실제값
val_predict = [] #예측값

time_error = []  #오차 그래프의 x축 시점
val_loss = [] #오차

plt.ion()   
fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1,figsize=(15, 8)) #fig:하나의 창 #ax0,ax1:서브플롯(좌표평면) 
#ax0 첫번째 좌표평면(x축과 y축과 label정보 입력하여 line 생성)
line_target, = ax0.plot(time_tar_pred, val_target,label='Target') 
line_predict, = ax0.plot(time_tar_pred, val_predict,label='Predict')
#ax1 두번째 좌펴평면(x축과 y축과 label정보 입력하여 line 생성)
line_loss, = ax1.plot(time_error, val_loss,label='Loss')

ax0.set_ylim([-1.1, 1.1]) 
################################################## 그래프 초기화 ###################




########################################### model ###########################
# LSTM 모델 구성
model = Sequential()
model.add(LSTM(128, input_shape=(10, 1)))  #input_shape=(ntime, 1)
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
########################################### model ###########################

time = 0
train_cnt = 0
after_n_train = 100
while(1):
    time += 1
    ### 그래프 이동하게###
    if (time>100): ## 그래프 창에 100개의 점만 표시되도록
        time_tar_pred.pop(0)
        val_target.pop(0)
        val_predict.pop(0)
    if (time > 100): #오차는 100개 점 봄
        time_error.pop(0)
        val_loss.pop(0)

    ### 값 입력 ###
    #ax0
    time_tar_pred.append(time)
    rand_choice = 0
    if (time > 500): # time 500 이상 됐을 때 오류 생긴다고 하자
        ax1.set_ylim([-0.001, 0.01])
        random_y = 0.1*random.gauss(1, 1)
        rand_list = [0 for i in range(29)]
        rand_list.append(random_y)
        rand_choice = random.choice(rand_list)
        if(rand_choice != 0):
            print("Error")
            sd.Beep(2000, 3000)
    target = sin_y[time%50] + rand_choice
    val_target.append(target)
    if(time < after_n_train+2):
        val_predict.append(-1.1)
    #ax1
    time_error.append(time)
    errorMSE = (1.0/2.0)*(val_target[-1]-val_predict[-1])**2
    val_loss.append(errorMSE)

    train_cnt += 1
    if time > after_n_train: #100번이후로 train 진행

        ###그래프 나타내기###
        line_target.set_xdata(time_tar_pred)
        line_target.set_ydata(val_target)
        line_predict.set_xdata(time_tar_pred)
        line_predict.set_ydata(val_predict)
        line_loss.set_xdata(time_error)
        line_loss.set_ydata(val_loss)

        ax0.relim()
        ax0.autoscale_view()
        ax0.legend()

        ax1.relim()
        ax1.autoscale_view()
        ax1.legend()

        plt.draw()
        plt.pause(0.01)

         # train
        if train_cnt > 30: # 30번 마다 업데이트
            #print("Update")
            ntime = 10
            nsample = len(val_target)-ntime
            val_target_np = np.array(val_target)
            X_ntime = [0 for i in range(nsample)]
            for i in range(nsample): #n_time개의 시점을 하나의 input으로 묶어주기 위해
                X_ntime[i] = val_target_np[i:i+ntime]
            X_train = np.reshape(X_ntime,(nsample,ntime,1)) #2차원 배열을 3차원 배열으로
            y_train = val_target_np[ntime:ntime+nsample]
            y_train = np.reshape(y_train,(nsample,1))
            """
            print("X_train",np.array(X_train).shape) #n_time개 만큼 묶인 것
            print(np.array(X_train))
            print("y_train",np.array(y_train).shape) #n_time개 만큼 묶인 것
            print(np.array(y_train))
            """
            model.fit(X_train, y_train, epochs=1, batch_size=9, verbose=0)
            train_cnt = 0
    
        # predict
        val_target_np = np.array(val_target)
        X_ntime = val_target_np[-ntime:]
        X_test = np.reshape(X_ntime,(1,ntime,1))
        y_test = model.predict(X_test,verbose=0)
        #print("X_ntime",X_ntime)
        val_predict.append(y_test)
   