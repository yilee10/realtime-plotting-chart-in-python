import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import math 
import random
import winsound as sd
import time

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

def get_error_matrix(sin_y,errorThreshold):
    #(batchsize,한번에 볼 시점,한 시점의 feature) (8,10,1)
    ################################################## 그래프 초기화 ###################
    start_time = time.time()
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

    #LSTM
    #model.add(LSTM(128, input_shape=(10, 1)))  #input_shape=(ntime, 1)
    #model.add(Dense(1))

    #LSTM-AE
    model.add(Dense(128, input_shape=(10, 1)))
    model.add(Dense(64, activation='relu',kernel_initializer='random_uniform'))
    model.add(Dense(32, activation='relu',kernel_initializer='random_uniform'))
    model.add(Dense(64, activation='relu',kernel_initializer='random_uniform'))
    model.add(Dense(128, activation='relu',kernel_initializer='random_uniform'))
    model.add(Dense(1))
    

    model.compile(loss='mean_squared_error', optimizer='adam')
    ########################################### model ###########################

    #오차 행렬
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    nowtime = 0
    train_cnt = 0
    after_n_train = 100 # train 시작
    start_predcit = 500 #300부터 예측 시작
    end_predict = 1500 #예측 끝
    update_train = 10 # 30번마다 새롭게 업데이트
    while(nowtime < end_predict): 
        #오차행렬 구하기 위한 초기화 오류가 일어 났을 때를 P라고 하자
        modelP = 0
        modelN = 0
        trueP = 0
        trueN = 0

        nowtime += 1
        ### 그래프 이동하게###
        if (nowtime>100): ## 그래프 창에 100개의 점만 표시되도록
            time_tar_pred.pop(0)
            val_target.pop(0)
            val_predict.pop(0)
        if (nowtime > 100): #오차는 100개 점 봄
            time_error.pop(0)
            val_loss.pop(0)

        ### 값 입력 ###
        #ax0
        time_tar_pred.append(nowtime)
        rand_choice = 0
        if (nowtime > start_predcit): # nowtime start_predcit 이상 됐을 때 오류 생긴다고 하자
            ax1.set_ylim([-0.001, 0.01])
            random_y = 0.1*random.gauss(0, 1)
            rand_list = [0 for i in range(10)]
            rand_list.append(random_y)
            rand_choice = random.choice(rand_list)
            if(rand_choice != 0):
                #print("Error",rand_choice)           
                trueP = 1
                #sd.Beep(2000, 3000)
            else: # 실제값이 이상없는 값 일경우
                trueN = 1
        target = sin_y[nowtime%50] + rand_choice
        val_target.append(target)
        if(nowtime < after_n_train+2):
            val_predict.append(-1.1)
        #ax1
        time_error.append(nowtime)
        errorMSE = (1.0/2.0)*(val_target[-1]-val_predict[-1])**2
        val_loss.append(errorMSE)
        # error 가 기준 값 이상일 때 이상으로 감지
        if (nowtime > start_predcit):
            #print("===========================")
            #print("errorMSE",errorMSE)
            #errorThreshold = 0.0005
            if(errorMSE>errorThreshold):
                modelP = 1
            else:
                modelN = 1

            if(modelP==1 and trueP == 1):
                tp += 1
            elif(modelP == 1 and trueN == 1):
                fp += 1
            elif(modelN == 1 and trueN == 1):
                tn += 1
            elif(modelN == 1 and trueP == 1):
                fn += 1
            
            #print(modelP,modelN,trueP,trueN)
            #print(tp,fp,tn,fn)
            #print("nowtime",nowtime)
            if(tp+tn+fp+fn != 0):
                Accuracy = float(tp+tn)/float(tp+tn+fp+fn)
                #print("Accuracy",Accuracy) # 전체 예측 중 맞게 예측
            if(tp+fn != 0):
                Recall = float(tp)/float(tp+fn)
                #print("Recall,Sensitivity",Recall) # 실제 오류 즁 오류라고 예측한 것
            if(tp+fp != 0):
                Percision = float(tp)/float(tp+fp)
                #print("Percision",Percision) # 오류라고 예측한 것 중 실제 오류
            #print("--------------------------")
            if(fp+tn != 0):
                Specificity = float(tn)/float(fp+tn)
                #print("Specificity",Specificity) # 오류가 아니라고 예측 한 것 중 정말 오류가 아닌 것
            #print("===========================")
            #time.sleep(2)
            #if(rand_choice != 0):
                #time.sleep(8)

        train_cnt += 1
        if nowtime > after_n_train: #100번이후로 train 진행

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
            if train_cnt > update_train and nowtime < start_predcit: # update번 마다 업데이트 start_predcit번 까지만(and nowtime < start_predcit) 없으면 계속 업데이트
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
    plt.close()
    end_time = time.time()
    runtime = end_time-start_time
    print("runtime",runtime)
    return Accuracy,Recall,Percision,Specificity,runtime



errorThreshold = 0.0002
error_list = [["Threshold", "Accuracy","Recall","Percision","Specificity","Time"]]
while(errorThreshold<0.0011):
    arps = []
    e_list = []
    A_list = []
    R_list = []
    P_list = []
    S_list = []
    T_list = []
    for i in range(5):
        Accuracy,Recall,Percision,Specificity,runtime = get_error_matrix(sin_y,errorThreshold)
        e_list.append(errorThreshold)
        A_list.append(Accuracy)
        R_list.append(Recall)
        P_list.append(Percision)
        S_list.append(Specificity)
        T_list.append(runtime)
        print(errorThreshold,i,Accuracy,Recall,Percision,Specificity)

    arps.append(e_list)
    arps.append(A_list)
    arps.append(R_list)
    arps.append(P_list)
    arps.append(S_list)
    arps.append(T_list)

    arps_np = np.array(arps)
    arps_np_mean = np.mean(arps_np, axis=1)#.tolist()
    print("arps_np_mean",arps_np_mean)
    error_list.append(np.round(arps_np_mean, decimals=5))
    errorThreshold += 0.0001
print(np.array(error_list))
# CSV 파일로 저장
file_path = '/Users/yirang/Desktop/data_AEmodel.csv'  # 사용자명은 본인의 계정명으로 변경해야 합니다.
np.savetxt(file_path, error_list, delimiter=',', fmt='%s')
#print("Accuracy,Recall,Percision,Specificity",Accuracy,Recall,Percision,Specificity)