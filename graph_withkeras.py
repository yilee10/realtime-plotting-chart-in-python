import serial
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

ser = serial.Serial('COM6',115200)
################################################## 그래프 ###################
graph_x = []
graph_loss_x =[]
graph_val0 = []
graph_pred0 = []
graph_loss =[]

plt.ion()   
fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1,figsize=(15, 8)) ### 한 창에 나타날 그래프 수 결정
line_val0, = ax0.plot(graph_x, graph_val0,label='Val0')
line_pred0, = ax0.plot(graph_x, graph_pred0,label='Pred0')
#####
line_loss, = ax1.plot(graph_loss_x, graph_loss,label='Loss')

ax0.set_ylim([-0.1, 1.1]) 
#ax1.set_ylim([-0.1, 0.6])   
################################################## 그래프 ###################


########################################### model ###########################
# LSTM 모델 구성
model = Sequential()
model.add(LSTM(128, input_shape=(100, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
########################################### model ###########################
#######################lose를 평탄화 하기위한 두 리스트의 차를 구하는 함수#########
def get_list_differences(list1, list2):
    differences = []
    for item1, item2 in zip(list1, list2):
        difference = abs(item1 - item2) #|실제값 - 예측 값| = loss
        differences.append(difference)
    return differences #각각의 원소마다 차가 나옴
#######################lose를 평탄화 하기위한 두 리스트의 차를 구하는 함수#########
cnt = 0
i = 0
prediction = 0.5
while(1):
    if (i>99): ## 그래프 창에 100개의 점만 표시되도록
        graph_x.pop(0)
        graph_val0.pop(0)
        graph_pred0.pop(0)
    if (i > 1000):
        graph_loss_x.pop(0)
        graph_loss.pop(0)

    ### 값 입력 ###
    i += 1 ## 시간축 추가
    graph_x.append(i)
    graph_loss_x.append(i)
    in_val0=float(ser.readline().decode().strip()) ##y축 Val 추가
    graph_val0.append(in_val0)
    
    print(len(graph_val0))
    if len(graph_val0) == 100: #데이터가 100개까지 들어오면
        model_val0 = np.array([graph_val0])
        model_val0 = np.reshape(model_val0,(1,100,1)) # keras 쓰기 위해 형식 맞췀
        prediction = model.predict(model_val0) # 현재 예측값이 한바퀴 돌아 
        
    ## y축 Pred 추가 여기에 예측값을 lstm 결과값으로 넣어야 하나 현재는 0.5넣어둠
    graph_pred0.append(prediction)
    graph_loss.append(sum(get_list_differences(graph_val0[-20:],graph_pred0[-20:]))) ## 각각의 원소 차를 다 더해야 하므로.sum , 100개 다 보면 느려서[-20:]

    # 모델을 새로운 데이터로 업데이트
    if len(graph_val0) == 100:
        X_train = np.array([graph_val0])
        X_train = np.reshape(X_train, (1, 100, 1))
        y_train = np.array([in_val0])
        model.fit(X_train, y_train, epochs=1, batch_size=50, verbose=0)

    ### 그래프선에 값 맵핑###
    line_val0.set_xdata(graph_x)
    line_val0.set_ydata(graph_val0)
    line_pred0.set_xdata(graph_x)
    line_pred0.set_ydata(graph_pred0)
    line_loss.set_xdata(graph_loss_x)
    line_loss.set_ydata(graph_loss)

    ### 그래프 나타내기 ###
    ax0.relim()
    ax0.autoscale_view()
    ax0.legend()

    ax1.relim()
    ax1.autoscale_view()
    ax1.legend()

    plt.draw()
    plt.pause(0.01)

    cnt += 1