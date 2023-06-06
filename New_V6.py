# 연결
import matplotlib.pyplot as plt
import serial
ser = serial.Serial('COM3',115200)

import time
import numpy as np


x = []
y = []
i = 0 # 현재 점의 인덱스
cnt = 0 # 한번에 업데이트 할 점의 개수
ADCvalue = 0

start = 0

def Initialization_graph(sizeX,sizeY):
    global fig
    global line
    global ax
    plt.ion()   # interactive 모드 활성화 : 그래프가 그려지는 동안에도 코드 실행됨
    fig, ax = plt.subplots(figsize=(sizeX,sizeY))
    line, = ax.plot(x, y)

def Move_graph(pointNum): ## 하나에 그래프에 나타낼 점의 수-> 그래프 이동 : cnt보다 커야 다 나타낼 수 있음 
    if(i>pointNum): 
        x.pop(0)
        y.pop(0)

def Receive_data():
    global start
    x.append(i)
    ADCvalue = int(ser.readline().decode().strip())
    if(i%10000 == 0):
        print(time.time()-start)
        start = time.time()
    #print(ADCvalue) #############################################################################################################삭제
    y.append(ADCvalue)
    return ADCvalue

def Draw_graph(cnt,perXpoint,speed):
    line.set_xdata(x)
    line.set_ydata(y)
    if (cnt % perXpoint == 0):
        #print("Graph")
        cnt = 0
        ax.relim()
        ax.autoscale_view()
        plt.ylim([800, 1300]) # y축 사이즈 설정 ###########################################################################변경
        plt.draw()
        plt.pause(speed)
    cnt = cnt + 1
    return cnt

Initialization_graph(13,4) # 창 크기 설정
while (1):
    Move_graph(400) #한 창에 표시되는 점400개 #######################################################################
    ADCvalue = Receive_data() 
    cnt = Draw_graph(cnt,200,0.01) # 200개의 값을 0.1초마다 그림 ########################################################################################두번째 인자 200으로 
    i = i + 1
    if (ADCvalue > 1100): # 결과 값 저장 : Thresold로 처음 Thresold를 넘는 값이 감지되면 ###################################################### 변경
        new_i = 0 #그래프가 올라갔다가 내려오는 값 까지 찍기 위해서
        while((ADCvalue > 1100) or (new_i < 100)): # Thresold를 넘지 않는 값이 나올 때 까지 
            Move_graph(400)######################################################################################
            i = i + 1
            ADCvalue = Receive_data()
            cnt = Draw_graph(cnt,200,0.01) #new_i값이 2번째 인자 이상 돌아야 그래프가 그려짐 ##############################################################두번째 인자 200으로
            new_i = new_i + 1 
        filename = f"particular_value/plot/plot_{i}.png"  # 파일 이름 설정
        plt.savefig(filename)
        np.savetxt(f"particular_value/data/data_{i}.txt", np.column_stack((x[-50:], y[-50:])))  # 값 저장

    