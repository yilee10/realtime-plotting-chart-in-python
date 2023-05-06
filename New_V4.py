# 난수 만들어서 노이즈와 전압값 커졌을 때 저장
import matplotlib.pyplot as plt
#import serial
#ser = serial.Serial('COM3',115200)

import random
import numpy as np

x = []
y = []
i = 0 # 현재 점의 인덱스
cnt = 0 # 한번에 업데이트 할 점의 개수
ADCvalue = 0

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

def Receive_data(n):
    x.append(i)
    ADCvalue = random.randrange(1,5)# int(ser.readline().decode().strip()) # 입력
    for k in range(n): 
        if ((i +50) % 100 == k): #50번째 마다 값 커지게 임의로 만들어 둠
            ADCvalue = ADCvalue + random.randrange(10,30) # 평상시 보다 큰 전압값 들어 왔을 때
    y.append(ADCvalue)
    return ADCvalue

def Draw_graph(cnt,perXpoint,speed):
    line.set_xdata(x)
    line.set_ydata(y)
    if (cnt % perXpoint == 0): # 200개마다 그래프 그림 -> 0.1초마다 변경
        #print("Graph")
        cnt = 0
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(speed)
    cnt = cnt + 1
    return cnt

Initialization_graph(15,4) # 창 크기 설정
while (1):
    Move_graph(50) #한 창에 표시되는 점100개
    ADCvalue = Receive_data(1) # arg 횟수 만큼 커진 값(Thresold 이상의 값) 들어옴
    cnt = Draw_graph(cnt,10,1) # 10개의 값을 1초마다 그림
    i = i + 1
    if (ADCvalue > 10): # 결과 값 저장 : 10을 Thresold로 처음 Thresold를 넘는 값이 감지되면
        new_i = 0 #그래프가 올라갔다가 내려오는 값 까지 찍기 위해서
        while((ADCvalue > 10) or (new_i < 10)): # Thresold를 넘지 않는 값이 나올 때 까지 
            Move_graph(50)
            #print("2:",i)
            i = i + 1
            ADCvalue = Receive_data(5)
            cnt = Draw_graph(cnt,10,5) #new_i값이 2번째 인자 이상 돌아야 그래프가 그려짐
            new_i = new_i + 1 
        filename = f"test/plot_{i}.png"  # 파일 이름 설정
        plt.savefig(filename)
        np.savetxt(f"test/data_{i}.txt", np.column_stack((x, y)))  # 값 저장

    