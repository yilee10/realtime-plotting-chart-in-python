import serial
import matplotlib.pyplot as plt

ser = serial.Serial('COM6',115200)

graph_x = []
graph_val0 = []
graph_pred0 = []
graph_loss =[]

plt.ion()   
fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1,figsize=(15, 8)) ### 한 창에 나타날 그래프 수 결정
line_val0, = ax0.plot(graph_x, graph_val0,label='Val0')
line_pred0, = ax0.plot(graph_x, graph_pred0,label='Pred0')
#####
line_loss, = ax1.plot(graph_x, graph_loss,label='Loss')

ax0.set_ylim([-0.1, 1.1]) 
ax1.set_ylim([-0.1, 0.6])   

cnt = 0
i = 0
while(1):
    if (i>50): ## 그래프 창에 50개의 점만 표시되도록
        graph_x.pop(0)
        graph_val0.pop(0)
        graph_pred0.pop(0)
        graph_loss.pop(0)
    
    ### 값 입력 ###
    i += 1 ## 시간축 추가
    graph_x.append(i)
    in_val0=float(ser.readline().decode().strip()) ##y축 Val 추가
    graph_val0.append(in_val0)
    ## y축 Pred 추가 여기에 예측값을 lstm 결과값으로 넣어야 하나 현재는 0.5넣어둠
    graph_pred0.append(0.5)
    graph_loss.append(abs(in_val0-0.5)) ## |실제값 - 예측 값| = loss
    print(in_val0-0.5)
    ### 그래프선에 값 맵핑###
    line_val0.set_xdata(graph_x)
    line_val0.set_ydata(graph_val0)
    line_pred0.set_xdata(graph_x)
    line_pred0.set_ydata(graph_pred0)
    line_loss.set_xdata(graph_x)
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