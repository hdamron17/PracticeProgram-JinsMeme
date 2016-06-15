'''
Created on Jun 10, 2016

@author: HDamron1
'''

import matplotlib.pyplot as plt
import numpy as np
import math
import random

def movingAvg(data, time, period):
    avgData, avgTime = [], []
    for i in range(0, len(data) - period + 1):
        dataSum = 0
        timeSum = 0
        for j in range(i, i + period):
            dataSum += data[j]
            timeSum += time[j]
        avgData.append(dataSum / period)
        avgTime.append(timeSum / period)
    return (avgTime, avgData)

def amplitude(data, time, period):
    amplt, slidingTime = [], []
    for i in range(0, len(data) - period + 1):
        periodArray = []
        timeSum = 0
        for j in range(i, i + period):
            periodArray.append(data[j])
            timeSum += time[j]
        amplt.append(np.max(periodArray) - np.min(periodArray))
        slidingTime.append(timeSum / period)
    return (slidingTime, amplt)
    
def peaksAndValleys(data, time):    
    increasing = data[1] - data[0] > 0
    peaks = []
    peakTimes = []
    valleys = []
    valleyTimes = []
    prev = data[0]
    prevTime = time[0]
    for i in range(0, len(data)):
        if (data[i] < prev) and increasing:
            increasing = not increasing
            peaks.append(prev)
            peakTimes.append(prevTime)
        if (data[i] > prev) and not increasing:
            increasing = not increasing
            valleys.append(prev)
            valleyTimes.append(prevTime)
        prev = data[i]
        prevTime = time[i]
    return (peaks, peakTimes, valleys, valleyTimes)

def stepAmplitudes(data):
    peaks, valleys = peaksAndValleys(data)
    pulses = np.min([len(peaks), len(valleys)])
    amplt = []
    for i in range(0, pulses):
        amplt.append(peaks[i] - valleys[i])
    return amplt

def stepCount(peaks, valleys, ampltThreshold):
    pulses = np.min([len(peaks), len(valleys)])
    count = 0;
    for i in range(0, pulses):
        if abs(peaks[i] - valleys[i]) > ampltThreshold:
            count += 1
    return count

if __name__ == '__main__':
    testdata = np.genfromtxt('../steps-data3-numerical.csv', delimiter=',')
    
    time = []
    
    bSpeed = []
    bStrength = []
    
    eyeUp = []
    eyeDown = []
    eyeLeft = []
    eyeRight = []
    
    accX = []
    accY = []
    accZ = []
    
    roll = []
    pitch = []
    yaw = []
    
    segment = []
    
    for row in testdata:
        time.append(row[1])
    
        # bSpeed.append(row[2])
        # bStrength.append(row[3])
        #
        # eyeUp.append(row[4])
        # eyeDown.append(row[5])
        # eyeLeft.append(row[6])
        # eyeRight.append(row[7])
    
        accX.append(row[8])
        accY.append(row[9])
        accZ.append(row[10])
    
        # roll.append(row[11])
        # pitch.append(row[12])
        # yaw.append(row[13])
    
        segment.append(row[18])
    
    
    # plt.figure(1)
    # plt.plot(time, bSpeed, 'b', label='Blink Speed')
    # plt.plot(time, bStrength, 'r', label='Blink Strength')
    # plt.title('Blink Speed and Strength versus Time')
    # plt.legend()
    #
    # plt.figure(2)
    # plt.plot(time, eyeUp, 'b', label='Eye Up')
    # plt.plot(time, eyeDown, 'r', label='Eye Down')
    # plt.plot(time, eyeLeft, 'g', label='Eye Left')
    # plt.plot(time, eyeRight, 'k', label='Eye Right')
    # plt.title('Eye Direction versus Time')
    # plt.legend()
    #
#     plt.figure(6)
#     plt.plot(time, accX, 'b', label='X acceleration')
#     plt.plot(time, accY, 'r', label='Y acceleration')
#     plt.plot(time, accZ, 'g', label='Z acceleration')
#     plt.xlabel('Time')
#     plt.ylabel('Acceleration')
#     plt.title('3D Acceleration versus Time')
#     plt.legend()
    #
    # plt.figure(4)
    # plt.plot(time, roll, 'b', label='Roll')
    # plt.plot(time, pitch, 'r', label='Pitch')
    # plt.plot(time, yaw, 'g', label='Yaw')
    # plt.title('Roll, Pitch, and Yaw versus Time')
    # plt.legend()
    
    vector = []
    for i in range(len(time)):
        vector.append(math.sqrt((accX[i] ** 2) + (accY[i] ** 2) + (accZ[i] ** 2)))
    
    accZ0 = [], []
    accZ1 = [], []
    accZ2 = [], []
    
    for i in range(0, len(accZ)):
        if(segment[i] == 0):
            accZ0[0].append(accZ[i])
            accZ0[1].append(time[i])
        elif(segment[i] == 1):
            accZ1[0].append(accZ[i])
            accZ1[1].append(time[i])
        elif(segment[i] == 2):
            accZ2[0].append(accZ[i])
            accZ2[1].append(time[i])
    
#     plt.figure(5)
#     plt.plot(accZ0[1], accZ0[0], 'r', label='Shaking head')
#     plt.plot(accZ1[1], accZ1[0], 'g', label='Walking')
#     plt.plot(accZ2[1], accZ2[0], 'b', label='Trash data')
#     plt.xlabel('Time')
#     plt.ylabel('Z Acceleration')
#     plt.legend()
    
    segmentedVector = [] # Segmented every 5 seconds
    endTime = 2
    temp = [0, []]
    for i in range(len(time)):
        if(time[i] < endTime):
            temp[1].append(vector[i])
        else:
            segmentedVector.append(temp)
            temp = [endTime, [vector[i]]]
            endTime += 2
    
    STDvTime = [[], []]
    for i in range(0, len(segmentedVector)):
        STDvTime[0].append(segmentedVector[i][0])
        STDvTime[1].append(np.std(segmentedVector[i][1]))
    
    segmentMeans = [] # Mean value for each 5 second interval
    for seg in segmentedVector:
        segmentMeans.append(np.mean(seg[1]))
    
    possibleSteps = [[], []] # Time in 0 and vectors in 1
    start = vector[i]
    crossed = False
    start = vector[0] - segmentMeans[i] > 0 # True if beginning is below mean, false if above mean
    temp = [[], []]
    for i in range(0, len(time)):
        if (vector[i] < segmentMeans[int(time[i] % 5)] != start) and not crossed:
            crossed = True
        if (vector[i] > segmentMeans[int(time[i] % 5)] != start) and crossed:
            crossed = False
            possibleSteps[0].append(temp[0])
            possibleSteps[1].append(temp[1])
            temp = [[], []]
        temp[0].append(time[i])
        temp[1].append(vector[i])
    
    
        # if(not crossed and startDirection xor vector[i] < segmentMeans[int(time[i] % 5)]):
        #     crossed = True
        #     temp[0].append(time[i])
        #     temp[1].append(vector[i])
        # elif(crossed and vector[i] > segmentMeans[int(time[i] % 5)]):
        #     crossed = False
        #     possibleSteps[0].append(temp[0])
        #     possibleSteps[1].append(temp[1])
        #     temp = [[], []]
        # else:
        #     temp[0].append(time[i])
        #     temp[1].append(vector[i])
    
    amplitudes = []
    for possibleStep in possibleSteps[1]:
        amplitudes.append(np.max(possibleStep) - np.min(possibleStep))
    
    stepTimes = []
    for stepTimeArray in possibleSteps[0]:
        stepTimes.append(stepTimeArray[0])
    
    steps = [[], []]
    THRESHOLD = 2
    for i in range(0, len(stepTimes)):
        if(amplitudes[i] > THRESHOLD):
            steps[0].append(stepTimes[i])
            steps[1].append(amplitudes[i])
    
    # plt.figure(1)
    # plt.plot(accY0[1], accY0[0], 'b', label='Standing')
    # plt.plot(accY1[1], accY1[0], 'r', label='Walking')
    # plt.plot(accY2[1], accY2[0], 'g', label='Jogging')
    # plt.title('Y Acceleration During Activity versus Time')
    # plt.legend()
    
#     plt.figure(1)
#     plt.plot(time, vector, 'b')
#     plt.xlabel('Time')
#     plt.ylabel('Magnitude')
#     plt.title('Vector magnitude vs Time')
    
#     plt.figure(2)
#     plt.plot(STDvTime[0], STDvTime[1], 'k', label='Standard Deviation')
#     plt.xlabel('Time')
#     plt.ylabel('Standard Deviation')
#     plt.title('Standard Deviation versus Time')
#     plt.legend()
    
#     plt.figure(3)
#     plt.plot(stepTimes, amplitudes, 'k', label='Amplitude')
#     plt.xlabel('Time')
#     plt.ylabel('Amplitude')
#     plt.title('Amplitude versus Time')
#     plt.legend()
    
#     plt.figure(4)
#     plt.plot(steps[0], steps[1], 'bo', label='Steps')
#     plt.xlabel('Time')
#     plt.ylabel('Step Magnitude')
#     plt.title('Steps and their Magnitudes vs Time')
#     plt.legend()
#     plt.show()

    plt.figure(1)
    plt.plot(time, accZ, 'k', label='Original')
    movingAvg = movingAvg(accZ, time, 5)
    plt.plot(movingAvg[0], movingAvg[1], 'b--', label='Moving Average')
    plt.xlabel('Time')
    plt.ylabel('Z Acceleration')
    plt.title('Moving Average')
    plt.legend()
    
#     plt.figure(2)
#     amplt = amplitude(movingAvg[1], movingAvg[0], 5)
#     plt.plot(amplt[0], amplt[1], 'k', label='Amplitudes')
#     plt.xlabel('Time')
#     plt.ylabel('Z Acceleration')
#     plt.legend()
    
    peaks, peakTimes, valleys, valleyTimes = peaksAndValleys(movingAvg[1], movingAvg[0])
    print('Number of steps : %d' % stepCount(peaks, valleys, 3.6))
    
    plt.figure(3)
    plt.plot(peakTimes, peaks, 'bo', label='Peaks')
    plt.plot(valleyTimes, valleys, 'ro', label='Valleys')
    plt.xlabel('Pulse Index')
    plt.ylabel('Z Acceleration')
    plt.legend()
    plt.title('Peaks and Valleys')
    
    plt.figure(4)
    plt.hist(stepAmplitudes(movingAvg[1]), bins=100)
    plt.xlabel('Amplitude')
    plt.ylabel('Number of Pulses')
    plt.title('Pulse Amplitude Histogram')
    
    plt.show()