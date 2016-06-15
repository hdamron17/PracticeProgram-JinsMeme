'''
Created on Jun 10, 2016

@author: HDamron1
'''

import matplotlib.pyplot as plt
import numpy as np
# import math
import random
import csv
from sklearn import cluster, neighbors, svm, naive_bayes
from matplotlib.colors import ListedColormap

def movingAvg(data, time, period, segments):
    avgData, avgTime, avgSegment = [], [], []
    for i in range(0, len(data) - period + 1):
        dataSum = 0
        timeSum = 0
        firstSeg = segments[i]
        include = True
        for j in range(i, i + period):
            if segments[j] != firstSeg:
                include = False
                break
            dataSum += data[j]
            timeSum += time[j]
        if include:
            avgData.append(dataSum / period)
            avgTime.append(timeSum / period)
            avgSegment.append(firstSeg)
    return (avgTime, avgData, avgSegment)

def amplitude(data, time, period, segments):
    amplt, slidingTime, avgSegment = [], [], []
    for i in range(0, len(data) - period + 1):
        periodArray = []
        timeSum = 0
        firstSeg = segments[i]
        include = True
        for j in range(i, i + period):
            if segments[j] != firstSeg:
                include = False
                break
            periodArray.append(data[j])
            timeSum += time[j]
        if include:
            amplt.append(np.max(periodArray) - np.min(periodArray))
            slidingTime.append(timeSum / period)
            avgSegment.append(firstSeg)
    return (slidingTime, amplt, avgSegment)

def standardDev(data, time, period, segments):
    stDev, slidingTime, avgSegment = [], [], []
    for i in range(0, len(data) - period + 1):
        periodArray = []
        timeSum = 0
        firstSeg = segments[i]
        include = True
        for j in range(i, i + period):
            if segments[j] != firstSeg:
                include = False
                break
            periodArray.append(data[j])
            timeSum += time[j]
        if include:
            stDev.append(np.std(periodArray))
            slidingTime.append(timeSum / period)
            avgSegment.append(firstSeg)
    return (slidingTime, stDev, avgSegment)

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

def pulseAmplitudes(stepData): 
    # stepData received from peaksAndValleys()
    peaks, peakTimes, valleys, valleyTimes = stepData
    pulses = np.min([len(peaks), len(valleys)])
    amplt = []
    pulseTimes = []
    for i in range(0, pulses):
        amplt.append(peaks[i] - valleys[i])
        pulseTimes.append((peakTimes[i] + valleyTimes[i]) / 2)
    return (amplt, pulseTimes)

def pulseTimeLengths(stepData):
    # stepData received from peaksAndValleys()
    peaks, peakTimes, valleys, valleyTimes = stepData
    pulses = np.min([len(peaks), len(valleys)])
    timeLengths = []
    pulseTimes = []
    for i in range(0, pulses):
        timeLengths.append(abs(valleyTimes[i] - peakTimes[i]))
        pulseTimes.append((peakTimes[i] + valleyTimes[i]) / 2)
    return (timeLengths, pulseTimes)
    
def stepCount(peaks, valleys, ampltThreshold):
    pulses = np.min([len(peaks), len(valleys)])
    count = 0;
    for i in range(0, pulses):
        if abs(peaks[i] - valleys[i]) > ampltThreshold:
            count += 1
    return count

def getDataSet(versionNumber=1):
    fileName = format('../steps-data%d-numerical.csv' % versionNumber)
    data = np.genfromtxt(fileName, delimiter=',')
    time = []
    accX = []
    accY = []
    accZ = []
    segment = []
    
    for row in data:
        time.append(row[1])
        accX.append(row[8])
        accY.append(row[9])
        accZ.append(row[10])
        segment.append(row[18])
    return (time, accX, accY, accZ, segment)

# def writeTrainingSet(pulseTimes, pulseAmplitudes, timeLengths):
#     with open('training-set1-numerical.csv', 'wt') as csvFile:
#         output = csv.writer(csvFile, delimiter=',')
# #         output.writerow(['Amplitude', 'Time Length', 'Target'])
#         target = 0
#         for i in range(0, len(pulseTimes)):
#             if pulseTimes[i] < 20:
#                 target = 0
#             elif pulseTimes[i] > 50 and pulseTimes[i] < 80:
#                 target = 1
#             elif pulseTimes[i] > 110 and pulseTimes[i] < 130:
#                 target = 2
#             output.writerow([pulseAmplitudes[i], pulseTimeLengths[i], target])

def writeTrainingSet(versionNum=1):
    npData = np.genfromtxt('../training-set-raw.csv', delimiter=',', skip_header=1, usecols=[2,11,19])
    time = npData[:,0].tolist()
    accZ = npData[:,1].tolist()
    filteredTime, filteredAccZ = movingAvg(accZ, time, 5)
    stepData = peaksAndValleys(filteredAccZ, filteredTime)
    amplitudes, pulseTimeArray = pulseAmplitudes(stepData)
    timeLengths = pulseTimeLengths(stepData)[0]
    
    with open('training-set%d.csv' % versionNum, 'wt') as csvFile:
        outFile = csv.writer(csvFile, delimiter=',')
        for i in range(0, len(amplitudes)):
            segment = -1
            if pulseTimeArray[i] >= 169.9806499:
                segment = 2
            elif pulseTimeArray[i] >= 62.73803997:
                segment = 1
            else: 
                segment = 0
            print('%f,%f,%d' % (amplitudes[i], timeLengths[i], segment))
            outFile.writerow([amplitudes[i], timeLengths[i], segment])
            
def getTrainingSet(versionNum=1, testSetFraction=0.75):
    fileName = format('../training-set%d.csv' % versionNum)
    npData = np.array(np.genfromtxt(fileName, delimiter=',', skip_header=1))
    data = npData.T
    inputArray = npData[:, :2].tolist()
    outputArray = data[2, :].tolist()
    inputTestArray = []
    outputTestArray = []
    
    for i in range(0, int(len(inputArray) * (1 - testSetFraction))):
        index = random.randint(0, len(inputArray) - 1)
        inputTestArray.append(inputArray.pop(index))
        outputTestArray.append(outputArray.pop(index))
    return (inputArray, outputArray, inputTestArray,  outputTestArray)

def splitTrainingSet(time, amplitude, stDev, segments, trainFraction = .75):
    trainTime = [x for x in time]
    testTime = []
    trainAmpDev = []
    for i in range(len(amplitude)):
        trainAmpDev.append([amplitude[i], stDev[i]])
    testAmpDev = []
    trainSeg = segments
    testSeg = []
    for i in range(0, int(len(time) * (1 - trainFraction))):
        index = random.randint(0, len(trainTime) - 1)
        testAmpDev.append(trainAmpDev.pop(index))
        testSeg.append(trainSeg.pop(index))
        testTime.append(trainTime.pop(index))
    return (trainTime, testTime, trainAmpDev, testAmpDev, trainSeg, testSeg)

def printPrecisionAndRecall(theoreticalOutput, actualOutput):
    tp = np.zeros(int(np.max(theoreticalOutput)) + 1).tolist()
    fp = np.zeros(int(np.max(theoreticalOutput)) + 1).tolist()
    fn = np.zeros(int(np.max(theoreticalOutput)) + 1).tolist()
    
    for i in range(0, len(actualOutput)):
        if(actualOutput[i] != theoreticalOutput[i]):
            fp[int(actualOutput[i])] += 1
            fn[int(theoreticalOutput[i])] += 1
        else:
            tp[int(actualOutput[i])] += 1
            
    precision = []
    recall = []
    for i in range(0, len(tp)):
        precision.append(tp[i] / (tp[i] + fp[i]))
        recall.append(tp[i] / (tp[i] + fn[i]))
    
    print('Standing Precision = %.4f ; Recall = %.4f' %(precision[0], recall[0]))
    print('Walking Precision = %.4f ; Recall = %.4f' %(precision[1], recall[1]))
    print('Jogging Precision = %.4f ; Recall = %.4f' %(precision[2], recall[2]))

def runSlideTraining():
    dataItems = getDataSet(5)
    segments = dataItems[4]
    times = dataItems[0]
    accZStuff = dataItems[3]
    dataItems = movingAvg(accZStuff, times, 5, segments)
    avgSegments = dataItems[2]
    avgTimes = dataItems[0]
    avgAccZ = dataItems[1]
    avgSlTime, avgAmplitude, avgSeg = amplitude(avgAccZ, avgTimes, 10, avgSegments)
    avgStdDev = standardDev(avgAccZ, avgTimes, 10, avgSegments)[1]
    trainTime, testTime, trainAmpDev, testAmpDev, trainSeg, testSeg = splitTrainingSet(avgSlTime, avgAmplitude, avgStdDev, avgSeg)

    for i in range(len(trainSeg)):
        trainSeg[i] = int(trainSeg[i])
    for i in range(len(testSeg)):
        testSeg[i] = int(testSeg[i])

    # plt.figure()
    # plt.plot(avgSlTime, avgAmplitude)
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    # plt.figure()
    # plt.plot(avgSlTime, avgStdDev)
    # plt.xlabel('Time')
    # plt.ylabel('Std Deviation')
    #classifying data
    classifier = neighbors.KNeighborsClassifier(15, weights='distance')
    classifier.fit(trainAmpDev, trainSeg)

    cmap_light = ListedColormap(['#FF3333', '#11FF11', '#3333FF'])
    cmap_bold = ListedColormap(['#FF0000', '#007022', '#0000FF'])

    testClassifications = classifier.predict(testAmpDev)

    plt.figure()
    plt.scatter(np.array(trainAmpDev)[:,1], np.array(trainAmpDev)[:,0], c=trainSeg, marker='x', edgecolor=None)
    plt.scatter(np.array(testAmpDev)[:,1], np.array(testAmpDev)[:,0], c=testClassifications, edgecolor=None)
    plt.xlim(0, np.array(trainAmpDev)[:,1].max() + 0.5)
    plt.ylim(0, np.array(trainAmpDev)[:,0].max() + 0.5)
    plt.xlabel('StdDev')
    plt.ylabel('Amplitude')
    plt.title('Standing, Walking, or Jogging?')

    printPrecisionAndRecall(testSeg, testClassifications)

    plt.show()

if __name__ == '__main__':
    runSlideTraining()
#     time, accX, accY, accZ, segment = getDataSet()
#
# #     plt.figure(1)
# #     plt.plot(time, bSpeed, 'b', label='Blink Speed')
# #     plt.plot(time, bStrength, 'r', label='Blink Strength')
# #     plt.title('Blink Speed and Strength versus Time')
# #     plt.legend()
# #
# #     plt.figure(2)
# #     plt.plot(time, eyeUp, 'b', label='Eye Up')
# #     plt.plot(time, eyeDown, 'r', label='Eye Down')
# #     plt.plot(time, eyeLeft, 'g', label='Eye Left')
# #     plt.plot(time, eyeRight, 'k', label='Eye Right')
# #     plt.title('Eye Direction versus Time')
# #     plt.legend()
#
# #     plt.figure(6)
# #     plt.plot(time, accX, 'b', label='X acceleration')
# #     plt.plot(time, accY, 'r', label='Y acceleration')
# #     plt.plot(time, accZ, 'g', label='Z acceleration')
# #     plt.xlabel('Time')
# #     plt.ylabel('Acceleration')
# #     plt.title('3D Acceleration versus Time')
# #     plt.legend()
#
# #     plt.figure(4)
# #     plt.plot(time, roll, 'b', label='Roll')
# #     plt.plot(time, pitch, 'r', label='Pitch')
# #     plt.plot(time, yaw, 'g', label='Yaw')
# #     plt.title('Roll, Pitch, and Yaw versus Time')
# #     plt.legend()
#
# #     vector = []
# #     for i in range(len(time)):
# #         vector.append(math.sqrt((accX[i] ** 2) + (accY[i] ** 2) + (accZ[i] ** 2)))
#
# #     accZ0 = [], []
# #     accZ1 = [], []
# #     accZ2 = [], []
# #
# #     for i in range(0, len(accZ)):
# #         if(segment[i] == 0):
# #             accZ0[0].append(accZ[i])
# #             accZ0[1].append(time[i])
# #         elif(segment[i] == 1):
# #             accZ1[0].append(accZ[i])
# #             accZ1[1].append(time[i])
# #         elif(segment[i] == 2):
# #             accZ2[0].append(accZ[i])
# #             accZ2[1].append(time[i])
#
# #     plt.figure(5)
# #     plt.plot(accZ0[1], accZ0[0], 'r', label='Shaking head')
# #     plt.plot(accZ1[1], accZ1[0], 'g', label='Walking')
# #     plt.plot(accZ2[1], accZ2[0], 'b', label='Trash data')
# #     plt.xlabel('Time')
# #     plt.ylabel('Z Acceleration')
# #     plt.legend()
#
# #     segmentedVector = [] # Segmented every 5 seconds
# #     endTime = 2
# #     temp = [0, []]
# #     for i in range(len(time)):
# #         if(time[i] < endTime):
# #             temp[1].append(vector[i])
# #         else:
# #             segmentedVector.append(temp)
# #             temp = [endTime, [vector[i]]]
# #             endTime += 2
#
# #     STDvTime = [[], []]
# #     for i in range(0, len(segmentedVector)):
# #         STDvTime[0].append(segmentedVector[i][0])
# #         STDvTime[1].append(np.std(segmentedVector[i][1]))
# #
# #     segmentMeans = [] # Mean value for each 5 second interval
# #     for seg in segmentedVector:
# #         segmentMeans.append(np.mean(seg[1]))
#
# #     possibleSteps = [[], []] # Time in 0 and vectors in 1
# #     start = vector[i]
# #     crossed = False
# #     start = vector[0] - segmentMeans[i] > 0 # True if beginning is below mean, false if above mean
# #     temp = [[], []]
# #     for i in range(0, len(time)):
# #         if (vector[i] < segmentMeans[int(time[i] % 5)] != start) and not crossed:
# #             crossed = True
# #         if (vector[i] > segmentMeans[int(time[i] % 5)] != start) and crossed:
# #             crossed = False
# #             possibleSteps[0].append(temp[0])
# #             possibleSteps[1].append(temp[1])
# #             temp = [[], []]
# #         temp[0].append(time[i])
# #         temp[1].append(vector[i])
#
#
# #         if(not crossed and startDirection xor vector[i] < segmentMeans[int(time[i] % 5)]):
# #             crossed = True
# #             temp[0].append(time[i])
# #             temp[1].append(vector[i])
# #         elif(crossed and vector[i] > segmentMeans[int(time[i] % 5)]):
# #             crossed = False
# #             possibleSteps[0].append(temp[0])
# #             possibleSteps[1].append(temp[1])
# #             temp = [[], []]
# #         else:
# #             temp[0].append(time[i])
# #             temp[1].append(vector[i])
#
# #     amplitudes = []
# #     for possibleStep in possibleSteps[1]:
# #         amplitudes.append(np.max(possibleStep) - np.min(possibleStep))
# #
# #     stepTimes = []
# #     for stepTimeArray in possibleSteps[0]:
# #         stepTimes.append(stepTimeArray[0])
# #
# #     steps = [[], []]
# #     THRESHOLD = 2
# #     for i in range(0, len(stepTimes)):
# #         if(amplitudes[i] > THRESHOLD):
# #             steps[0].append(stepTimes[i])
# #             steps[1].append(amplitudes[i])
#
# #     plt.figure(1)
# #     plt.plot(accY0[1], accY0[0], 'b', label='Standing')
# #     plt.plot(accY1[1], accY1[0], 'r', label='Walking')
# #     plt.plot(accY2[1], accY2[0], 'g', label='Jogging')
# #     plt.title('Y Acceleration During Activity versus Time')
# #     plt.legend()
#
# #     plt.figure(1)
# #     plt.plot(time, vector, 'b')
# #     plt.xlabel('Time')
# #     plt.ylabel('Magnitude')
# #     plt.title('Vector magnitude vs Time')
#
# #     plt.figure(2)
# #     plt.plot(STDvTime[0], STDvTime[1], 'k', label='Standard Deviation')
# #     plt.xlabel('Time')
# #     plt.ylabel('Standard Deviation')
# #     plt.title('Standard Deviation versus Time')
# #     plt.legend()
#
# #     plt.figure(3)
# #     plt.plot(stepTimes, amplitudes, 'k', label='Amplitude')
# #     plt.xlabel('Time')
# #     plt.ylabel('Amplitude')
# #     plt.title('Amplitude versus Time')
# #     plt.legend()
#
# #     plt.figure(4)
# #     plt.plot(steps[0], steps[1], 'bo', label='Steps')
# #     plt.xlabel('Time')
# #     plt.ylabel('Step Magnitude')
# #     plt.title('Steps and their Magnitudes vs Time')
# #     plt.legend()
# #     plt.show()
#
#     movingAverage = movingAvg(accZ, time, 5)
#
# #     plt.figure(1)
# #     plt.plot(time, accZ, 'k', label='Original')
# #     plt.plot(movingAverage[0], movingAverage[1], 'b--', label='Moving Average')
# #     plt.xlabel('Time')
# #     plt.ylabel('Z Acceleration')
# #     plt.title('Moving Average')
# #     plt.legend()
#
#     amplt = amplitude(movingAverage[1], movingAverage[0], 5)
#
# #     plt.figure(2)
# #     plt.plot(amplt[0], amplt[1], 'k', label='Amplitudes')
# #     plt.xlabel('Time')
# #     plt.ylabel('Z Acceleration')
# #     plt.legend()
#
#     peaks, peakTimes, valleys, valleyTimes = peaksAndValleys(movingAverage[1], movingAverage[0])
# #     print('Number of steps : %d' % stepCount(peaks, valleys, 3.6))
#
# #     plt.figure(3)
# #     plt.plot(peakTimes, peaks, 'bo', label='Peaks')
# #     plt.plot(valleyTimes, valleys, 'ro', label='Valleys')
# #     plt.xlabel('Pulse Index')
# #     plt.ylabel('Z Acceleration')
# #     plt.legend()
# #     plt.title('Peaks and Valleys')
#
# #     plt.figure(4)
# #     pulseAmplt, pulseTimes = pulseAmplitudes(movingAvg[1], movingAvg[0])
# #     plt.hist(pulseAmplt, bins=100)
# #     plt.xlabel('Amplitude')
# #     plt.ylabel('Number of Pulses')
# #     plt.title('Pulse Amplitude Histogram')
#
#     stepData = peaksAndValleys(movingAverage[1], movingAverage[0])
#     stepAmplt, pulseTimes = pulseAmplitudes(stepData)
#     timeLengths = pulseTimeLengths(stepData)[0]
#
#
#
# #     plt.figure(5)
# #     plt.plot(pulseTimeLengths, stepAmplt, 'ro')
# #     plt.title('Amplitude vs Time Length')
# #     plt.ylabel('Amplitude')
# #     plt.xlabel('Time Length')
#
#     clusterArray = np.array([timeLengths, stepAmplt])
#     clusterArray = clusterArray.T
#     centroids = cluster.k_means(clusterArray, 3)
#     cluster0 = [[], []]
#     cluster1 = [[], []]
#     cluster2 = [[], []]
#     for i in range(0, centroids[1].size):
#         if(centroids[1][i] == 0):
#             cluster0[0].append(timeLengths[i])
#             cluster0[1].append(stepAmplt[i])
#         elif(centroids[1][i] == 1):
#             cluster1[0].append(timeLengths[i])
#             cluster1[1].append(stepAmplt[i])
#         elif(centroids[1][i] == 2):
#             cluster2[0].append(timeLengths[i])
#             cluster2[1].append(stepAmplt[i])
#
# #     plt.figure(6)
# #     plt.plot(cluster0[0], cluster0[1], 'bo', label='Cluster 0')
# #     plt.plot(cluster1[0], cluster1[1], 'ro', label='Cluster 1')
# #     plt.plot(cluster2[0], cluster2[1], 'go', label='Cluster 2')
# #     plt.xlabel('Pulse Time Lengths')
# #     plt.ylabel('Pulse Amplitudes')
# #     plt.title('K Means Clustered Amplitude vs Time Length')
# #     plt.legend()
#
# #     plt.figure(7)
# #     plt.plot(pulseTimes, stepAmplt, 'bo', label='Amplitudes')
# #     plt.plot(pulseTimes, np.multiply(pulseTimeLengths, 4), 'ro', label='Pulse Time Lengths')
# #     plt.xlabel('Pulse Time')
# #     plt.ylabel('Amplitude or Time Length (Time Length * 4 for better viewing')
# #     plt.title('Amplitude and Time Length for each Pulse')
# #     plt.legend()
#
#     trainingSetInput, trainingSetOutput, testSetInput, testSetOutput = getTrainingSet()
#     trainingSetInput = np.array(trainingSetInput)
#     trainingSetOutput = np.array(trainingSetOutput)
#     testSetInput = np.array(testSetInput)
#     testSetOutput = np.array(testSetOutput)
#
#     classifier = neighbors.KNeighborsClassifier(15, weights='distance')
#     classifier.fit(trainingSetInput, trainingSetOutput)
#
#     cmap_light = ListedColormap(['#FF3333', '#11FF11', '#3333FF'])
#     cmap_bold = ListedColormap(['#FF0000', '#007022', '#0000FF'])
#
#     testClassifications = classifier.predict(testSetInput)
#
#     plt.figure(8)
#     plt.scatter(trainingSetInput[:,1], trainingSetInput[:,0], c=trainingSetOutput, cmap=cmap_light, marker='x')
#     plt.scatter(testSetInput[:,1], testSetInput[:,0], c=testClassifications, cmap=cmap_bold)
#     plt.xlim(0, trainingSetInput[:,1].max() + 0.5)
#     plt.ylim(0, trainingSetInput[:,0].max() + 0.5)
#     plt.xlabel('Time Length')
#     plt.ylabel('Amplitude')
#     plt.title('Standing, Walking, or Jogging?')
#
#     printPrecisionAndRecall(testSetOutput, testClassifications)
#
#     plt.show()
