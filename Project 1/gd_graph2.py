import tensorflow as tf
import csv
import numpy as np
import matplotlib.pyplot as plt

rateList = []
stepList = []
f = open('3. Gradient Descent/gd_graph1.csv', 'r', encoding='utf-8', newline='')
data = list(csv.reader(f))

rateList = data[0]
stepList=[]

for i in range(0,50):
    temp = 0
    for j in range(0,50):
        temp += float(data[1][i*50+j])
    stepList.append(temp/50)

refinedStepList = stepList[4:-5]
refinedLateList = rateList[4:-5]

f = open('3. Gradient Descent/gd_graph2.csv', 'w', encoding='utf-8', newline='')
r = csv.writer(f)
r.writerow(refinedLateList)
r.writerow(refinedStepList)
f.close()

plt.figure()
plt.grid()
plt.plot(refinedLateList,refinedStepList,stepList)
plt.show()
