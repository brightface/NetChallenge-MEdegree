import tensorflow as tf
import numpy as np
import matlab.engine
import matplotlib.pyplot as plt
import csv


eng = matlab.engine.start_matlab()
act = ["Walk", "Stand", "Empty", "Sit down", "Stand up"]
while 1:
    k = 1
    t = 0
    csi_trace = eng.read_bf_file('./aTl100')
    if len(csi_trace) < 500:
        continue
    ARR_FINAL = np.empty([0, 90], float)
    xx = np.empty([1, 500, 90], float)
    xx1 = np.empty([0], float)
    yy1 = np.empty([0], float)
    zz1 = np.empty([0], float)
    try:
        while (k <= 500):
            csi_entry = csi_trace[t]
            try:
                csi = eng.get_scaled_csi(csi_entry)
                A = eng.abs(csi)
                ARR_OUT = np.empty([0], float)

                ARR_OUT = np.concatenate((ARR_OUT, A[0][0]), axis=0)
                ARR_OUT = np.concatenate((ARR_OUT, A[0][1]), axis=0)
                ARR_OUT = np.concatenate((ARR_OUT, A[0][2]), axis=0)

                xx1 = np.concatenate((xx1, A[0][0]), axis = 0)
                yy1 = np.concatenate((yy1, A[0][1]), axis = 0)
                zz1 = np.concatenate((zz1, A[0][2]), axis = 0)
                ARR_FINAL = np.vstack((ARR_FINAL, ARR_OUT))
                k = k + 1
                t = t + 1
            except matlab.engine.MatlabExecutionError:
                print('MatlabExecutionError occured!!!')
                break
        xx[0] = ARR_FINAL

        outputfilename1 = "aTl101.csv"

        with open(outputfilename1, "w") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerows(xx[0])
            break
    except ValueError:
        print('ValueError occured!!!')
        continue

