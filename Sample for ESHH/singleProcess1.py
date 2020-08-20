#!/usr/bin/python
# -*- coding: utf-8 -*-
from scipy.optimize import differential_evolution
import math
import datetime
import numpy as np
import matplotlib.pyplot as plt
import runoff1
import os
import pandas as pd
np.seterr(invalid='ignore')
np.set_printoptions(suppress=True, precision=3,
                    linewidth=100)  
__log = False
# observation data
Qd = np.array([0.06,0.417,0.458, 0.497, 0.362, 0.213, 0.179, 0.136, 0.106, 0.083,
               0.069, 0.05])
Isotop = np.array([-9.13, -10.04, -9.81, -9.47, -8.99, -8.65, -8.77,-8.72,
                   -8.78, -8.97, -8.97, -8.97])
Qi = np.array([0.0,0.16, 0.224, 0.344, 0.157, 0.111, 0.089, 0.072, 0.062,
               0.0575, 0.0547, 0.04])
Isotopi = np.array([-8.97, -9.91, -9.30, -8.9, -8.78, -8.79, -8.77,-8.61,
                   -8.61, -8.73, -8.74, -8.74])
SW1 = np.array([0.314, 0.34, 0.335, 0.33, 0.322, 0.321, 0.32, 0.32,
                0.319, 0.319, 0.319, 0.319])
SW2 = np.array([0.379, 0.40, 0.415, 0.41, 0.405, 0.401, 0.4, 0.399,
               0.398,0.397, 0.397, 0.396])
concs1t=np.array([-9.36,-11.41,-11.67,-11.65,-11.86])
concs2t=np.array([-10.85,-11.32,-11.34,-11.32,-11.32])
def nse(simulation_s, evaluation):
    nse_ = 1 - (np.sum((evaluation - simulation_s) ** 2, dtype=np.float64) /
                np.sum((evaluation - np.mean(evaluation)) ** 2, dtype=np.float64))
    return nse_
def kge(simulation_s, evaluation):
    sim_mean = np.mean(simulation_s)
    obs_mean = np.mean(evaluation)
    r = np.sum((simulation_s - sim_mean) * (evaluation - obs_mean)) / \
        np.sqrt(np.sum((simulation_s - sim_mean) ** 2) *
                np.sum((evaluation - obs_mean) ** 2))
    alpha = np.std(simulation_s) / np.std(evaluation)
    beta = np.sum(simulation_s) / np.sum(evaluation)
    kge_ = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return kge_
def eval__(params):
    R_ = runoff1.eval(params)
    sim_Qd = R_[0]
    sim_Isotop = R_[1]
    sim_Qi = R_[2]
    sim_Isotopi = R_[3]
    sim_Moisture1 = R_[4]
    sim_Moisture2 = R_[5]
    sim_PreF1 = R_[6]
    sim_PisF1 = R_[7]
    sim_PreF2 = R_[8]
    sim_PisF2 = R_[9]
    sim_concs1m = R_[10]
    sim_concs1im= R_[11]
    sim_concs1t = R_[12]
    sim_concs2m = R_[13]
    sim_concs2im = R_[14]
    sim_concs2t = R_[15]
    sim_Error = R_[16]
    if np.any(sim_Moisture1 < 0.01):
        f1 = 10000.0 + np.sum(np.square(1000-sim_Moisture1))
        f2 = 10000.0
    else:
        f2 = np.sum((sim_Qd - Qd)**2)+np.sum((sim_Qi - Qi)**2)+np.sum((sim_Isotop - Isotop)**2)+np.sum((sim_Isotopi - Isotopi)**2)+np.sum(np.square(sim_concs1t[0] - concs1t[0]))++np.sum(np.square(sim_concs1t[3] - concs1t[1]))+np.sum(np.square(sim_concs1t[4] - concs1t[2]))+np.sum(np.square(sim_concs1t[6] - concs1t[3]))+np.sum(np.square(sim_concs1t[8] - concs1t[4]))+np.sum(np.square(sim_concs2t[0] - concs2t[0]))++np.sum(np.square(sim_concs2t[3] - concs2t[1]))+np.sum(np.square(sim_concs2t[4] - concs2t[2]))+np.sum(np.square(sim_concs2t[6] - concs2t[3]))+np.sum(np.square(sim_concs2t[6] - concs2t[4]))+100*np.sum((sim_Moisture2 - SW2)**2)+100*np.sum((sim_Moisture1 - SW1)**2)+sim_Error[11]**2
    return f2
    
def calibrate(run_counts):
    bounds = [(0.01, 1), (0.01, 1), (0.01, 1), (0.01, 1), (0, 1), 
              (0.0, 1), (0.0, 1), (0.0, 1), (0.0, 100), (0.0, 100), (0.0, 1),(0.0, 1),(0.0, 1)]

    __log = False
    starttime = datetime.datetime.now()
    nondominated_solutions = np.zeros([run_counts, len(bounds)], dtype=float)
    
    for iTries in range(run_counts):
        try:
            print(iTries)
            result = differential_evolution(
                eval__, bounds, strategy='randtobest1bin', popsize=50,  disp=False, updating='deferred', workers=4)  
        except:
            result = differential_evolution(
                eval__, bounds, strategy='randtobest1bin', popsize=50,  disp=True)  
        nondominated_solutions[iTries, :] = np.array(result.x)
    endtime = datetime.datetime.now()
    print("calibration is finished, time is" + str((endtime - starttime).seconds) + " s")
    print(result)
    # plot
    fig = plt.figure(figsize=(8, 6))
    params_x = result.x
    print('=======================')
    print(params_x)
    R_ = runoff1.eval0(params_x)
    sim_Qd = R_[0]
    sim_Isotop = R_[1]
    sim_Qi = R_[2]
    sim_Isotopi = R_[3]
    sim_Moisture1 = R_[4]
    sim_Moisture2 = R_[5]
    sim_PreF1 = R_[6]
    sim_PisF1 = R_[7]
    sim_PreF2 = R_[8]
    sim_PisF2 = R_[9]
    sim_concs1m = R_[10]
    sim_concs1im= R_[11]
    sim_concs1t = R_[12]
    sim_concs2m = R_[13]
    sim_concs2im = R_[14]
    sim_concs2t = R_[15]
    sim_Error = R_[16]
    ax = fig.add_subplot(8, 2, 1)
    ax.plot(sim_Qd)
    ax.plot(Qd,'r.')
    ax.set_title('Runoffg')

    ax = fig.add_subplot(8, 2, 2)
    ax.plot(sim_Isotop)
    ax.plot(Isotop,'r.')
    ax.set_title('Isotopeg')
    
    ax = fig.add_subplot(8, 2, 3)
    ax.plot(sim_Qi)
    ax.plot(Qi,'r.')
    ax.set_title('Runoffi')

    ax = fig.add_subplot(8, 2, 4)
    ax.plot(sim_Isotopi)
    ax.plot(Isotopi,'r.')
    ax.set_title('Isotopei')
    
    ax = fig.add_subplot(8, 2, 5)
    ax.plot(sim_Moisture1)
    ax.plot(SW1,'r.')
    ax.set_title('Moisture1')

    ax = fig.add_subplot(8, 2, 6)
    ax.plot(sim_Moisture2)
    ax.plot(SW2,'r.')
    ax.set_title('Moisture2')
    
    ax = fig.add_subplot(8, 2, 7)
    ax.plot(sim_PreF1)
    ax.set_title('PreF1')

    ax = fig.add_subplot(8, 2, 8)
    ax.plot(sim_PisF1)
    ax.set_title('PisF1')

    ax = fig.add_subplot(8, 2, 9)
    ax.plot(sim_PreF2)
    ax.set_title('PreF2')

    ax = fig.add_subplot(8, 2, 10)
    ax.plot(sim_PisF2)
    ax.set_title('PisF2')
    
    ax = fig.add_subplot(8, 2, 11)
    ax.plot(sim_concs1m)
    ax.set_title('mobile sw1')

    ax = fig.add_subplot(8, 2, 12)
    ax.plot(sim_concs1im)
    ax.set_title('immobile sw1')

    ax = fig.add_subplot(8, 2, 13)
    ax.plot(sim_concs1t)
    ax.scatter(0,concs1t[0],s=15, c="#ff1212", marker='o')
    ax.scatter(3,concs1t[1],s=15, c="#ff1212", marker='o')
    ax.scatter(4,concs1t[2],s=15, c="#ff1212", marker='o')
    ax.scatter(6,concs1t[3],s=15, c="#ff1212", marker='o')
    ax.scatter(8,concs1t[4],s=15, c="#ff1212", marker='o')
    ax.set_title('total sw1')

    ax = fig.add_subplot(8, 2, 14)
    ax.plot(sim_concs2m)
    ax.set_title('mobile sw2')

    ax = fig.add_subplot(8, 2, 15)
    ax.plot(sim_concs2im)
    ax.set_title('immobile sw2')

    ax = fig.add_subplot(8, 2, 16)
    ax.plot(sim_concs2t)
    ax.scatter(0,concs2t[0],s=15, c="#ff1212", marker='o')
    ax.scatter(3,concs2t[1],s=15, c="#ff1212", marker='o')
    ax.scatter(4,concs2t[2],s=15, c="#ff1212", marker='o')
    ax.scatter(6,concs2t[3],s=15, c="#ff1212", marker='o')
    ax.scatter(8,concs2t[4],s=15, c="#ff1212", marker='o')
    ax.set_title('total sw2')
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                        wspace=0.2, hspace=0.3)
    plt.show()

    print('Qd',sim_Qd)
    print('Qdi',sim_Isotop)
    print('Qi',sim_Qi)
    print('Qii',sim_Isotopi)
    print('SW1',sim_Moisture1)
    print('SW2',sim_Moisture2)
    print('SW1i',sim_concs1t)
    print('SW2i',sim_concs2t)
    print('PreF1',sim_PreF1)
    print('PisF1',sim_PisF1)
    print('PreF2',sim_PreF2)
    print('PisF2',sim_PisF2)
    print('m1',sim_concs1m)
    print('im1',sim_concs1im)
    print('m2',sim_concs2m)
    print('im2',sim_concs2im)
        
    nseuf=nse(sim_Qd,Qd)
    print('nseuf=',nseuf)
    nseiuf=nse(sim_Isotop,Isotop)
    print('nseiuf=',nseiuf)
    nseif=nse(sim_Qi,Qi)
    print('nseif=',nseif)
    nseiif=nse(sim_Isotopi,Isotopi)
    print('nseiif=',nseiif)
    kgeuf=kge(sim_Qd,Qd)
    print('kgeuf=',kgeuf)
    kgeiuf=kge(sim_Isotop,Isotop)
    print('kgeiuf=',kgeiuf)
    kgeif=kge(sim_Qi,Qi)
    print('kgeif=',kgeif)    
    kgeiif=kge(sim_Isotopi,Isotopi)
    print('kgeiif=',kgeiif) 
    nsesw1=nse(sim_Moisture1,SW1)
    print('nsesw1=',nsesw1) 
    kgesw1=kge(sim_Moisture1,SW1)
    print('kgesw1=',kgesw1) 
    nsesw2=nse(sim_Moisture2,SW2)
    print('nsesw2=',nsesw2) 
    kgesw2=kge(sim_Moisture2,SW2)
    print('kgesw2=',kgesw2) 
    print('Error=',sim_Error) 
    num_bins = 13
    fig = plt.figure(figsize=(8, 6))
    (r, c) = nondominated_solutions.shape
    i = 0
    for i in range(c):
        a = nondominated_solutions[:, i]
        ax = fig.add_subplot(4, 4, i+1)
        ax.hist(a, num_bins, facecolor='blue', alpha=0.5)
                
        for iTries in range(run_counts):
            X = pd.DataFrame(nondominated_solutions)
            file = os.getcwd() + '\\7222.csv'
             
            X.to_csv(file, index=False)
        
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                        wspace=0.25, hspace=0.25)
    plt.show()

if __name__ == "__main__":
    run_counts = 500
    calibrate(run_counts)

