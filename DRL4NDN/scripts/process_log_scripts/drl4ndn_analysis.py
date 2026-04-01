import numpy as np
import math
import scipy.stats as st
# should be imported before importing pylab or pyplot
import matplotlib
import string
from sklearn.ensemble._gradient_boosting import np_zeros
# matplotlib.use('Agg') # AGG for PNG defualt, PS for PS and EPS defualt
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_agg
import array
import sys
import cmath
import os
# import IPython
import zipfile

# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42

# -------------------------------------------------------------------------

###################################

with_processing = 1

###################################

filesdirectory = "/home/reda/PycharmProjects/DRL4NDN/logs/"
# logs-all 
# each file inside the subdirectories has the following encoding drls.log

scenario = 7
scenarios = ['static_with_one_optimal_face', 'dynamic_RTT', 'permanent_face_specific',
             'permanent_faulty_face', 'streak_disruption',
             'transient_face_specific', 'transient_faulty_face']

reward = 2
rewards = ['simple', 'streak']

state = 4
states = ['state_one', 'state_two', 'state_three', 'state_four']

agent = 4
agents = ['BestAgent', 'MAB Agents', 'CMAB Agents', 'DRL Agents']

algo = 11
algos = ['BestAgent',
         'EGreedyMultiArmedBandit', 'GaussianThompsonSampling', 'SoftmaxMultiArmedBandit', 'UCBMultiArmedBandit',
         'LinUCB',
         'PPO', 'A2C', 'DQN', 'QRDQN', 'TRPO']

# drl  = 5
# drls = ['PPO', 'A2C', 'DQN', 'QRDQN', 'TRPO']

# map = 4
# maps = ['EGreedyMultiArmedBandit', 'GaussianThompsonSampling', 'SoftmaxMultiArmedBandit', 'UCBMultiArmedBandit']

# cmab = 1
# cmabs = ['LinUCB']
seed = 100
face = 3
faces = ['3', '10', '30']

resl_array = np.empty([scenario, reward, state, algo, face, 3])  # 3 : mean_reward, mean_loss, and mean RTT
resl_array[:] = np.NAN

erro_array = np.empty([scenario, reward, state, algo, face, 3])
erro_array[:] = np.NAN

if with_processing:
    for sc in range(scenario):  #
        for re in range(reward):
            for stt in range(state):
                for al in range(algo):
                    if al == 0:  # best
                        x = 0
                    elif al >= 1 and al <= 4:  # 4 mab algorithms
                        x = 1
                    elif al == 5:  # cmab LinUCB
                        x = 2
                    else:
                        x = 3
                    if al == 0:
                        dirname = ("%s%s/%s/%s/%s/" % (
                        filesdirectory, scenarios[sc], rewards[re], states[stt], agents[x]))
                    else:
                        dirname = ("%s%s/%s/%s/%s/%s/" % (
                        filesdirectory, scenarios[sc], rewards[re], states[stt], agents[x], algos[al]))

                    if (os.path.exists(dirname)):
                        for file in os.listdir(dirname):
                            if file.endswith(".log"):
                                filename = ("%s%s" % (dirname, file))
                                f1 = open(filename, "r")
                                # print(filename)

                                # intermdiray table to get the mean rtt
                                rtt_res = np.zeros([face, seed])
                                rtt_res[:] = np.NAN
                                # intermdiray table to get the mean success rate
                                success_res = np.zeros([face, seed])
                                success_res[:] = np.NAN
                                # init face and seed to -1
                                f = -1
                                se = -1

                                # -----------------------------------------------------------------------------------------------------
                                lines = f1.readlines()
                                for line in lines:
                                    s = line.split()
                                    if (len(s) == 0):
                                        continue
                                    # print (s)
                                    if (s[0] == 'Evaluating' and len(s) == 9):  # best agent
                                        rlAlg = 'other'
                                        f = faces.index(s[5])  # to get the index of a value in a numpy table
                                        m = s[8].rstrip(']:')
                                        se = (int)(m)

                                    if (s[0] == 'Evaluating' and (len(s) == 5 or len(s) == 10)):  # mab, cmab, and drl
                                        m = s[4].lstrip('[').rstrip(']:')
                                        d = m.split('-')
                                        if (not d[0].isdigit() or not d[1].isdigit()):
                                            print("Problem : face/seed not a digit")
                                            continue
                                        f = faces.index(d[0])  # to get the index of a value in a numpy table
                                        se = (int)(d[1])
                                        # print (("face= , seed="), (f, se))
                                        rlAlg = 'other'
                                        if (len(s) == 10):
                                            rlAlg = 'drl'
                                            if (not s[8].isdigit()):
                                                print("Problem : #iterations not a digit")
                                                continue

                                            nbiter = (int)(s[8]) + 1
                                            iter = 0
                                            seed_success_array = np.zeros(nbiter)
                                            seed_success_array[:] = np.NAN

                                            seed_rtt_array = np.zeros(nbiter)
                                            seed_rtt_array[:] = np.NAN

                                    if (s[0] == 'total_interest_sent:' and len(s) == 5):
                                        if rlAlg == 'drl':
                                            seed_success_array[iter] = (float)((int)(s[4]) / (int)(s[1]))
                                            iter += 1
                                            if iter == 11:
                                                success_res[f, se] = np.nanmean(seed_success_array)
                                                iter = 0

                                        else:
                                            success_res[f, se] = (float)((int)(s[4]) / (int)(s[1]))
                                        #
                                        # print (success_res[f, se])

                                    if (s[0] == 'Average' and s[1] == 'RTT:' and len(s) == 3):
                                        if rlAlg == 'drl':
                                            seed_rtt_array[iter] = (float)(s[2])
                                            iter += 1
                                            if iter == 11:
                                                rtt_res[f, se] = np.nanmean(seed_rtt_array)
                                                iter = 0
                                        else:
                                            rtt_res[f, se] = (float)(s[2])
                                        # print (rtt_res[f, se])

                                    if (s[0] == 'Average' and s[1] == 'Reward' and len(s) == 7):
                                        resl_array[sc, re, stt, al, f, 0] = (float)(s[4])  # avg reward
                                        # if f == 0: print (f, resl_array[sc, re, stt, al, f, 0])
                                        resl_array[sc, re, stt, al, f, 1] = np.nanmean(success_res)  # averege ISR
                                        resl_array[sc, re, stt, al, f, 2] = np.nanmean(rtt_res)  # averege RTT

                                        # print (resl_array[0, 0, 0, al, 0, 1])

                                        erro_array[sc, re, stt, al, f, 0] = (float)(s[6])
                                        erro_array[sc, re, stt, al, f, 1] = np.nanstd(success_res) / cmath.sqrt(seed)
                                        erro_array[sc, re, stt, al, f, 2] = np.nanstd(rtt_res) / cmath.sqrt(seed)
                    else:
                        continue

    np.save(filesdirectory + "result_array", resl_array)
    np.save(filesdirectory + "error_array", erro_array)

else:

    resl_array = np.load("result_array.npy")
    erro_array = np.load("error_array.npy")

# print (resl_array)    
# print ('-------------------')
# print (erro_array)  

# print (resl_array[0,0,0,1:6,0,0])
# print (resl_array[0,0,0,1:6,1,0])
# print (resl_array[0,0,0,1:6,2,0])
metric = ['reward', 'ISR', 'RTT']
print(algos)
for sc in range(scenario):  #
    for re in range(reward):
        for stt in range(state):
            for fc in range(face):
                if (not math.isnan(resl_array[sc, re, stt, 0, fc, 0])):
                    print(scenarios[sc], rewards[re], states[stt], faces[fc])
                    for count in range(3):
                        print(metric[count] + ' res '), print(resl_array[sc, re, stt, :, fc, count])
                        print(metric[count] + ' std '), print(erro_array[sc, re, stt, :, fc, count])

# # ----------------------- speed + density -----------------------

# y_labels =['Average Reward', 'Average ISR (%)','Average RTT (s)']

# x_labels =['RL/DRL Algos']

# # fig_labels=['LOADng', 'LOADng-Ring', 'LOADng-Smart_Ring', 'LOADng-Trickle', 'LOADng-Trickle based Smart_Ring']

# le= ['a', 'b', 'c', 'd']

# fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize=(9 ,3)) 
# fig.tight_layout()
# # algos = [1,2,3,4,5]
# # y = [3461.41161888, 4238.07190565, 2098.10866491, 1504.20017114, 3144.82251766]
# # r1=axes[pos_x, pos_y].bar(algos, y, capsize=4)
# # IndexError: too many indices for array

# for sc in range (scenario): # 
#         for re in range (reward):
#             for stt in range (state):
#                 for fc in range (face):
#                     for count in range (3):
#                         if count == 0: pos_x = 0; pos_y = 0
#                         elif count == 1: pos_x = 0; pos_y = 1
#                         elif count == 2: pos_x = 0; pos_y = 2
#                         r1=axes[pos_x, pos_y].bar(algos, resl_array[sc,re,stt,:,fc,count], xerr=None, yerr=erro_array[sc, re, stt,:, fc, count], capsize=4) 
#                         # print(resl_array[0,0,0,1:6,fc,count])
#                         # print(erro_array[0,0,0,1:6,fc,count])
#                         # r1=axes[pos_x, pos_y].bar(algos, y) 
#                         axes[pos_x, pos_y].set_xlabel(x_labels[0], fontsize=8, labelpad=0.3)
#                         axes[pos_x, pos_y].set_ylabel(y_labels[count], fontsize=8, labelpad=0.3)

#                         [x.set_fontsize(6) for x in axes[pos_x, pos_y].xaxis.get_ticklabels()]
#                         axes[pos_x, pos_y].xaxis.set_label_coords(0.5, -0.10)

#                         [y.set_fontsize(6) for y in axes[pos_x, pos_y].yaxis.get_ticklabels()]
#                         axes[pos_x, pos_y].yaxis.set_label_coords(-0.12, 0.5)


#                         axes[pos_x, pos_y].set_ylim(0, 1.1*(axes[pos_x, pos_y].get_ylim()[1]))
#                         axes[pos_x, pos_y].set_title(le[count], fontsize=10)

#                     fig.subplots_adjust(hspace = 0.35, wspace = 0.3)
#                     figname=("%s%s-%s-%s-%s.png"%(filesdirectory,scenarios[sc], rewards[re], states[stt], faces[fc]))

#                     plt.savefig(figname, format='png', bbox_inches='tight', pad_inches=0.05, dpi=400)
#                     # plt.savefig(figname, format='eps', bbox_inches='tight', pad_inches=0.05)
#                     # plt.savefig(figname, format='pdf', bbox_inches='tight', pad_inches=0.05)
#                     # plt.clf() 


# # leg=plt.figlegend((r1, r2, r3, r4, r5),fig_labels, loc= (0, 0.95), ncol= 5, fontsize=8, borderaxespad=0.08, labelspacing=0.0, columnspacing = 3, handlelength=3, handleheight=1.2, borderpad=.2 )
# # #    leg=axes[pos_x, pos_y].legend((r1, r2, r3, r4),fig_labels, loc= 0, fontsize=8, borderaxes[pos_x, pos_y]pad=0.08, labelspacing=0.0, columnspacing =3, handlelength=3, handleheight=1.2, borderpad=.5 )
