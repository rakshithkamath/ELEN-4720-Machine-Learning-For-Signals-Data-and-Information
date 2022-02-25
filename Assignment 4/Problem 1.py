import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs

TeamNames = pd.read_csv("TeamNames.txt",header = None)
ScoreList = pd.read_csv("CFB2019_scores.csv",header=None, names=['Team A index','Team A points', 'Team B index','Team B points'])
print(TeamNames.shape,ScoreList.shape)

#create the matrix M
M =np.zeros((TeamNames.shape[0],TeamNames.shape[0]))
print(M.shape)

#Calculating the updates to M based on the formula mentioned in the pdf
for i in range(len(ScoreList)):
    team_i,points_i,team_j,points_j=ScoreList.iloc[i]
    M[team_i-1,team_i-1]+=np.array([(points_i > points_j)]).astype('uint8')+points_i/(points_i+points_j)
    M[team_j-1,team_j-1]+=np.array([(points_i < points_j)]).astype('uint8')+points_j/(points_i+points_j)
    M[team_i-1,team_j-1]+=np.array([(points_i < points_j)]).astype('uint8')+points_j/(points_i+points_j)
    M[team_j-1,team_i-1]+=np.array([(points_i > points_j)]).astype('uint8')+points_i/(points_i+points_j)
print(M[1,:])

#Normalizing the rows of M
for i in range(TeamNames.shape[0]):
    M[i]=M[i]/sum(M[i])
print(sum(M[1,:]))

#Declaring w0 to be uniform distibution
w0= np.ones(TeamNames.shape[0])/TeamNames.shape[0]

T_list=[10,100,1000,10000]
w_t_list =[]
w_t = w0
for t in range(T_list[-1]+1):
    w_t = np.dot(w_t,M)
    if t in T_list:
        w_t_list.append(w_t)

#w_t_list is an array of 769,4 

#sort the elements based on their score and store it along with the score 
rank=np.argsort(w_t_list[0])
top25=rank[-25:]
top25_names=TeamNames.iloc[top25].reset_index(drop=True)
top25_scores=np.around(w_t_list[0][top25],5)
print(top25_names)
print(top25_scores)
final_list=pd.DataFrame({'College':top25_names,'Scores':top25_scores})
final_list.to_excel('TeamRanks.xlsx',sheet_name=0)

