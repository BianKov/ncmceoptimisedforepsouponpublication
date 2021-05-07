#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import networkx as nx
import numpy as np
import embedding as em


savePath=os.getcwd()+"/exampleGraph/" #the path of the directory which contains the edge list; all the results will be saved here
edgeListFileName="edgeListExample.txt" #the name of the text file containing the edge list
skipRows=0 #the number of lines to be skipped at the beginning of the text file containing the edge list
delimiter="\t" #the string used to separate the columns in the text file to be loaded
#startingm,startingBeta,startingT: the initial estimations of the 1<=m, 0.1<beta<0.99 and 0.1<T<0.99 model parameters. The default is None, meaning that initially m will be set to the half of the average degree, while beta and T will be set to 0.5
startingm=None #use the default setting
startingBeta=None #use the default setting
startingT=None  #use the default setting
initStepSizeFactor=0.2 #the size of the first step will be initStepSizeFactor times the initial distance from the side of the allowed interval
precision=0.01 #the optimization stops, if the norm of the next displacement vector in the parameter space would be smaller than this value
maxIters=100 #the maximum number of steps to be carried out in the parameter space
maxm=None #maxm is the allowed maximum of the parameter m, determining the average degree of the network together with the parameter L via the formula <k>=2*(m+L). The default is used in the case of None, which is the average degree.
weightingType="RA1" #the name of the pre-weighting rule to be applied; it can be set to "RA1", "RA2", "RA3" or "RA4" for the repulsion-attraction rules or "EBC" for the edge betweenness centrality rule
CAorEA="EA" #after determining the network node coordinates in the reduced space circular ("CA") or equidistant adjustment ("EA") can be applied to get the angular coordinates on the Poincare disk
numberOfSwappingRounds=5 #the number of correction rounds in which the range of the tested angular coordinates is restricted for each node between its current second angular neighbors
numberOfAnglesInSwappingRounds=6 #the number of each node's tested angular positions in the swapping rounds
numberOfNonswappingRounds=3 #the number of correction rounds in which the range of the tested angular coordinates is restricted for each node between its current first angular neighbors
numberOfAnglesInNonswappingRounds=5 #the number of each node's tested angular positions in the non-swapping rounds

#####################################################################


#load the graph to be embedded and create a NetworkX graph from it
G=em.loadGraph(savePath+edgeListFileName,savePath,skipRows,delimiter)
N = len(G)
nodeList = list(G.nodes)

#determine the corresponding PSO model parameters
[m,beta,T,R] = em.parameterEstimation_logloss(G, N, weightingType, CAorEA, savePath, startingm, startingBeta, startingT, initStepSizeFactor, precision, maxIters, maxm) #in the file named paramsInEachStep_... the columns are the following: m, beta, T, logarithmic loss


#determine the radial coordinates of the nodes on the native disk representation of the hyperbolic plane of curvature K=-1
radialNodeOrder = em.determineRadialOrder(G,savePath)
radialCoords = em.determine_r(N,beta,radialNodeOrder) ##radialCoords[i] is the radial coordinate of the node that is the ith according to the order in G.nodes


#determine the angular coordinates of the network nodes on the native disk representation of the hyperbolic plane with the ncMCE method
angularCoords_ncMCE=em.ncMCE(G,N,weightingType,CAorEA)
em.saveCoordinates(nodeList,radialCoords,angularCoords_ncMCE,savePath+"embeddedCoordinates_ncMCE.txt") #the columns: nodeID   radialCoord  angularCoord


#perform correction of the angular coordinates based on logarithmic loss minization
[angularCoords_ncMCE_corr,LL_eachRound]=em.angleCorrection(G, N, T, R, radialCoords, angularCoords_ncMCE, numberOfSwappingRounds, numberOfAnglesInSwappingRounds, numberOfNonswappingRounds, numberOfAnglesInNonswappingRounds)
em.saveCoordinates(nodeList,radialCoords,angularCoords_ncMCE_corr,savePath+"embeddedCoordinates_ncMCEcorr.txt") #the columns: nodeID   radialCoord  angularCoord
np.savetxt(savePath+"logLoss_eachAngularCorrectionRound.txt",LL_eachRound,delimiter='\t') #save the logarithmic losses after each round of the correction


#plot the embedded network in the native representation of the hyperbolic plane
em.PoincDisk(savePath+'layoutOnNativeDisk_ncMCE.pdf', nodeList, list(G.edges), N, radialCoords, angularCoords_ncMCE, [0 for i in range(N)], max(radialCoords))
em.PoincDisk(savePath+'layoutOnNativeDisk_ncMCEcorr.pdf', nodeList, list(G.edges), N, radialCoords, angularCoords_ncMCE_corr, [0 for i in range(N)], max(radialCoords))


#weight the edges according to hyperbolic distances (promoting community detection)
em.weighting_hypDist(G,radialCoords,angularCoords_ncMCE_corr)
fileHandler = open(savePath+"HypDistWeightedEdgeList.txt","w") #save the weighted edge list
fileHandler.write("nodeID1\tnodeID2\tweight\n")
for e in G.edges.data("weight"):
    fileHandler.write(e[0]+"\t"+e[1]+"\t"+str(e[2])+"\n")
fileHandler.close()

