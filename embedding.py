#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import networkx as nx
import math
import scipy.special as spec
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import patches

###LOAD THE GRAPH###

#A function for loading the edge list of the undirected network to be embedded. Edge weights are disregarded, self-loops are removed, multi-edges are converted to single edges, and only the largest connected component is returned as a NetworkX Graph.
#loadPath is a string with the path of the text file containing the edge list to be loaded
    #In the edge list each line has to correspond to one connected node pair with the identifiers of the connected two nodes in the first and the second column. Additional columns will be disregarded, each edge will have weight 1.
#savePath is a string with the path of the text file where the edge list of the embedded network will be saved
    #In the saved edge list each line corresponds to one connected node pair with the identifiers of the connected two nodes in the first and the second column. The columns are separated by tabs.
#skipRows is the number of lines to be skipped at the beginning of the text file containing the edge list; the default is 0
#delimiter is the string used to separate the columns in the text file to be loaded; the default is "\t"
#Example for function call:
#   G=embedding.loadGraph(os.getcwd()+"/"+directoryName+"/edgeList.txt",os.getcwd()+"/"+directoryName+"/",1,"\t")
def loadGraph(loadPath,savePath,skipRows=0,delimiter="\t"):
    edgeList = [] #initialize the list of the (node1 identifier,node2 identifier) edge tuples
    fileHandler = open(loadPath,"r")
    for l in range(skipRows):
        line = fileHandler.readline()
    while True:
        line = fileHandler.readline() #get the next line from file
        if not line: #line is empty (end of the file)
            break;
        listOfWords = line.split(delimiter)
        sourceNodeID = listOfWords[0] #string from the first column as the identifier of one of the connected nodes
        if listOfWords[1][-1]=="\n": #the second column is the last in the currently loaded line
            targetNodeID = listOfWords[1][:-1] #string from the second column without "\n" as the identifier of the other node
        else: #there are more than two columns in the currently loaded line
            targetNodeID = listOfWords[1] #string from the second column as the identifier of the other node
        if sourceNodeID != targetNodeID: #the self-loops are disregarded
            edgeList.append((sourceNodeID,targetNodeID))
    fileHandler.close()
    edgeList_singleEdges = list(set(edgeList)) #each multi-edge is converted to a single edge

    G_total = nx.Graph()
    G_total.add_edges_from(edgeList_singleEdges)
    #extract the largest connected component (the embedding is defined only for connected graphs)
    G=max([G_total.subgraph(comp).copy() for comp in nx.connected_components(G_total)],key=len) #.copy(): create a subgraph with its own copy of the edge/node attributes -> changes to attributes in the subgraph are NOT reflected in the original graph; without copy the subgraph is a frozen graph for which edges can not be added or removed
    nx.write_edgelist(G,savePath+"embeddedEdgeList.txt",delimiter='\t',data=False)
    return G





###Determine the popularity fading parameter beta from the exponent of the degree distribution###

#A function for calculating the complementary cumulative distribution function (CCDF(k)=P(k<=K)) of the node degrees listed in degreeList.
def CCDFcalculation(degreeList):
    minDegree=int(np.amin(degreeList))
    maxDegree=int(np.amax(degreeList))
    hist,bins=np.histogram(degreeList,bins=range(minDegree,maxDegree+2),density=True)
            #bins is the array [minDegree,minDegree+1,...,maxdegree+1]
            #Note that the degree is an integer for every node, therefore the bins into which the nodes have been classified are the following:
                #[minDegree,minDegree+1)=minDegree, ..., [maxdegree-1,maxdegree)=maxdegree-1 and [maxdegree,maxdegree+1]=maxdegree, since there is no node with degree maxdegree+1
    degreeAxis=bins[:-1] #=[minDegree,minDegree+1,...,maxdegree]=all the possible degree values between the observed minimum and maximum
    CCDF=np.cumsum(hist[::-1])[::-1]
    return [degreeAxis,CCDF]

#A function for determining the degree decay exponent gamma for a given degree distribution.
#degreeList contains the degree of each node in the examined graph
#CCDF lists the probability P(k<=K) for all k degree value in degreeAxis
#minNumOfSamples is an int corresponding to the minimum number of degree values to be taken into consideration during curve fitting
#savePath is the string containing the path of the saved figures
def HillPlot(degreeList,degreeAxis,CCDF,minNumOfSamples,savePath):
    #The degree distribution is assumed to be of the form P(k)=constant*k^-gamma at a range k_min<=k.
    #CCDF(k)=P(k<=K)=[constant/(gamma-1)]*k^-(gamma-1) -> ln(P(k<=K))=ln(constant/(gamma-1))-(gamma-1)*ln(k)
    sortedListOfOccurringDegreeValues = sorted(set(degreeList)) #ascending order
    numOfDifferentOccurringDegreeValues = len(sortedListOfOccurringDegreeValues) #the number of different occurring degrees
    gammaValues=[]
    constantValues=[]
    DValues=[] #Kolmogorov-Smirnov statistic: the distance between the measured and the fitted distributions (both are renormalized!)
    for kminIndex in range(numOfDifferentOccurringDegreeValues):
        kmin=sortedListOfOccurringDegreeValues[kminIndex]
        properDegreeList=degreeList[kmin<=degreeList] #list with the degree of those nodes which has a degree>=kmin
        n=len(properDegreeList) #the number of network nodes contributing to the curve fitting
        if minNumOfSamples<=n: #once n become smaller than minNumOfSamples, do nothing
            gammaValues.append(1+n/np.sum(np.log(properDegreeList/(kmin-0.5))))
            kminIndexInTotalDegreelist=np.where(degreeAxis==kmin)[0][0]
            degreesInTail=degreeAxis[kminIndexInTotalDegreelist:]
            constantValues.append(math.exp(np.mean(np.log(CCDF[kminIndexInTotalDegreelist:])+(gammaValues[kminIndex]-1)*np.log(degreesInTail)))*(gammaValues[kminIndex]-1))
            DValues.append(np.amax(np.absolute((CCDF[kminIndexInTotalDegreelist:]/(spec.zeta(gammaValues[kminIndex],kmin)*constantValues[kminIndex]))-((1/((gammaValues[kminIndex]-1)*spec.zeta(gammaValues[kminIndex],kmin)))*(degreesInTail**(1-gammaValues[kminIndex]))))))
    kminNumber=len(gammaValues) #the number of tested limiting degree values
    best_kminIndex=np.argmin(DValues)
    best_kmin=sortedListOfOccurringDegreeValues[best_kminIndex]
    best_D=DValues[best_kminIndex]
    best_gamma=gammaValues[best_kminIndex]
    best_constant=constantValues[best_kminIndex]

    #create the Hill plot
    fig = plt.figure(figsize=(30,8))
    ax = plt.subplot(1,2,1)
    plt.plot(sortedListOfOccurringDegreeValues[:kminNumber],gammaValues,'b.')
    plt.plot(best_kmin,best_gamma,'r^',markersize=8)
    plt.title("Hill plot",fontsize=24,y=1.01)
    plt.legend([r'$\hat{\gamma}(k_{\mathrm{min}})$',r'$\hat{\gamma}_{\mathrm{best}}='+("%.2f" % best_gamma)+r'$ at $k_{\mathrm{min}}='+("%.0f" % best_kmin)+'$'],fontsize=22,loc='best')
    plt.xlabel(r'$k_{\mathrm{min}}$',fontsize=26)
    plt.ylabel(r'$\gamma$',fontsize=26)
    plt.xlim((sortedListOfOccurringDegreeValues[0]-0.1,sortedListOfOccurringDegreeValues[kminNumber-1]+0.1))
    plt.ylim((min(gammaValues)-0.1,max(gammaValues)+0.1))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.tick_params(length=8,width=2)
    #plot the Kolmogorov-Smirnov statistic
    ax = plt.subplot(1,2,2)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.yaxis.set_ticks_position('both')
    plt.plot(sortedListOfOccurringDegreeValues[:kminNumber],DValues,'b.')
    plt.plot(best_kmin,best_D,'r^',markersize=8)
    plt.title("Kolmogorov-Smirnov statistic",fontsize=24,y=1.01)
    plt.legend([r'$D=\mathrm{max}|\mathrm{CCDF}^{\mathrm{renormalized}}_{\mathrm{measured}}(k)-\mathrm{CCDF}^{\mathrm{renormalized}}_{\mathrm{fitted}}(k)|$ for $k_{\mathrm{min}}\leq k$',r'$D_{\mathrm{min}}='+("%.3f" % best_D)+r'$ at $k_{\mathrm{min}}='+("%.0f" % best_kmin)+r'$'],fontsize=22,loc='best')
    plt.xlabel(r'$k_{\mathrm{min}}$',fontsize=26)
    plt.ylabel(r'$D$',fontsize=26,rotation=270,labelpad=30)
    plt.xlim((sortedListOfOccurringDegreeValues[0]-0.1,sortedListOfOccurringDegreeValues[kminNumber-1]+0.1))
    #plt.ylim((min(DValues)-0.05,max(DValues)+0.05))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.tick_params(length=8,width=2)
    plt.subplots_adjust(wspace=0.05)
    fig.savefig(savePath+'HillKS.png',bbox_inches="tight")
    plt.close(fig)
    #plot the CCDF
    fig = plt.figure(figsize=(15,8))
    plt.loglog(degreeAxis,CCDF,'b-',degreeAxis[best_kmin<=degreeAxis],(best_constant/(best_gamma-1))*np.power(degreeAxis[best_kmin<=degreeAxis],-(best_gamma-1)),'r-')
    plt.xlim((int(np.amin(degreeList))-1.1,int(np.amax(degreeList))+5))
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    ax.tick_params(length=8,width=2)
    plt.legend([r'$\mathrm{CCDF}_{\mathrm{measured}}(k)$',r'$\mathrm{CCDF}_{\mathrm{fitted}}(k)='+("%.2f" % best_constant)+r'\cdot k^{-('+("%.2f" % best_gamma)+'-1)}$'],fontsize=22,loc='best')
    plt.xlabel(r'$k$',fontsize=30)
    plt.ylabel(r'$\mathcal{P}(k\leq K)$',fontsize=30)
    plt.title("complementary cumulative distribution function",fontsize=24,y=1.01)
    fig.savefig(savePath+'CCDF.png',bbox_inches="tight")
    plt.close(fig)

    return best_gamma

#A function for determining the absolute value of the exponent of the degree distribution's tail (P(k)~k^(-gamma)) for a given graph.
#G is the NetworkX Graph to be embedded
#savePath is the string containing the path of the saved figures
#minNumOfSamples is an int corresponding to the minimum number of degree values to be taken into consideration during curve fitting; the default is 50
#Example for function call:
#   gamma=embedding.fitGamma(G,os.getcwd()+'/'+directoryName+'/')
def fitGamma(G,savePath,minNumOfSamples=50):
    #create the complementary cumulative distribution function of the degrees (CCDF(k)=P(k<=K))
    degreeArray = np.array([Deg for (nodeID,Deg) in G.degree],dtype=int) #array with the degree of all nodes
    [degreeAxis,CCDF] = CCDFcalculation(degreeArray)
    #fit a power-law to the complementary cumulative distribution function of the in-degrees (CCDF_in(k)=P(k<=K_in))
    gamma = HillPlot(degreeArray,degreeAxis,CCDF,minNumOfSamples,savePath)
    return gamma





###Determine the parameters needed for the logarithmic loss calculation via logarithmic loss minimization###

#A function for creating an initial embedding that can be used for estimating the model parameters.
#G is the NetworkX graph to be embedded
#N is the number of nodes in the graph to be embedded
#weightingType is a string corresponding to the name of the pre-weighting rule to be used, it can be set to 'RA1', 'RA2', 'RA3', 'RA4', 'EBC' or, if no pre-weighting is needed, to None
#CAorEA is a string that can be set to 'CA' or to 'EA', determining whether to use circular adjustment or equidistant adjustment to create angular coordinates on the native disk representation of the hyperbolic plane from the node coordinates obtained in the reduced space
def initialEmbeddingForParameterEstimation(G,N,weightingType,CAorEA):
    #determine angular distances between the nodes
    angularCoords_ncMCE = ncMCE(G,N,weightingType,CAorEA) #the angular coordinates without correction
    angularDist_dict = {}
    for i in range(1,N):
        for j in range(i):
            angularDist_dict[(i,j)]=math.pi-math.fabs(math.pi-math.fabs(angularCoords_ncMCE[i]-angularCoords_ncMCE[j])) #angular distance between the ith and the jth node according to the order in G.nodes
            angularDist_dict[(j,i)]=angularDist_dict[(i,j)]
    #determine a radial order of the nodes: descending order of the node degrees with ties broken randomly
    nodeANDdegree = dict(G.degree()) #a dictionary with nodes as keys and degree as values
    degreeArray = np.array([Deg for (nodeID,Deg) in G.degree],dtype=int) #array with the degree of all nodes
    randArray = np.random.random(degreeArray.size)
    radialNodeOrder = np.lexsort((randArray,degreeArray))[::-1] #node indices corresponding to the order in G.nodes sorted in a descending order of the degree
    ID_radialID_dict = {} #key=nodeID according to the order in G.nodes, value=nodeID according to the radial order of the nodes
    radNodeID=0
    for nodeID in radialNodeOrder:
        ID_radialID_dict[nodeID]=radNodeID
        radNodeID=radNodeID+1
    return [angularCoords_ncMCE,angularDist_dict,radialNodeOrder,ID_radialID_dict,nodeANDdegree]



#A function for calculating the hyperbolic distance h between two nodes having polar coordinates [r1,Theta1] and [r2,Theta2]
def hypDist(r1,Theta1,r2,Theta2):
    cos_dTheta = math.cos(math.pi-math.fabs(math.pi-math.fabs(Theta1-Theta2)))  # cosine of the angular distance between the two nodes
    if cos_dTheta==1:  # in this case the hyperbolic distance between the two nodes is h=acosh(cosh(r1-r2))=|r1-r2|
        h = math.fabs(r1-r2)
    else:
        argument_of_acosh = math.cosh(r1)*math.cosh(r2)-math.sinh(r1)*math.sinh(r2)*cos_dTheta
        if argument_of_acosh<1:  #a rounding error occurred, because the hyperbolic distance is close to zero
            print("The argument of acosh is "+str(argument_of_acosh)+", less than 1.\nr1="+str(r1)+"\nr2="+str(r2)+"\nTheta1="+str(Theta1)+"\nTheta2="+str(Theta2))
            h = 0
        else:
            h = math.acosh(argument_of_acosh)
    return h



#A function for calculating the -logarithm of the likelihood for a given edge list and node coordinates.
#T is the temperature used for the embedding
#N is the number of nodes in the graph to be embedded
#R is the cutoff distance of the connection probability
#nodeList is a list of the node identifiers
#edgeList is a list of tuples of node identifiers corresponding to connected node pairs
#radialCoords is a NumPy array containing the radial coordinates of the nodes, where the ith element is the coordinate of the ith node in nodeList
#angularCoords is a NumPy array containing the angular coordinates of the nodes, where the ith element is the coordinate of the ith node in nodeList
def calculateLikelihood(T,N,R,nodeList,edgeList,radialCoords,angularCoords):
    if T==0: #for T=0 the function returns the likelihood
        L2 = 1 #likelihood!
        for i in range(1, N):
            for j in range(i):
                h = hypDist(radialCoords[i],angularCoords[i],radialCoords[j],angularCoords[j])  #hyperbolic distance between the ith and the jth node
                if h<R:
                    globalConnectionProbability = 1
                elif h==R:
                    globalConnectionProbability = 0.5
                else: #R<h
                    globalConnectionProbability = 0
                if (nodeList[i],nodeList[j]) in edgeList or (nodeList[j],nodeList[i]) in edgeList:  #there is an edge between the ith and the jth node
                    L2 = L2*globalConnectionProbability
                else:  #there is no edge between node i and j
                    L2 = L2*(1-globalConnectionProbability)
    else: #for 0<T the function returns the -logarithm of the likelihood, i.e. the logarithmic loss
        L2 = 0 #log-likelihood!
        for i in range(1,N):
            for j in range(i):
                h = hypDist(radialCoords[i],angularCoords[i],radialCoords[j],angularCoords[j]) #hyperbolic distance between the ith and the jth node
                globalConnectionProbability = 1/(1+math.exp((h-R)/(2*T)))
                if (nodeList[i],nodeList[j]) in edgeList or (nodeList[j],nodeList[i]) in edgeList: #there is an edge between the ith and the jth node
                    if globalConnectionProbability == 0:
                        L2 = float('inf')
                    else:
                        L2 = L2-math.log(globalConnectionProbability)
                else: #there is no edge between the ith and the jth node
                    if globalConnectionProbability == 1:
                        L2 = float('inf')
                    else:
                        L2 = L2-math.log(1-globalConnectionProbability)
    return L2



#A function for calculating the gradient of the logarithmic loss at given m-beta-T values.
#nodeList is a list of the node identifiers
#edgeList is a list of tuples corresponding to the links in the examined graph
#N is the number of nodes in the examined graph
#m is a model parameter determining the average degree of the network together with the parameter L via the formula <k>=2*(m+L)
#b stands for the popularity fading parameter beta, determining the degree decay exponent
#T is the temperature regulating the clustering coefficient
#angularDist_dict is a dictionary created by the function initialEmbeddingForParameterEstimation, containing the angular distance for each node pair
#ID_radialID_dict is a dictionary created by the function initialEmbeddingForParameterEstimation, describing the radial order of the nodes
def gradLL(nodeList,edgeList,N,m,b,T,angularDist_dict,ID_radialID_dict):
    diffm=0
    diffb=0
    diffT=0
    for i in range(1,N): #i=nodeID according to the order in nodeList
        for j in range(i): #j=nodeID according to the order in nodeList
            d = angularDist_dict[(i,j)]
            ii = ID_radialID_dict[i]
            jj = ID_radialID_dict[j]

            if (nodeList[i],nodeList[j]) in edgeList or (nodeList[j],nodeList[i]) in edgeList: #there is an edge between the ith node and the jth node
                diffm = diffm - math.exp((math.acosh(math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1)) - math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1))*math.cos(d)) + 2*math.log((2*T*(N**(b - 1) - 1))/(m*math.sin(math.pi*T)*(b - 1))) - 2*math.log(N))/(2*T))/(T*m*(math.exp((math.acosh(math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1)) - math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1))*math.cos(d)) + 2*math.log((2*T*(N**(b - 1) - 1))/(m*math.sin(math.pi*T)*(b - 1))) - 2*math.log(N))/(2*T)) + 1))

                diffb = diffb - (math.exp((math.acosh(math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1)) - math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1))*math.cos(d)) + 2*math.log((2*T*(N**(b - 1) - 1))/(m*math.sin(math.pi*T)*(b - 1))) - 2*math.log(N))/(2*T))*((math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1))*(2*math.log(ii + 1) - 2*math.log(N)) + math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1))*math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*(2*math.log(jj + 1) - 2*math.log(N)) - math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1))*math.cos(d)*math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*(2*math.log(ii + 1) - 2*math.log(N)) - math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.cos(d)*math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1))*(2*math.log(jj + 1) - 2*math.log(N)))/((math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1)) - math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1))*math.cos(d))**2 - 1)**(1/2) + (m*math.sin(math.pi*T)*((2*T*(N**(b - 1) - 1))/(m*math.sin(math.pi*T)*(b - 1)**2) - (2*N**(b - 1)*T*math.log(N))/(m*math.sin(math.pi*T)*(b - 1)))*(b - 1))/(T*(N**(b - 1) - 1))))/(2*T*(math.exp((math.acosh(math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1)) - math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1))*math.cos(d)) + 2*math.log((2*T*(N**(b - 1) - 1))/(m*math.sin(math.pi*T)*(b - 1))) - 2*math.log(N))/(2*T)) + 1))

                diffT = diffT - (math.exp((math.acosh(math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1)) - math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1))*math.cos(d)) + 2*math.log((2*T*(N**(b - 1) - 1))/(m*math.sin(math.pi*T)*(b - 1))) - 2*math.log(N))/(2*T))*((math.acosh(math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1)) - math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1))*math.cos(d)) + 2*math.log((2*T*(N**(b - 1) - 1))/(m*math.sin(math.pi*T)*(b - 1))) - 2*math.log(N))/(2*T**2) - (m*math.sin(math.pi*T)*((2*(N**(b - 1) - 1))/(m*math.sin(math.pi*T)*(b - 1)) - (2*math.pi*T*math.cos(math.pi*T)*(N**(b - 1) - 1))/(m*math.sin(math.pi*T)**2*(b - 1)))*(b - 1))/(2*T**2*(N**(b - 1) - 1))))/(math.exp((math.acosh(math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1)) - math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1))*math.cos(d)) + 2*math.log((2*T*(N**(b - 1) - 1))/(m*math.sin(math.pi*T)*(b - 1))) - 2*math.log(N))/(2*T)) + 1)


            else: #there is no edge between the ith node and the jth node
                diffm = diffm - math.exp((math.acosh(math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1)) - math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1))*math.cos(d)) + 2*math.log((2*T*(N**(b - 1) - 1))/(m*math.sin(math.pi*T)*(b - 1))) - 2*math.log(N))/(2*T))/(T*m*(1/(math.exp((math.acosh(math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1)) - math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1))*math.cos(d)) + 2*math.log((2*T*(N**(b - 1) - 1))/(m*math.sin(math.pi*T)*(b - 1))) - 2*math.log(N))/(2*T)) + 1) - 1)*(math.exp((math.acosh(math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1)) - math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1))*math.cos(d)) + 2*math.log((2*T*(N**(b - 1) - 1))/(m*math.sin(math.pi*T)*(b - 1))) - 2*math.log(N))/(2*T)) + 1)**2)

                diffb = diffb - (math.exp((math.acosh(math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1)) - math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1))*math.cos(d)) + 2*math.log((2*T*(N**(b - 1) - 1))/(m*math.sin(math.pi*T)*(b - 1))) - 2*math.log(N))/(2*T))*((math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1))*(2*math.log(ii + 1) - 2*math.log(N)) + math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1))*math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*(2*math.log(jj + 1) - 2*math.log(N)) - math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1))*math.cos(d)*math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*(2*math.log(ii + 1) - 2*math.log(N)) - math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.cos(d)*math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1))*(2*math.log(jj + 1) - 2*math.log(N)))/((math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1)) - math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1))*math.cos(d))**2 - 1)**(1/2) + (m*math.sin(math.pi*T)*((2*T*(N**(b - 1) - 1))/(m*math.sin(math.pi*T)*(b - 1)**2) - (2*N**(b - 1)*T*math.log(N))/(m*math.sin(math.pi*T)*(b - 1)))*(b - 1))/(T*(N**(b - 1) - 1))))/(2*T*(1/(math.exp((math.acosh(math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1)) - math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1))*math.cos(d)) + 2*math.log((2*T*(N**(b - 1) - 1))/(m*math.sin(math.pi*T)*(b - 1))) - 2*math.log(N))/(2*T)) + 1) - 1)*(math.exp((math.acosh(math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1)) - math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1))*math.cos(d)) + 2*math.log((2*T*(N**(b - 1) - 1))/(m*math.sin(math.pi*T)*(b - 1))) - 2*math.log(N))/(2*T)) + 1)**2)

                diffT = diffT - (math.exp((math.acosh(math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1)) - math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1))*math.cos(d)) + 2*math.log((2*T*(N**(b - 1) - 1))/(m*math.sin(math.pi*T)*(b - 1))) - 2*math.log(N))/(2*T))*((math.acosh(math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1)) - math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1))*math.cos(d)) + 2*math.log((2*T*(N**(b - 1) - 1))/(m*math.sin(math.pi*T)*(b - 1))) - 2*math.log(N))/(2*T**2) - (m*math.sin(math.pi*T)*((2*(N**(b - 1) - 1))/(m*math.sin(math.pi*T)*(b - 1)) - (2*math.pi*T*math.cos(math.pi*T)*(N**(b - 1) - 1))/(m*math.sin(math.pi*T)**2*(b - 1)))*(b - 1))/(2*T**2*(N**(b - 1) - 1))))/((1/(math.exp((math.acosh(math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1)) - math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1))*math.cos(d)) + 2*math.log((2*T*(N**(b - 1) - 1))/(m*math.sin(math.pi*T)*(b - 1))) - 2*math.log(N))/(2*T)) + 1) - 1)*(math.exp((math.acosh(math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.cosh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1)) - math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(ii + 1))*math.sinh(math.log(N)*(2*b - 2) - 2*b*math.log(jj + 1))*math.cos(d)) + 2*math.log((2*T*(N**(b - 1) - 1))/(m*math.sin(math.pi*T)*(b - 1))) - 2*math.log(N))/(2*T)) + 1)**2)

    return[diffm,diffb,diffT]



#A function for determining the parameters needed for the logarithmic loss calculation via logarithmic loss minimization.
#G is the NetworkX graph to be embedded
#N is the number of nodes in the graph to be embedded
#weightingType is a string corresponding to the name of the pre-weighting rule to be used for creating the initial embedding that is used for the parameter estimation. It can be set to 'RA1', 'RA2', 'RA3', 'RA4', 'EBC' or, if no pre-weighting is needed, to None.
#CAorEA is a string that can be set to 'CA' or to 'EA', determining whether to use circular adjustment or equidistant adjustment to create angular coordinates on the native disk representation of the hyperbolic plane from the node coordinates obtained in the reduced space
#pathString is a string of the path where we want to save the resulted model parameters and the changes in the model parameters and the logarithmic loss throughout the optimization process
#startingm,startingBeta,startingT: the initial estimations of the 1<=m, 0.1<beta<0.99 and 0.1<T<0.99 model parameters. The default is None, meaning that initially m will be set to the half of the average degree, while beta and T will be set to 0.5
#initStepSizeFactor: the size of the first step will be initStepSizeFactor times the initial distance from the side of the allowed interval
#precision: the optimization stops, if the norm of the next displacement vector in the parameter space would be smaller than this value
#maxIters is the maximum number of steps to be carried out in the parameter space
#maxm is the allowed maximum of the parameter m, determining the average degree of the network together with the parameter L via the formula <k>=2*(m+L). The default maximum of m is the average degree.
def parameterEstimation_logloss(G, N, weightingType, CAorEA, pathString, startingm=None, startingBeta=None, startingT=None, initStepSizeFactor=0.2, precision=0.01, maxIters=100, maxm=None):
    [angularCoords_ncMCE,angularDist_dict,radialNodeOrder,ID_radialID_dict,nodeANDdegree] = initialEmbeddingForParameterEstimation(G,N,weightingType, CAorEA)
    halfOfAvDeg=sum(list(nodeANDdegree.values()))/(2*N)
    print('smallest degree='+str(min(nodeANDdegree.values()))+', half of average degree='+str(halfOfAvDeg))
    if startingm==None:
        startingm=halfOfAvDeg
    if startingBeta==None:
        startingBeta=0.5 #0<beta<1
    if startingT==None:
        startingT=0.5 #0<T<1
    minm=1 #m values smaller than this are not allowed
    if maxm==None:
        maxm=2*halfOfAvDeg #m values larger than this are not allowed
    minbeta=0.1 #popularity fading parameter values smaller than this are not allowed (in order to avoid computational difficulties)
    maxbeta=0.99 #popularity fading parameter values larger than this are not allowed (in order to avoid computational difficulties)
    minT=0.1 #temperature values smaller than this are not allowed (in order to avoid computational difficulties)
    maxT=0.99 #temperature values larger than this are not allowed (in order to avoid computational difficulties)

    r=np.zeros(N) #array for storing the radial coordinates
    mInEachStep=[]
    betaInEachStep=[]
    TinEachStep=[]
    LLinEachStep=[]

    nodeList=list(G.nodes)
    edgeList=list(G.edges)

    #set the starting point
    m=startingm
    beta=startingBeta
    T=startingT
    mInEachStep.append(m)
    betaInEachStep.append(beta)
    TinEachStep.append(T)
    print("m="+str(m))
    print("beta="+str(beta))
    print("T="+str(T))

    #calculate the initial logarithmic loss
    j=0
    for i in radialNodeOrder:
        r[i]=2*(beta*math.log(j+1)+(1-beta)*math.log(N)) #the radial coordinate of the node that is the ith according to the order in G.nodes
        j=j+1
    R = 2*math.log(N)-2*math.log((2*T*(1-math.exp(-(1-beta)*math.log(N))))/(math.sin(T*math.pi)*m*(1-beta))) #this in true only for beta<1!
    LL=calculateLikelihood(T,N,R,nodeList,edgeList,r,angularCoords_ncMCE)
    print("LL="+str(LL))
    LLinEachStep.append(LL)

    #calculate the gradient of the logarithmic loss at the initial m-beta-T values
    [diffm,diffb,diffT] = gradLL(nodeList,edgeList,N,m,beta,T,angularDist_dict,ID_radialID_dict)
    print([diffm,diffb,diffT])

    iters=0
    #move in the parameter space in the direction of the negative gradient
    #size of first step := initStepSizeFactor*initial distance from the side of the allowed interval in the direction of the initial -derivative
    if diffm<0: #the optimum is at an m value that is larger than the current m value
        if m<maxm:
            mStepSize = initStepSizeFactor*(m-maxm)
            mNew = m-mStepSize
            stepSizeMultiplier_m = abs(mStepSize/diffm) #this multiplying factor will be used in the following steps
        else: #we can not step in the desired direction
            mStepSize = 0
            mNew = m
            stepSizeMultiplier_m = 0.01 #this multiplying factor will be used in the following steps
    elif 0<diffm: #the optimum is at an m value that is smaller than the current m value
        if minm<m:
            mStepSize = initStepSizeFactor*(m-minm)
            mNew = m-mStepSize
            stepSizeMultiplier_m = abs(mStepSize/diffm) #this multiplying factor will be used in the following steps
        else: #we can not step in the desired direction
            mStepSize = 0
            mNew = m
            stepSizeMultiplier_m = 0.01 #this multiplying factor will be used in the following steps
    else: #diffm=0
        mStepSize = 0
        mNew = mm
        stepSizeMultiplier_m = 0.01 #this multiplying factor will be used in the following steps

    if diffb<0: #the optimum is at a beta value that is larger than the current popularity fading parameter
        if beta<maxbeta:
            betaStepSize = initStepSizeFactor*(beta-maxbeta)
            betaNew = beta-betaStepSize
            stepSizeMultiplier_beta = abs(betaStepSize/diffb) #this multiplying factor will be used in the following steps
        else: #we can not step in the desired direction
            betaStepSize = 0
            betaNew = beta
            stepSizeMultiplier_beta = 0.01 #this multiplying factor will be used in the following steps
    elif 0<diffb: #the optimum is at a beta value that is smaller than the current popularity fading parameter
        if minbeta<beta:
            betaStepSize = initStepSizeFactor*(beta-minbeta)
            betaNew = beta-betaStepSize
            stepSizeMultiplier_beta = abs(betaStepSize/diffb) #this multiplying factor will be used in the following steps
        else: #we can not step in the desired direction
            betaStepSize = 0
            betaNew = beta
            stepSizeMultiplier_beta = 0.01 #this multiplying factor will be used in the following steps
    else: #diffb=0
        betaStepSize = 0
        betaNew = beta
        stepSizeMultiplier_beta = 0.01 #this multiplying factor will be used in the following steps

    if diffT<0: #the optimum is at a T value that is larger than the current temperature
        if T<maxT:
            TstepSize = initStepSizeFactor*(T-maxT)
            Tnew = T-TstepSize
            stepSizeMultiplier_T = abs(TstepSize/diffT) #this multiplying factor will be used in the following steps
        else: #we can not step in the desired direction
            TstepSize = 0
            Tnew = T
            stepSizeMultiplier_T = 0.01 #this multiplying factor will be used in the following steps
    elif 0<diffT: #the optimum is at a T value that is smaller than the current temperature
        if minT<T:
            TstepSize = initStepSizeFactor*(T-minT)
            Tnew = T-TstepSize
            stepSizeMultiplier_T = abs(TstepSize/diffT) #this multiplying factor will be used in the following steps
        else: #we can not step in the desired direction
            TstepSize = 0
            Tnew = T
            stepSizeMultiplier_T = 0.01 #this multiplying factor will be used in the following steps
    else: #diffT=0
        TstepSize = 0
        Tnew = T
        stepSizeMultiplier_T = 0.01 #this multiplying factor will be used in the following steps
    print(str([mStepSize,betaStepSize,TstepSize]))

    totalStepSize=np.linalg.norm(np.array([mStepSize,betaStepSize,TstepSize]))

    while precision<totalStepSize and iters<maxIters: #stop the optimization if the norm of the next displacement vector in the parameter space is smaller than precision or we have already taken maxIters number of steps
        #move in the parameter space in the direction of the negative gradient
        m = mNew
        mInEachStep.append(m)
        beta = betaNew
        betaInEachStep.append(beta)
        T = Tnew
        TinEachStep.append(T)
        iters=iters+1 #one step has been done
        print('iter '+str(iters)+" is done")
        print("m="+str(m))
        print("beta="+str(beta))
        print("T="+str(T))

        #calculate the new LL
        j=0
        for i in radialNodeOrder:
            r[i]=2*(beta*math.log(j+1)+(1-beta)*math.log(N))
            j=j+1
        R = 2*math.log(N)-2*math.log((2*T*(1-math.exp(-(1-beta)*math.log(N))))/(math.sin(T*math.pi)*m*(1-beta))) #beta<1-re igaz csak!
        LL=calculateLikelihood(T,N,R,nodeList,edgeList,r,angularCoords_ncMCE)
        print("LL="+str(LL))
        LLinEachStep.append(LL)
        
        #calculate the gradient of the logarithmic loss at the current m-beta-T values
        [diffm,diffb,diffT] = gradLL(nodeList,edgeList,N,m,beta,T,angularDist_dict,ID_radialID_dict)

        #calculate the step sizes and the possible new parameters
        if diffm<0: #the optimum is at an m value that is larger than the current m value
            if m<maxm:
                mNew = m-stepSizeMultiplier_m*diffm
                if maxm<mNew:
                    mNew = maxm
                mStepSize = m-mNew #the step that we actually take can be smaller than stepSizeMultiplier_m*diffm because of the boundaries
            else: #we can not step in the desired direction
                mStepSize = 0
                mNew = m
        elif 0<diffm: #the optimum is at an m value that is smaller than the current m value
            if minm<m:
                mNew = m-stepSizeMultiplier_m*diffm
                if mNew<minm:
                    mNew = minm
                mStepSize = m-mNew #the step that we actually take can be smaller than stepSizeMultiplier_m*diffm because of the boundaries
            else: #we can not step in the desired direction
                mStepSize = 0
                mNew = m
        else: #diffm=0
            mStepSize = 0
            mNew = m

        if diffb<0: #the optimum is at a beta value that is larger than the current popularity fading parameter
            if beta<maxbeta:
                betaNew = beta-stepSizeMultiplier_beta*diffb
                if maxbeta<betaNew:
                    betaNew = maxbeta
                betaStepSize = beta-betaNew #the step that we actually take can be smaller than stepSizeMultiplier_beta*diffb because of the boundaries
            else: #we can not step in the desired direction
                betaStepSize = 0
                betaNew = beta
        elif 0<diffb: #the optimum is at a beta value that is smaller than the current popularity fading parameter
            if minbeta<beta:
                betaNew = beta-stepSizeMultiplier_beta*diffb
                if betaNew<minbeta:
                    betaNew = minbeta
                betaStepSize = beta-betaNew #the step that we actually take can be smaller than stepSizeMultiplier_beta*diffb because of the boundaries
            else: #we can not step in the desired direction
                betaStepSize = 0
                betaNew = beta
        else: #diffbeta=0
            betaStepSize = 0
            betaNew = beta

        if diffT<0: #the optimum is at a T value that is larger than the current temperature
            if T<maxT:
                Tnew = T-stepSizeMultiplier_T*diffT
                if maxT<Tnew:
                    Tnew = maxT
                TstepSize = T-Tnew #the step that we actually take can be smaller than stepSizeMultiplier_T*diffT because of the boundaries
            else: #we can not step in the desired direction
                TstepSize = 0
                Tnew = T
        elif 0<diffT: #the optimum is at a T value that is smaller than the current temperature
            if minT<T:
                Tnew = T-stepSizeMultiplier_T*diffT
                if Tnew<minT:
                    Tnew = minT
                TstepSize = T-Tnew #the step that we actually take can be smaller than stepSizeMultiplier_T*diffT because of the boundaries
            else: #we can not step in the desired direction
                TstepSize = 0
                Tnew = T
        else: #diffT=0
            TstepSize = 0
            Tnew = T
        print(str([mStepSize,betaStepSize,TstepSize]))
        totalStepSize=np.linalg.norm(np.array([mStepSize,betaStepSize,TstepSize]))

    #carry out the last step (it is already calculated...)
    m = mNew
    mInEachStep.append(m)
    beta = betaNew
    betaInEachStep.append(beta)
    T = Tnew
    TinEachStep.append(T)
    #calculate the new LL
    j=0
    for i in radialNodeOrder:
        r[i]=2*(beta*math.log(j+1)+(1-beta)*math.log(N))
        j=j+1
    R = 2*math.log(N)-2*math.log((2*T*(1-math.exp(-(1-beta)*math.log(N))))/(math.sin(T*math.pi)*m*(1-beta))) #this in only true for beta<1!
    LL=calculateLikelihood(T,N,R,nodeList,edgeList,r,angularCoords_ncMCE)
    print("LL="+str(LL))
    LLinEachStep.append(LL)

    print("m="+str(m)+", beta="+str(beta)+", T="+str(T)+"\n\n\n")
    #save the obtained parameter set
    ReportFileHandler = open(pathString + "EPSOparameters.txt", 'w')
    ReportFileHandler.write('N:\t' + str(N))
    ReportFileHandler.write('\nm:\t' + str(m))
    ReportFileHandler.write('\nL:\t' + str(halfOfAvDeg-m))
    ReportFileHandler.write('\nbeta:\t' + str(beta))
    ReportFileHandler.write('\nT:\t' + str(T))
    ReportFileHandler.close()

    #save how the parameters have been changed
    paramsAndLLinEachStep=np.column_stack((np.array(mInEachStep),np.array(betaInEachStep),np.array(TinEachStep),np.array(LLinEachStep)))
    np.savetxt(pathString+"paramsInEachStep_startedFrom_m"+("%.1f"%startingm)+"_beta"+("%.1f"%startingBeta)+"_T"+("%.1f"%startingT)+".txt",paramsAndLLinEachStep,delimiter='\t') #columns: m, beta, T, logarithmic loss

    #calculate the cutoff distance of the global connection probability for the estimated PSO model parameters
    R = 2*math.log(N)-2*math.log((2*T*(1-math.exp(-(1-beta)*math.log(N))))/(math.sin(T*math.pi)*m*(1-beta))) #0<T, beta<1 was assumed here

    return [m,beta,T,R] #return the optimalized model parameters




###RADIAL COORDINATES###

#A function for determining the radial order of the nodes with ties in the degrees broken randomly.
#G is the NetworkX Graph to be embedded
#pathString is a string of the path where we want to save the radial node order
#Example for function call:
#   radOrder=embedding.determineRadialOrder(G,os.getcwd()+"/"+directoryName+"/")
def determineRadialOrder(G,pathString):
    #choose a descending degree order with ties broken randomly
    degreeArray = np.array([Deg for (nodeID,Deg) in G.degree],dtype=int) #array with the degree of all nodes
    randArray = np.random.random(degreeArray.size)
    radialNodeOrder = np.lexsort((randArray,degreeArray))[::-1] #node indices corresponding to the order in G.nodes sorted in a descending order of the degree
            #e.g. if list(G.nodes) is ['a','b','c','d','e','f'] and the degrees are the following: 'a':2, 'b':4, 'c':1, 'd':4, 'e':5, 'f':3,
                 #then degreeArray is np.array([2,4,1,4,5,3]) and the radialNodeOrder is array([4,3,1,5,0,2]) or array([4,1,3,5,0,2])
    #save the selected radial order
    nodeIdentifierList = list(G.nodes)
    fileHandler = open(pathString+"radialNodeOrder.txt","w")
    for nodeID in radialNodeOrder:
        fileHandler.write(str(nodeIdentifierList[nodeID])+"\n")
    fileHandler.close()
    return radialNodeOrder


#A function for determining the radial coordinates of the network nodes in the native representation of the hyperbolic plane of curvature K=-1.
#N is the number of nodes in graph G
#beta is the popularity fading parameter
#radialNodeOrder is a list of the node identifiers in the increasing order of the radial coordinates
#The function returns the NumPy array r containing the radial coordinates of the network nodes in the native representation of the hyperbolic plane. The ith element of the returned array (i=0,1,...,N-1) is the coordinate of the node that is the ith according to the order in G.nodes.
def determine_r(N,beta,radialNodeOrder):
    #create the array of the radial coordinates of the nodes
    r = np.zeros(N)
    j = 0
    for i in radialNodeOrder:
        r[i] = 2*(beta*math.log(j+1)+(1-beta)*math.log(N)) #r[i] is the radial coordinate of the node that is the ith according to the order in G.nodes
        j = j+1
    return r





#ANGULAR COORDINATES

#A function for pre-weighting the edges in order to facilitate the estimation of the angular arrangement of the nodes in hyperbolic embeddings. Larger weight corresponds to less similarity.
#G is the NetworkX Graph to be pre-weighted
#weightingType is a string corresponding to the name of the pre-weighting rule to be used, it can be set to 'RA1', 'RA2', 'RA3'. 'RA4' or 'EBC'
#Example for function call:
#   G_preweighted = embedding.preWeighting(G,'RA1')
def preWeighting(G,weightingType):
    G_weighted = nx.Graph()
    G_weighted.add_nodes_from(G.nodes) #keep the node order of the original graph
    if weightingType=='RA1':
        for (i, j) in G.edges():  # (i,j)=a tuple, which denotes the edge between node u and v
            CNset = set(nx.common_neighbors(G, i, j))  # set of the common neighbors' indices
            # set=unordered collection with no duplicate elements,
            # set operations (union, intersect, complement) can be executed (see RAtype==2)
            CN = len(CNset)  # number of common neighbors of nodes i and j
            # assign a weight to the i-j edge:
            w = (G.degree(i) + G.degree(j) + G.degree(i) * G.degree(j)) / (1 + CN)
            G_weighted.add_edge(i,j,weight=w)
    elif weightingType=='RA2':
        for (i, j) in G.edges():  # (i,j)=a tuple, which denotes the edge between node u and v
            CNset = set(nx.common_neighbors(G, i, j))  # set of the common neighbors' indices
            CN = len(CNset)  # number of common neighbors of nodes i and j

            # ei=the external degree of the node i with respect to node j,
            # i.e. the number of links from i to neither j nor the common neighbors with j,
            # i.e. the number of i's neighbors without node j and the common neighbors with j
            neighborSet_i = {n for n in G[i]}  # set with the indices of the neighbors of node i
            # G[i]=adjacency dictionary of node i -> iterating over its keys(=neighboring node indices)
            ei = len(neighborSet_i - {j} - CNset)

            # ej=the external degree of the node j with respect to node i,
            # i.e. the number of links from j to neither i nor the common neighbors with i,
            # i.e. the number of j's neighbors without node i and the common neighbors with i
            neighborSet_j = {n for n in G[j]}  # set with the indices of the neighbors of node j
            # G[j]=adjacency dictionary of node j -> iterating over its keys(=neighboring node indices)
            ej = len(neighborSet_j - {i} - CNset)

            # assign a weight to the i-j edge:
            w = (1 + ei + ej + ei * ej) / (1 + CN)
            G_weighted.add_edge(i,j,weight=w)
    elif weightingType=='RA3':
        for (i, j) in G.edges():  # (i,j)=a tuple, which denotes the edge between node u and v
            CNset = set(nx.common_neighbors(G, i, j))  # set of the common neighbors' indices
            CN = len(CNset)  # number of common neighbors of nodes i and j
            # assign a weight to the i-j edge:
            w = (G.degree(i) + G.degree(j)) / (1 + CN)
            G_weighted.add_edge(i,j,weight=w)
    elif weightingType=='RA4':
        for (i, j) in G.edges():  # (i,j)=a tuple, which denotes the edge between node u and v
            CNset = set(nx.common_neighbors(G, i, j))  # set of the common neighbors' indices
            CN = len(CNset)  # number of common neighbors of nodes i and j
            # assign a weight to the i-j edge:
            w = (G.degree(i) * G.degree(j)) / (1 + CN)
            G_weighted.add_edge(i,j,weight=w)
    elif weightingType=='EBC':
        #create a dictionary, which contains all the shortest paths between all node pairs
            #shortestPathsDict[(source,target)] is the list of shortest paths from node with ID source to node with ID target
                #a path is a list of nodes following each other in the path
            #the graph to be embedded should be connected (all nodes can be reached from any node)
        shortestPathsDict = {}
        nodeList=list(G.nodes)
        N = len(nodeList)
        for u in range(N-1): #u=0,1,...,N-2
            for v in range(u+1,N): #v=u+1,...,N-1
            #these loops are sufficient only if graph G is undirected (the same number of paths lead from the uth node to the vth node and from the vth node to the uth node) and does not contain any self-loops
                node_u = nodeList[u]
                node_v = nodeList[v]
                shortestPathsDict[(node_u,node_v)]=[p for p in nx.all_shortest_paths(G,source=node_u,target=node_v,weight=None)] #weight=None: every edge has weight/distance/cost 1 (the possible current weights are disregarded)

        #weight all the edges
        for (i,j) in G.edges():
            w=0 #initialize the weight of the i-j edge
            for u in range(N-1):
                for v in range(u+1,N):
                    shortestPathsBetween_uv = shortestPathsDict[(nodeList[u],nodeList[v])] #list of shortest paths between the uth node and the vth node
                    sigma = len(shortestPathsBetween_uv) #the total number of shortest paths between the uth node and the vth node
                    #count those paths between node u and node v which contains the i-j edge
                    sigma_ij = 0
                    for q in shortestPathsBetween_uv: #q=list of nodes following each other in a path between the uth node and the vth node
                        if i in q and j in q: #since q is a shortest path, therefore in this case abs(q.index(i)-q.index(j))==1 is already granted
                            sigma_ij = sigma_ij+1
                    w=w+(sigma_ij/sigma)
            G_weighted.add_edge(i,j,weight=w) #assign a weight to the i-j edge
    else:
        print('False parameter: weightingType\n')
    return G_weighted



#A function for performing circular adjustment of the node coordinates (rescaling into the inteval [0,2pi)) obtained with dimension reduction.
#coordinatesAfterDimensionReduction is a NumPy array containing the node coordinates in the reduced space (the ith element is the coordinate of the node that is the ith according to the order in G.nodes)
#N is the number of nodes in the graph to be embedded
def rescalingTo0_2pi(coordinatesAfterDimensionReduction, N):
    minCoord = np.min(coordinatesAfterDimensionReduction)
    maxCoord = np.max(coordinatesAfterDimensionReduction)
    normalizedCoords = (coordinatesAfterDimensionReduction - minCoord) / (maxCoord - minCoord)  # in [0,1]
    rightBoundary = 2 * math.pi * (N - 1) / N;  # somewhat smaller than 2pi
    phi = normalizedCoords * rightBoundary  # in [0,rightBoundary], close to the required [0,2pi)
    return phi



#A function for performing equidistant adjustment of the angular coordinates. This means the organization of the angular coordinates equidistantly in the interval [0,2pi), retaining the order of the nodes concerning the coordinates in the reduced space.
#coordinatesAfterDimensionReduction is a NumPy array containing the node coordinates in the reduced space (the ith element is the coordinate of the node that is the ith according to the order in G.nodes)
#N is the number of nodes in the graph to be embedded
def EA(coordinatesAfterDimensionReduction, N):
    # the indices of the nodes in the order corresponding to the increasing coordinates in the reduced space:
    nodeOrder = np.argsort(coordinatesAfterDimensionReduction) # the angular coordinates will be assigned in this order

    dTheta = 2 * math.pi / N
    j = 0
    Theta = np.zeros(N)
    for i in nodeOrder:
        Theta[i] = j * dTheta
        j = j + 1
    return Theta



#A function for performing noncentered minimum curvilinear embedding.
#G is the NetworkX graph to be embedded
#N is the number of nodes in the graph to be embedded
#weightingType is a string corresponding to the name of the pre-weighting rule to be used, it can be set to 'RA1', 'RA2', 'RA3', 'RA4', 'EBC' or, if no pre-weighting is needed, to None
#CAorEA is a string that can be set to 'CA' or to 'EA', determining whether to use circular adjustment or equidistant adjustment to create angular coordinates on the native disk representation of the hyperbolic plane from the node coordinates obtained in the reduced space
#The function returns a NumPy array containing the angular node coordinates on the hyperbolic plane (the ith element is the angle of the ith node according to the order in G.nodes)
def ncMCE(G,N,weightingType,CAorEA):
    if weightingType!=None:
        G_weighted = preWeighting(G,weightingType)
        Tree=nx.minimum_spanning_tree(G_weighted) #default: Kruskal’s algorithm (If G is not connected, the minimum spanning tree algorithm finds the spanning FOREST!)
    else: #no preweighting
        Tree=nx.minimum_spanning_tree(G)
    pathLenList=dict(nx.shortest_path_length(Tree,weight='weight')) #distances between all node pairs over the minimum spanning tree Tree
        #pathLenList[source][target]=length of the shortest path from the node with index "source" to the node with index "target"
    D=np.array([[pathLenList[i][j] for j in G.nodes] for i in G.nodes]) #a matrix with the distances
    U,S_allElements,VH=np.linalg.svd(D,full_matrices=False)
    S_reducedElements=np.concatenate((S_allElements[:2],np.zeros(len(S_allElements)-2))) #only the first 2 singular values are retained
    Sreduced=np.diag(S_reducedElements)
    X=np.transpose(np.matmul(np.sqrt(Sreduced),VH)) #the ith column contains the ith coordinate of all nodes in the reduced space
    secondCoords=X[:,1]
    if CAorEA=='CA': #circular adjustment
        Theta=rescalingTo0_2pi(secondCoords,N)
    elif CAorEA=='EA': #equidistant adjustment
        Theta=EA(secondCoords,N)
    else:
        print('False parameter: CAorEA\n')
    return Theta



#CORRECTION OF THE ANGULAR COORDINATES
#This function performs the angular correction of an embedding (e.g. the ncMCE method's result) based on logarithmic loss minimization and returns the corrected angular coordinates of the network nodes and the logarithmic loss of the embedding after each correction round, each in a NumPy array.
#G is the embedded NetworkX graph
#N is the number of nodes
#T is the temperature
#R is the cutoff distance of the global connection probability
#radialCoords is a NumPy array with the radial coordinates of the network nodes resulted from the embedding (the ith element is the coordinate of the ith node according to the order in G.nodes)
#originalAngularCoords is a NumPy array with the angular coordinates of the network nodes resulted from a fast embedding method, e.g. from the ncMCE (the ith element is the coordinate of the ith node according to the order in G.nodes)
#numberOfSwappingRounds is a non-negative integer, the number of correction rounds in which the range of the tested angular coordinates is restricted for each node between its current second angular neighbors
#numberOfAnglesInSwappingRounds is a non-negative integer, the number of each node's tested angular positions in the swapping rounds
#numberOfNonswappingRounds is a non-negative integer, the number of correction rounds in which the range of the tested angular coordinates is restricted for each node between its current first angular neighbors
#numberOfAnglesInNonswappingRounds is a non-negative integer, the number of each node's tested angular positions in the non-swapping rounds
#Example for function call:
#        [angularCoords_ncMCE_corr,LL_eachRound]=embedding.angleCorrection(G,N,m,beta,T,R,radialCoords_ncMCE,angularCoords_ncMCE,5,6,3,5)
def angleCorrection(G, N, T, R, radialCoords, originalAngularCoords, numberOfSwappingRounds, numberOfAnglesInSwappingRounds, numberOfNonswappingRounds, numberOfAnglesInNonswappingRounds):
    edgeList = list(G.edges()) #list of tuples denoting the edges
    nodeList = list(G.nodes()) #list of the node identifiers

    #declaring the returned variables
    angularCoords = np.array(originalAngularCoords,copy=True) #array for storing the corrected angular coordinates of the nodes; initially set to the noncorrected angular coordinates (resulted e.g. from the ncMCE method)
    LLglobal_eachRound = np.zeros(numberOfSwappingRounds+numberOfNonswappingRounds+1) #LLglobal_eachRound[0] is the logarithmic loss, i.e. -log(global likelihood) of the embedding without angular correction


    #calculate the initial logarithmic loss (global likelihood without angle correction)
    LLmatrix = np.zeros((N,N)) #LLmatrix[n][m] is the current contribution of the pair of the nth and the mth nodes to the logarithmic loss
    LL_list = np.zeros(N) #LL_list[n] is the current contribution to the logarithmic loss of all node pairs that can be formed with the nth node, i.e. the total contribution of the nth node to the logarithmic loss
    for n in range(1,N):
        for m in range(n):
            h = hypDist(radialCoords[n],originalAngularCoords[n],radialCoords[m],originalAngularCoords[m]) #hyperbolic distance between the nth node and the mth node
            globalConnectionProbability = 1/(1+math.exp((h-R)/(2*T)))
            if (nodeList[n],nodeList[m]) in edgeList or (nodeList[m],nodeList[n]) in edgeList: #there is an edge between the nth and the mth nodes
                LLmatrix[n][m] = -math.log(globalConnectionProbability)
                LLmatrix[m][n] = LLmatrix[n][m]
                LL_list[n] = LL_list[n]+LLmatrix[n][m]
                LL_list[m] = LL_list[m]+LLmatrix[n][m]
                LLglobal_eachRound[0] = LLglobal_eachRound[0]+LLmatrix[n][m]
            else: #there is no edge between the nth and the mth nodes
                LLmatrix[n][m] = -math.log(1-globalConnectionProbability)
                LLmatrix[m][n] = LLmatrix[n][m]
                LL_list[n] = LL_list[n]+LLmatrix[n][m]
                LL_list[m] = LL_list[m]+LLmatrix[n][m]
                LLglobal_eachRound[0] = LLglobal_eachRound[0]+LLmatrix[n][m]


    #create the variables characterizing the current angular arrangement of the network nodes and initialize them according to the result of the original (noncorrected) embedding
    angularNodeOrder = np.argsort(angularCoords) #array with the ordinal number of the network nodes in the current angular node order (the angle is defined to increase for counterclockwise rotations)
    firstNeighborsPosition = np.zeros((N,2)) #matrix for storing the current angular coordinate of the current first clockwise and first counterclockwise neighbor of each node
    secondNeighborsPosition =np.zeros((N,2)) #matrix for storing the current angular coordinate of the current second clockwise and second counterclockwise neighbor of each node
    for n in range(N):
        indexOfn=np.where(angularNodeOrder==n)[0][0] #position of the node that is the nth in nodeList in the angular node order
        firstNeighborsPosition[n][0]=angularCoords[angularNodeOrder[indexOfn-1]] #angle of the first clockwise neighbor of the node that is the nth in nodeList
        firstNeighborsPosition[n][1]=angularCoords[angularNodeOrder[(indexOfn+1)%N]] #angle of the first counterclockwise neigbor of the node that is the nth in nodeList
        secondNeighborsPosition[n][0]=angularCoords[angularNodeOrder[indexOfn-2]] #angle of the second clockwise neighbor of the node that is the nth in nodeList
        secondNeighborsPosition[n][1]=angularCoords[angularNodeOrder[(indexOfn+2)%N]] #angle of the second counterclockwise neighbor of the node that is the nth in nodeList



    #do the angle correction
    changeInLL = 0 #the total change in the logarithmic loss compared to the noncorrected angular arrangement (resulted e.g. from the ncMCE method); it decreases as the embedding quality increases

    for i in range(numberOfSwappingRounds): #perform the swapping rounds
        print(str(i+1)+". swapping round")
        for n in range(N): #iterate over all nodes
            #determine the angular coordinates to be tested for the node that is the nth in nodeList
            angularRange=math.pi-math.fabs(math.pi-math.fabs(secondNeighborsPosition[n][0]-secondNeighborsPosition[n][1])) #the angular distance between the second neighbors of the nth node; it is the width of the explored angular range
            angleStep=angularRange/(numberOfAnglesInSwappingRounds+1) #angular distance between the equidistantly distributed test positions; the borders of the explored angular range are not investigated
            anglesToTry=np.zeros(numberOfAnglesInSwappingRounds) #storing the tested angular positions of the nth node
            for a in range(numberOfAnglesInSwappingRounds):
                anglesToTry[a]=(secondNeighborsPosition[n][0]+(a+1)*angleStep)%(2*math.pi)

            #calculate the logaritmic losses related to the nth node for the tested angular positions
            otherNodes = list(range(N))
            otherNodes.remove(n) #otherNodes is a list of the indices of all nodes but n
            LL_n = np.zeros((numberOfAnglesInSwappingRounds,N)) #LL_n[j][m] is the contribution of the pair of the nth and the mth nodes to the logarithmic loss when the nth node is placed at the jth test position
            LL_n_total = np.zeros(numberOfAnglesInSwappingRounds+1) #LL_n_total[j] is the total contribution of the nth node to the logarithmic loss when it is placed at the jth test position
            #the last element of the array LL_n_total stores the current total contribution of the nth node (at its current angular position), thus if none of the new angles are better than the current angle, the current can be kept
            LL_n_total[-1]=LL_list[n]
            for j in range(numberOfAnglesInSwappingRounds):
                for m in otherNodes:
                    h = hypDist(radialCoords[n],anglesToTry[j],radialCoords[m],angularCoords[m]) #hyperbolic distance between node n and m
                    globalConnectionProbability = 1/(1+math.exp((h-R)/(2*T)))
                    if (nodeList[n],nodeList[m]) in edgeList or (nodeList[n],nodeList[m]) in edgeList: #there is an edge between the nth and the mth nodes
                        LL_n[j][m]=-math.log(globalConnectionProbability)
                        LL_n_total[j] = LL_n_total[j]+LL_n[j][m]
                    else: #there is no edge between the nth and the mth nodes
                        LL_n[j][m]=-math.log(1-globalConnectionProbability)
                        LL_n_total[j] = LL_n_total[j]+LL_n[j][m]

            #optimize the angular position of the nth node
            bestTrialIndex = np.argmin(LL_n_total)
            #if bestTrialIndex == numberOfAnglesInSwappingRounds: the last element of LL_n_total is the smallest, i.e. the current angular coordinate of the nth node is the best -> nothing changes
            if bestTrialIndex<numberOfAnglesInSwappingRounds: #the angle of the nth node must be modified
                angularCoords[n] = anglesToTry[bestTrialIndex] #update the position of the nth node
                bestL = LL_n_total[bestTrialIndex] #the total contribution of the nth node to the logarithmic loss at its new position
                changeInLL = changeInLL + (bestL - LL_list[n])
                LL_list[n] = bestL #update the total contribution of the nth node to the logarithmic loss
                for m in otherNodes: #update the contribution of the other nodes to the logarithmic loss
                    LL_list[m]=LL_list[m]-LLmatrix[n][m]+LL_n[bestTrialIndex][m] #update the total contribution of the mth node to the logarithmic loss
                    #update the contribution of the n-m node pair to the logarithmic loss:
                    LLmatrix[n][m]=LL_n[bestTrialIndex][m]
                    LLmatrix[m][n]=LL_n[bestTrialIndex][m]

                #update the angular position of the neighbors of the involved nodes and the angular node order
                if ((firstNeighborsPosition[n][0]+(2*math.pi-angularCoords[n]))%(2*math.pi))<math.pi: #the nth node was swapped with its first clockwise neighbor
                    indexOfn=np.where(angularNodeOrder==n)[0][0] #the ordinal number of the node that is the nth in nodeList in the angular node order
                    neighbor_m3=angularNodeOrder[indexOfn-3] #the identifier of the former third clockwise neighbor of the nth node
                    neighbor_m2=angularNodeOrder[indexOfn-2] #the identifier of the former second clockwise neighbor of the nth node
                    neighbor_m1=angularNodeOrder[indexOfn-1] #the identifier of the former first clockwise neighbor of the nth node which was swapped with the nth node
                    neighbor_p1=angularNodeOrder[(indexOfn+1)%N] #the identifier of the former first counterclockwise neighbor of the nth node
                    neighbor_p2=angularNodeOrder[(indexOfn+2)%N] #the identifier of the former second counterclockwise neighbor of the nth node

                    secondNeighborsPosition[neighbor_p2][0]=angularCoords[neighbor_m1] #the second clockwise neighbor of the former second counterclockwise neighbor of the nth node becomes that node with which node n was swapped,
                    secondNeighborsPosition[neighbor_p1][0]=angularCoords[n] #the second clockwise neighbor of the former first counterclockwise neighbor of the nth node becomes node n,
                    firstNeighborsPosition[neighbor_p1][0]=angularCoords[neighbor_m1] #the first clockwise neighbor of the former first counterclockwise neighbor of the nth node becomes that node with which node n was swapped,
                    secondNeighborsPosition[neighbor_m1][1]=angularCoords[neighbor_p2] #the second counterclockwise neighbor of that node with which node n was swapped becomes the former second counterclockwise neighbor of the nth node, etc.
                    firstNeighborsPosition[neighbor_m1][1]=angularCoords[neighbor_p1]
                    firstNeighborsPosition[neighbor_m1][0]=angularCoords[n]
                    secondNeighborsPosition[neighbor_m1][0]=angularCoords[neighbor_m2]
                    secondNeighborsPosition[neighbor_m2][1]=angularCoords[neighbor_m1]
                    firstNeighborsPosition[neighbor_m2][1]=angularCoords[n]
                    secondNeighborsPosition[neighbor_m3][1]=angularCoords[n]
                    firstNeighborsPosition[n][0]=angularCoords[neighbor_m2]
                    firstNeighborsPosition[n][1]=angularCoords[neighbor_m1]
                    secondNeighborsPosition[n][0]=angularCoords[neighbor_m3]
                    secondNeighborsPosition[n][1]=angularCoords[neighbor_p1]

                    #update the angular node order
                    angularNodeOrder[indexOfn]=angularNodeOrder[indexOfn-1]
                    angularNodeOrder[indexOfn-1]=n

                elif ((firstNeighborsPosition[n][1]+(2*math.pi-angularCoords[n]))%(2*math.pi))>=math.pi: #node n was swapped with its first counterclockwise neighbor
                    indexOfn=np.where(angularNodeOrder==n)[0][0] #the ordinal number of the node that is the nth in nodeList in the angular node order
                    neighbor_m2=angularNodeOrder[indexOfn-2] #the identifier of the former second clockwise neighbor of the nth node
                    neighbor_m1=angularNodeOrder[indexOfn-1] #the identifier of the former first clockwise neighbor of the nth node
                    neighbor_p1=angularNodeOrder[(indexOfn+1)%N] #the identifier of the former first counterclockwise neighbor of the nth node
                    neighbor_p2=angularNodeOrder[(indexOfn+2)%N] #the identifier of the former second counterclockwise neighbor of the nth node
                    neighbor_p3=angularNodeOrder[(indexOfn+3)%N] #the identifier of the former third counterclockwise neighbor of the nth node

                    secondNeighborsPosition[neighbor_m2][1]=angularCoords[neighbor_p1]
                    secondNeighborsPosition[neighbor_m1][1]=angularCoords[n]
                    firstNeighborsPosition[neighbor_m1][1]=angularCoords[neighbor_p1]
                    secondNeighborsPosition[neighbor_p1][0]=angularCoords[neighbor_m2]
                    firstNeighborsPosition[neighbor_p1][0]=angularCoords[neighbor_m1]
                    firstNeighborsPosition[neighbor_p1][1]=angularCoords[n]
                    secondNeighborsPosition[neighbor_p1][1]=angularCoords[neighbor_p2]
                    secondNeighborsPosition[neighbor_p2][0]=angularCoords[neighbor_p1]
                    firstNeighborsPosition[neighbor_p2][0]=angularCoords[n]
                    secondNeighborsPosition[neighbor_p3][0]=angularCoords[n]
                    firstNeighborsPosition[n][1]=angularCoords[neighbor_p2]
                    firstNeighborsPosition[n][0]=angularCoords[neighbor_p1]
                    secondNeighborsPosition[n][1]=angularCoords[neighbor_p3]
                    secondNeighborsPosition[n][0]=angularCoords[neighbor_m1]

                    #update the angular node order
                    angularNodeOrder[indexOfn]=angularNodeOrder[(indexOfn+1)%N]
                    angularNodeOrder[(indexOfn+1)%N]=n
                    
                else: #no change in the angular node order
                    indexOfn=np.where(angularNodeOrder==n)[0][0] #the ordinal number of the node that is the nth in nodeList in the angular node order
                    neighbor_m2=angularNodeOrder[indexOfn-2] #the identifier of the former second clockwise neighbor of the nth node
                    neighbor_m1=angularNodeOrder[indexOfn-1] #the identifier of the former first clockwise neighbor of the nth node
                    neighbor_p1=angularNodeOrder[(indexOfn+1)%N] #the identifier of the former first counterclockwise neighbor of the nth node
                    neighbor_p2=angularNodeOrder[(indexOfn+2)%N] #the identifier of the former second counterclockwise neighbor of the nth node

                    firstNeighborsPosition[neighbor_m1][1]=angularCoords[n]
                    firstNeighborsPosition[neighbor_p1][0]=angularCoords[n]
                    secondNeighborsPosition[neighbor_m2][1]=angularCoords[n]
                    secondNeighborsPosition[neighbor_p2][0]=angularCoords[n]
        LLglobal_eachRound[i+1] = LLglobal_eachRound[0]+changeInLL


    for i in range(numberOfNonswappingRounds): #perform the non-swapping rounds
        print(str(i+1)+". nonswapping round")
        for n in range(N): #iterate over all nodes
            #determine the angular coordinates to be tested for the node that is the nth in nodeList
            angularRange=math.pi-math.fabs(math.pi-math.fabs(firstNeighborsPosition[n][0]-firstNeighborsPosition[n][1])) #the angular distance between the first neighbors of the nth node; it is the width of the explored angular range
            angleStep=angularRange/(numberOfAnglesInNonswappingRounds+1) #angular distance between the equidistantly distributed test positions; the borders of the explored angular range are not investigated
            anglesToTry=np.zeros(numberOfAnglesInNonswappingRounds) #storing the tested angular positions of the nth node
            for a in range(numberOfAnglesInNonswappingRounds):
                anglesToTry[a]=(firstNeighborsPosition[n][0]+(a+1)*angleStep)%(2*math.pi)

            #calculate the logaritmic losses related to node n for the tested angular positions
            otherNodes = list(range(N))
            otherNodes.remove(n)
            LL_n = np.zeros((numberOfAnglesInNonswappingRounds,N)) #LL_n[j][m] is the contribution of the n-m node pair to the logarithmic loss when node n is placed at the jth test position
            LL_n_total = np.zeros(numberOfAnglesInNonswappingRounds+1) #LL_n_total[j] is the total contribution of the nth node to the logarithmic loss when it is placed at the jth test position
            #the last element of the array LL_n_total stores the current total contribution of the nth node (at its current angular position), thus if none of the new angles are better than the current angle, the current can be kept
            LL_n_total[-1]=LL_list[n]
            for j in range(numberOfAnglesInNonswappingRounds):
                for m in otherNodes:
                    h = hypDist(radialCoords[n],anglesToTry[j],radialCoords[m],angularCoords[m]) #hyperbolic distance between node n and m
                    globalConnectionProbability = 1/(1+math.exp((h-R)/(2*T)))
                    if (nodeList[n],nodeList[m]) in edgeList or (nodeList[n],nodeList[m]) in edgeList: #there is an edge between node n and m
                        LL_n[j][m]=-math.log(globalConnectionProbability)
                        LL_n_total[j] = LL_n_total[j]+LL_n[j][m]
                    else: #there is no edge between node i and j
                        LL_n[j][m]=-math.log(1-globalConnectionProbability)
                        LL_n_total[j] = LL_n_total[j]+LL_n[j][m]

            #optimize the angular position of the nth node
            bestTrialIndex = np.argmin(LL_n_total)
            #if bestTrialIndex == numberOfAnglesInNonswappingRounds: the last element of LL_n_total is the smallest, i.e. the current angular coordinate of the nth node is the best -> nothing changes
            if bestTrialIndex<numberOfAnglesInNonswappingRounds: #the angle of the nth node must be modified
                angularCoords[n] = anglesToTry[bestTrialIndex] #update the position of the nth node
                bestL = LL_n_total[bestTrialIndex] #the total contribution of the nth node to the logarithmic loss at its new position
                changeInLL = changeInLL + (bestL - LL_list[n])
                LL_list[n] = bestL #update the total contribution of the nth node to the logarithmic loss
                for m in otherNodes: #update the contribution of the other nodes to the logarithmic loss
                    LL_list[m]=LL_list[m]-LLmatrix[n][m]+LL_n[bestTrialIndex][m] #update the total contribution of node m to the logarithmic loss
                    #update the contribution of the n-m node pair to the logarithmic loss:
                    LLmatrix[n][m]=LL_n[bestTrialIndex][m]
                    LLmatrix[m][n]=LL_n[bestTrialIndex][m]

                #update the angular position of the neighbors of the involved nodes
                indexOfn=np.where(angularNodeOrder==n)[0][0] #the ordinal number of the node that is the nth in nodeList in the angular node order
                neighbor_m2=angularNodeOrder[indexOfn-2] #the identifier of the second clockwise neighbor of the nth node
                neighbor_m1=angularNodeOrder[indexOfn-1] #the identifier of the first clockwise neighbor of the nth node which was swapped with node n
                neighbor_p1=angularNodeOrder[(indexOfn+1)%N] #the identifier of the first counterclockwise neighbor of the nth node
                neighbor_p2=angularNodeOrder[(indexOfn+2)%N] #the identifier of the second counterclockwise neighbor of the nth node
                
                firstNeighborsPosition[neighbor_m1][1]=angularCoords[n]
                firstNeighborsPosition[neighbor_p1][0]=angularCoords[n]
                secondNeighborsPosition[neighbor_m2][1]=angularCoords[n]
                secondNeighborsPosition[neighbor_p2][0]=angularCoords[n]
        LLglobal_eachRound[numberOfSwappingRounds+i+1] = LLglobal_eachRound[0]+changeInLL
    
    return [angularCoords,LLglobal_eachRound]



#This function weights the edges in graph G according to the hyperbolic distance between the connected nodes. Less similarity of two connected nodes results in larger hyperbolic distance between them on the hyperbolic disk, thereby smaller edge weight. This weighting may promote community detection.
#G is a NetworkX graph
#radialCoords is a NumPy array with the radial coordinates of the network nodes on the hyperbolic disk
#angularCoords is a NumPy array with the angular coordinates of the network nodes on the hyperbolic disk
#Example for function call:
#       embedding.weighting_hypDist(G,radialCoords,angularCoords)
def weighting_hypDist(G,radialCoords,angularCoords):
    nodeList = list(G.nodes)
    for (i,j) in G.edges():
        node1ID = nodeList.index(i) #the position of the node named i in the nodeList
        node2ID = nodeList.index(j) #the position of the node named j in the nodeList
        h = hypDist(radialCoords[node1ID],angularCoords[node1ID],radialCoords[node2ID],angularCoords[node2ID])
        G[i][j]['weight'] = 1/(1+h) #assign a weight to the i-j edge



#A function for saving the radial and the angular coordinates of the nodes for an embedded graph.
#Each line is of the form
#nodeID   radialCoord  angularCoord
#where id is the node identifier, angularCoord is the angular and radialCoord is the radial coordinate of this node.
#nodeList is a list of the node identifiers
#radialCoords,angularCoords=NumPy arrays lists with the node coordinates, where the ith element is the coordinate of the ith node in nodeList
#pathString is a string of the path where we want to save the file with the coordinates
def saveCoordinates(nodeList,radialCoords,angularCoords,pathString):
    fileHandler=open(pathString,"w")
    for i in range(len(nodeList)):
        fileHandler.write(str(nodeList[i]) + "\t") #node identifier
        fileHandler.write(str(radialCoords[i])+"\t") #radial coordinate of the node
        fileHandler.write(str(angularCoords[i]) + "\n") #angular coordinate of the node
    fileHandler.close()




#A function for determining the center and the radius of the circle that passes 3 points given by the two-dimensional Cartesian coordinate arrays p1, p2 and p3.
#If the 3 points form a line, the function returns (None, infinity).
def define_circle(p1, p2, p3):
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return (None, np.inf)

    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return ((cx, cy), radius)


#A function for plotting the embedded graph on the native representation of the hyperbolic plane.
#pathString is a string of the path where we want to save the layout
#nodeList is a list of the node identifiers
#edgeList is a list of tuples corresponding to connected node pairs.
#N is the number of nodes in the graph to be plotted
#r is a NumPy array containing the radial coordinates of the networks nodes, where the ith element is the coordinate of the ith node in nodeList
#Theta is a NumPy array containing the angular coordinates of the networks nodes, where the ith element is the coordinate of the ith node in nodeList
#commStructList is a list of N elements, where the ith element is the identifier of the community to which the ith node in nodeList belongs. Note that it is assumed that the community identifiers have already been converted to 0,1,...,number of communities-1. If the community structure of the network is not known, set commStructList to a list of N number of zeros - this way all the nodes and egdes will have the same color.
#rmax is the occurring largest radial coordinate
def PoincDisk(pathString, nodeList, edgeList, N, r, Theta, commStructList, rmax):
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes()
    ax.set_aspect('equal')
    numberOfGroups = len(set(commStructList))
    cm_subsection = np.linspace(0.0, 1.0, numberOfGroups) #the colors will be determined by the community identifiers
    colorMap = [cm.Paired(i) for i in cm_subsection] #List of colors in (R, G, B, alpha) form. The color values are normalized by a division with 255. Smaller alpha (in [0,1]) means more transparency.
    x = np.multiply(r, np.cos(Theta)) # Cartesian coordinates of the nodes (the ith element is the coordinate of the ith node in nodeList)
    y = np.multiply(r, np.sin(Theta))
    CartCoord = np.transpose(np.array([x,y])) # first column=x coordinates, second column=y coordinates
    for nodeID in range(N): #plot the nodes in the order dictated by nodeList
        ax.plot(x[nodeID], y[nodeID], linestyle='', marker='o', color=colorMap[commStructList[nodeID]], markeredgecolor=colorMap[commStructList[nodeID]], markersize=2, zorder=2)
    for j in range(len(edgeList)): #plot the edges
        node1name = edgeList[j][0]
        node2name = edgeList[j][1]
        node1ID = nodeList.index(node1name) #the position of the node named node1name in the nodeList
        node2ID = nodeList.index(node2name) #the position of the node named node2name in the nodeList
        if commStructList[node1ID]==commStructList[node2ID]: #if 2 nodes belong to the same community, the color of the edge between them will be the same as the color corresponding to the given community
            edgeColor = (colorMap[commStructList[node1ID]][0],colorMap[commStructList[node1ID]][1],colorMap[commStructList[node1ID]][2],0.6)
        else: #the 2 nodes belong to different communities
            edgeColor = (0,0,0,0.3) #transparent black
        x1 = x[node1ID]
        y1 = y[node1ID]
        x2 = x[node2ID]
        y2 = y[node2ID]
        if r[node1ID]!=0:
            thirdPoint = (rmax*rmax*x1/(r[node1ID]*r[node1ID]),rmax*rmax*y1/(r[node1ID]*r[node1ID])) #coordinates of a point that lies on the arc connecting the two examined network nodes
        if r[node1ID]==0 or abs((x1-x2)*(y2-thirdPoint[1])-(x2-thirdPoint[0])*(y1-y2))<1.0e-6: #the two network nodes have to be connected with a straight line instead of an arc
            ax.plot([x1,x2],[y1,y2],color=edgeColor,linestyle='-',linewidth=0.1,zorder=1)
        else: #the two network nodes have to be connected with a hyperbolic line, i.e. an arc
            ((x0,y0),r0) = define_circle((x1,y1),(x2,y2),thirdPoint)
            fi1 = 360*((math.atan2(y1-y0,x1-x0)+2*math.pi)%(2*math.pi))/(2*math.pi)
            fi2 = 360*((math.atan2(y2-y0,x2-x0)+2*math.pi)%(2*math.pi))/(2*math.pi)
            if (max(fi1,fi2)-min(fi1,fi2))<(min(fi1,fi2)-(max(fi1,fi2)-360)):
                hyperline = patches.Arc((x0,y0),2*r0,2*r0,0,min(fi1,fi2),max(fi1,fi2),edgecolor=edgeColor,linewidth=0.1,zorder=1) #draw an elliptical arc
                ax.add_patch(hyperline)
            else:
                hyperline = patches.Arc((x0,y0),2*r0,2*r0,0,max(fi1,fi2),min(fi1,fi2),edgecolor=edgeColor,linewidth=0.1,zorder=1) #draw an elliptical arc
                ax.add_patch(hyperline)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    ax.tick_params(length=2,width=1)
    plt.xlim((-rmax*1.1,rmax*1.1))
    plt.ylim((-rmax*1.1,rmax*1.1))
    #plt.axis('off')
    fig.savefig(pathString,bbox_inches="tight",dpi=100)
    plt.close(fig)



#x and y are NumPy arrays containing the Cartesian coordinates of the network nodes in the native representation of the hyperbolic plane. The ith element of each array (i=0,1,...,N-1) is the coordinate of the node that is the ith according to the order in G.nodes.
def convertToPolar(x,y): #conversion from Cartesian to polar coordinates
    r = np.sqrt(np.multiply(x,x)+np.multiply(y,y)) #radial coordinates of the nodes
    phi = np.mod(np.arctan2(y,x)+2*math.pi,2*math.pi) #phi in [0,2pi], measured in the x-y plane from the x axis
    return [r,phi]

#r and phi are NumPy arrays containing the polar coordinates of the network nodes in the native representation of the hyperbolic plane. The ith element of each array (i=0,1,...,N-1) is the coordinate of the node that is the ith according to the order in G.nodes.
def convertToCartesian(r,phi):
    x = np.multiply(r,np.cos(phi))
    y = np.multiply(r,np.sin(phi))
    return [x,y]
