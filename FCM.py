'''
Created on Sep 13, 2019

@author: vaishnaviv
'''
import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")
import itertools
from Assigment1.DBI import DBI
import matplotlib.pyplot as plt
class FCM:
    def __init__(self,data,k,max_iteration=100,fuzzifier_m=2):
        self.dataset=np.asarray(data)
        self.k=k 
        self.iteration=max_iteration
        self.m=fuzzifier_m
        self.membership_matrix=[[ 0 for i in range(len(self.dataset)) ] for c in range(self.k)]
        self.centroids={}
        self.clusters={}
       
        #print(self.membership_matrix)
    def initialize_membership(self):
        random.seed(120)
        cluster_assigned=[]
        
        for data_column in range(len(self.dataset)):
            if data_column not in cluster_assigned:
                self.membership_matrix[random.randint(0,self.k-1)][data_column]=1
                cluster_assigned.append(data_column)
                

        
        #print("membership",np.array(self.membership_matrix).reshape(self.k,len(self.dataset)))
        
                
    def cluster_centers(self):
        '''Centroid:Any point x has a set of coefficients giving the degree of being in the kth cluster wk(x). 
    With fuzzy c-means, the centroid of a cluster is the mean of all points, weighted by their degree of belonging to the
    cluster, or, mathematically,{\displaystyle c_{k}={{\sum _{x}{w_{k}(x)}^{m}x} \over {\sum _{x}{w_{k}(x)}^{m}}},}
    {\displaystyle c_{k}={{\sum _{x}{w_{k}(x)}^{m}x} \over {\sum _{x}{w_{k}(x)}^{m}}},}''' 
        centroid_matrix=np.nan_to_num(np.matmul((np.power(self.membership_matrix,self.m)),self.dataset)/np.power(self.membership_matrix,self.m).sum(axis=1)[:,None])
        current_centroids_matrix=centroid_matrix
        #print("current centroid \n",current_centroids_matrix) 
        #print("dataset \n",self.dataset)
        return current_centroids_matrix
        
    def update_membership(self,current_centroids_matrix):  
        distances=[]
        for C,X in itertools.product(current_centroids_matrix,self.dataset):
            distances.append(np.linalg.norm(C-X))
            #print(C,X,distances.append(np.linalg.norm(C-X)),np.linalg.norm(C-X))
        distances_matrix=np.array(distances).reshape(self.k,len(self.dataset))
        #print("distance\n",distances_matrix)
        previous_centroids_matrix=current_centroids_matrix
        for d in range(len(self.dataset)):
            for c in range(self.k):
                distance=distances_matrix[c][d]
                #print("distance",distance)
                self.membership_matrix[c][d]=1/(np.sum([[np.power(distance/distances_matrix[c1][d] ,(2/(self.m-1)))] for c1 in range(self.k)]))
        #print("membership",np.array(self.membership_matrix).reshape(self.k,len(self.dataset)))
        if np.sum(self.membership_matrix,axis=0).all() !=1:
            print("invalid membershipvalues")
        
        return previous_centroids_matrix
        
    def FCM(self,FCM_Object):
     
        FCM_Object.initialize_membership() 
        for iterationcount in range(self.iteration): 
            #print("itera",iterationcount)
            old_centroid_matrix=FCM_Object.update_membership(FCM_Object.cluster_centers() )
            new_centroid_matrix=FCM_Object.cluster_centers()
            #print("old \n",old_centroid_matrix,"\n","new \n",new_centroid_matrix)
            #print("membership \n",np.array(self.membership_matrix).reshape(self.k,len(self.dataset)))
            for centroidid in range(self.k):
                self.centroids[centroidid+1]=new_centroid_matrix[centroidid,:] 
            Converge=True 
            #print(old_centroid_matrix==new_centroid_matrix)  
            #print(old_centroid_matrix[0][0],new_centroid_matrix[0][0]) 0.000000000000001
            if np.isclose(old_centroid_matrix,new_centroid_matrix,rtol=0.00001,atol=0.00001).all():
                #print(old_centroid_matrix[0][0],new_centroid_matrix[0][0])
                Converge=True
            else:
                Converge=False
            
            if Converge:
                break                                         
            
    def hard_clustering(self):
        
        hard_cluster_labels=[]
        for datapoints in range(len(self.dataset)):
            hard_cluster_labels.append(np.argmax(np.array(self.membership_matrix).reshape(self.k,len(self.dataset))[:,datapoints]))
        #print(np.array(hard_cluster_labels))
        #print(self.membership_matrix)
        #print(self.dataset)
        #print(self.centroids)
        
        for index,clusterid in enumerate(hard_cluster_labels):
            if clusterid+1 not in self.clusters:
                self.clusters[clusterid+1]=[list(np.array(self.dataset[index,:]))]
            else:
                self.clusters[clusterid+1].append(list(self.dataset[index,:]))  
        
        #print(self.clusters)
        #print(hard_cluster_labels)
if __name__ == "__main__":
    
    
    data_path='BSOM_DataSet_revised.csv'
    data=pd.read_csv(data_path)
    data_norm=data[['all_NBME_avg_n4','all_PIs_avg_n131','HD_final','all_irats_avg_n34','HD_final']]
    data = data_norm.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    #print(data)
    'question 3 a'
    validity_tool={2: 0.8704137935346908, 3: 1.0136207956495478, 4: 1.1304062283424223, 5: 1.459237143652168, 6: 1.4878305632612374, 7: 1.4462734769109218, 8: 1.4689945217823421, 9: 1.4763662232426402, 10: 1.6220083196698105}
    validity_measure_5feature_FCM={}
    for k in range(2,11):
        FCM_object=FCM(data,k)
        FCM_object.FCM(FCM_object)
        FCM_object.hard_clustering()
        #print(FCM_object.centroids)
        DBI_object=DBI(FCM_object.centroids,FCM_object.clusters,k)
        db_index_validity=DBI_object.validityMeasure_DaviesBouldin_Index()
        validity_measure_5feature_FCM[k]=db_index_validity
        print("DBvalue for cluster with 5 feature with k = ",k,db_index_validity)
    print(validity_measure_5feature_FCM)
    plt.plot(validity_measure_5feature_FCM.keys(),validity_measure_5feature_FCM.values(),marker='o',label="Db_FCM")
    plt.plot(validity_tool.keys(),validity_tool.values(),marker='.',label="Db_FCM_tool")
    plt.xlabel('K - Number of clusters')
    plt.ylabel('DaviesBouldin_Index')
    plt.title("Validity Measure - DaviesBouldin_Index using FCM")
    plt.legend()
    plt.show() 
    
'''
 [1 1 1 2 1 1 0 1 0 2 1 1 2 1 2 1 1 1 0 1 1 0 1 2 2 1 2 2 2 0 1 0 2 1 2 2 1
 1 1 1 1 2 0 1 1 2 0 2 1 0 1 1 0 1 2 0 0 1 2 2 1 2 0 2 2 0 0 0 2 1 0 0 2 1
 2 2 1 1 1 1 2 1 1 1 1 2 1 1 1 0 2 1 1 1 2 1 0 2 2 0 2 1 2 2 2 1 1 1 0 1 0
 1 2 1 1]
[[0.3177314  0.32249876 0.35447283] 0 1
 [0.61009032 0.54137646 0.66405427] 1 0
 [0.84269256 0.81612319 0.82947505]] 2 2
 
 [0 0 0 2 0 0 1 0 1 2 0 0 2 0 2 0 0 0 1 0 0 1 0 2 2 0 2 2 2 1 0 1 2 0 2 2 0
 0 0 0 0 2 1 0 0 2 1 2 0 1 0 0 1 0 2 1 1 0 2 2 0 2 1 2 2 1 1 1 2 0 1 1 2 0
 2 2 0 0 0 0 2 0 0 0 0 2 0 0 0 1 2 0 0 0 2 0 1 2 2 1 2 0 2 2 2 0 0 0 1 0 1
 0 2 0 0]
 
 k=3 final {1: array([0.85818182, 0.83406406, 0.84983165]), 2: array([0.62969492, 0.56230722, 0.66591337]), 3: array([0.31443478, 0.31987164, 0.34492754])}
final {1: [[0.9920000000000002, 0.9070077271516119, 0.8222222222222223], [0.83200000000
entroids for k = 2 {1: array([0.43693617, 0.39192476, 0.48841608, 0.34650865, 0.53506698]),
 2: array([0.76717647, 0.72995329, 0.76928105, 0.65821678, 0.74945534])}
 
 [0.77845598 0.74577616 0.77709421 0.66894365 0.76160624]
 [0.45069189 0.4104843  0.5005228  0.36457494 0.54299878]]
 
 [[0.45069148, 0.41048401, 0.50052236, 0.3645747 , 0.54299855],
       [0.77845573, 0.74577584, 0.77709406, 0.66894337, 0.76160607]]
 '''  
    
    