'''
Created on Sep 8, 2019

@author: vaishnaviv
'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import random
import warnings
from Assigment1.DBI import DBI
from Assigment1.FCM import FCM
warnings.filterwarnings("ignore")
import copy


class KMeans:
    'Constructor of K-means class to initialize number of clusters and maximum number of iterations of k-means operations'
    def __init__(self,k,max_iteration=100):
        self.k=k
        self.max_iter=max_iteration
        
        
    '''Fit module to randomly initialize centroids from the dataset points and for each data point of given feature calculate the Euclidiean distances
    and select the data point that is close to centroid and assign it to the nearest centroid cluster.This process is done repetitively untill converge
    condition either no change in centroid or maximum iterations is exhausted'''    
    def fit(self,data,*features):
        random.seed(120)
        self.centroids={}
        self.features=[feature for feature in features]
        'Random initialize of centroids'
        for j in range(self.k):
            randomnumber=random.randint(0,data.shape[0])
            for feature in self.features:
    
                if j+1 in self.centroids.keys():
                    self.centroids[j+1].append(data.loc[randomnumber,feature])
                else:
                    self.centroids[j+1]=[data.loc[randomnumber,feature]]
        'iteration for max number of iterations one of the convergence condition'
        for j in range(self.max_iter):
            self.clusters={}
            self.labels=[]
            
            for i in range(self.k):
                self.clusters[i+1]=[]
             
            'Euclidiean distance calculation'     
            for featureset in data[self.features].values:
                Euclidiean_Distance=[np.linalg.norm(np.array(featureset)-self.centroids[centroidindex],axis=0)for centroidindex in self.centroids.keys()]
                clusterid=np.argmin(Euclidiean_Distance)+1
                self.clusters[clusterid].append(list(featureset))
                self.labels.append(clusterid-1)
            'Store earlier centroids for convergence condition check'
            Earlier_centroid=copy.deepcopy(self.centroids)
            
            'Re-calculated centroids based on new membership assignment'
            for cluster_id in self.clusters.keys():
                self.centroids[cluster_id]=np.array(np.average(self.clusters[cluster_id],axis=0))
            'Initially assume convergence is True'
            Cluster_Converge = True
            
            'Check for convergence condition'
            for centroidIndex in self.centroids.keys():
                convergecount=0
                original_centroid=Earlier_centroid[centroidIndex]
                current_centroid=self.centroids[centroidIndex]
                if np.array_equal(original_centroid,current_centroid):
                    convergecount=convergecount+1
                    if convergecount==self.k:
                        Cluster_Converge=True
                else:
                    Cluster_Converge=False
            
            'if convergence condition is reached break the loop'
            if Cluster_Converge:
                break
    'Module for 3D visualization of clusters for three features'        
    def clusters_visualize(self,Object):
            fig=plt.figure()
            ax = Axes3D(fig)
            colors = 1*['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
            for centroidId in Object.centroids.keys():
                    print("cluster centers during plots",Object.centroids[centroidId][0],Object.centroids[centroidId][1],Object.centroids[centroidId][2])
                    #plt.plot
                    ax.scatter3D(Object.centroids[centroidId][0], Object.centroids[centroidId][1],Object.centroids[centroidId][2],marker="o", color="k", s=30,linewidths=5)
            for clusterId in Object.clusters.keys():
                    c=colors[clusterId]
                    for features in Object.clusters[clusterId]:
                        #print()
                        ax.scatter(features[0],features[1],features[2],marker="x",color=c,s=40,linewidths=4)    
            ax.set_xlabel('all_NBME_avg_n4 - feature1')
            ax.set_ylabel('all_PIs_avg_n131 - feature2')
            ax.set_zlabel('HD_final -  feature3')
            
    
    'Module to plot the DB index w.r.t to each cluster size from 2 to 10'
    def validity_measure_graph(self,validity_measure_X, validity_measure_Y):
        plt.plot(validity_measure_X,validity_measure_Y,marker='o',)
        #plt.plot(validity_tool.keys(),validity_tool.values(),marker='.')
        plt.xlabel('K - Number of clusters')
        plt.ylabel('DaviesBouldin_Index')
        #plt.title("Validity Measure - DaviesBouldin_Index - three features['all_NBME_avg_n4','all_PIs_avg_n131','HD_final']")
        plt.title("Validity Measure - DaviesBouldin_Index - Four features['all_NBME_avg_n4','all_PIs_avg_n131','HD_final','all_PIs_avg_n131']")
        plt.show() 
    
    def validity_measure_graph_345(self,validity_measure_X, validity_measure_Y,validity_measure_X2,validity_measure_Y2,validity_measure_X3,validity_measure_Y3):
         
        plt.plot(validity_measure_X,validity_measure_Y,marker='o',label="DBI - 4 Features")
        plt.plot(validity_measure_X2,validity_measure_Y2,marker='.',label="DBI - 5 Features")
        plt.plot(validity_measure_X3,validity_measure_Y3,marker='>',label="DBI - 3 Features")
        plt.xlabel('K - Number of clusters')
        plt.ylabel('DaviesBouldin_Index')
        plt.title("Validity Measure - DaviesBouldin_Index-For different number of features/Attributes")
        plt.legend()
        plt.show() 

def K_clusters3():   
    Obj=KMeans(3)
    Obj.fit(data,'all_NBME_avg_n4','all_PIs_avg_n131','HD_final')
    Obj.clusters_visualize(Obj)
    print("Kmeans Centroids for k = 2",Obj.centroids)
    plt.title("K-means Cluster Visualization for k = 2")
    plt.show()
    
def K_clusters2_10():
    colors = 100*['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for k in range(2,11):
        K_Object=KMeans(k)
        K_Object.fit(data,'all_NBME_avg_n4','all_PIs_avg_n131','HD_final')
        #print("k=",k,K_Object.labels)
        K_Object.clusters_visualize(K_Object)
        plt.title("Kmeans Cluster Visualization for K = "+str(k))
        plt.savefig("%s_Clusters.png" % k)
    plt.show() 
    
def Davies_Bouldin2_10():
    #validity_measure={}
    K_Object=KMeans(2)
    
    for k in range(2,11):
        K_Object=KMeans(k)
        K_Object.fit(data,'all_NBME_avg_n4','all_PIs_avg_n131','HD_final')
        #K_Object.fit(data,'x1','x2')
        DBI_object=DBI(K_Object.centroids,K_Object.clusters,k)
        #print("centroids",k,K_Object.centroids,"*")
        #for i in K_Object.clusters.keys():
         #   print("cluster",i,K_Object.clusters[i])
        db_index_validity=DBI_object.validityMeasure_DaviesBouldin_Index()
        validity_measure[k]=db_index_validity
        #print(db_index_validity)
    K_Object.validity_measure_graph(validity_measure.keys(),validity_measure.values())
    
def  Davies_Bouldin_4Features():
    data=pd.read_csv(data_path)
    #print("before normalize",data[['all_NBME_avg_n4','all_PIs_avg_n131','HD_final']].describe())
    data_norm=data[['all_NBME_avg_n4','all_PIs_avg_n131','HD_final','all_irats_avg_n34']]
    data = data_norm.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    K_1=KMeans(3)
    K_1.fit(data,'all_NBME_avg_n4','all_PIs_avg_n131','HD_final','all_irats_avg_n34')
    for k in range(2,11):
        K_Object=KMeans(k)
        K_Object.fit(data,'all_NBME_avg_n4','all_PIs_avg_n131','HD_final','all_irats_avg_n34')
        DBI_object=DBI(K_Object.centroids,K_Object.clusters,k)
        #print(k,K_Object.centroids,"*")
        db_index_validity=DBI_object.validityMeasure_DaviesBouldin_Index()
        validity_measure_4feature[k]=db_index_validity
    K_Object.validity_measure_graph(validity_measure_4feature.keys(),validity_measure_4feature.values())
      
def  Davies_Bouldin_5Features():   
    data=pd.read_csv(data_path)
    data_norm=data[['all_NBME_avg_n4','all_PIs_avg_n131','HD_final','all_irats_avg_n34','HA_final']]
    data = data_norm.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    for k in range(2,11):
        K_Object=KMeans(k)
        K_Object.fit(data,'all_NBME_avg_n4','all_PIs_avg_n131','HD_final','all_irats_avg_n34','HA_final')
        #print("centroids for k =",k,K_Object.centroids)
        DBI_object=DBI(K_Object.centroids,K_Object.clusters,k)
        db_index_validity=DBI_object.validityMeasure_DaviesBouldin_Index()
        validity_measure_5feature[k]=db_index_validity
        #print("DBvalue for cluster with 5 feature",k,db_index_validity)
    K_Object.validity_measure_graph(validity_measure_5feature.keys(),validity_measure_5feature.values())
    #print(validity_measure_5feature,"\n",validity_measure_4feature)
    K_Object.validity_measure_graph_345(validity_measure_4feature.keys(),validity_measure_4feature.values(), validity_measure_5feature.keys(),validity_measure_5feature.values(),validity_measure.keys(),validity_measure.values())
    print(validity_measure_5feature)
    
def Fuzzy_c_Kmeans():
    data=pd.read_csv(data_path)
    data_norm=data[['all_NBME_avg_n4','all_PIs_avg_n131','HD_final']]
    data = data_norm.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    
    for k in range(2,11):
        FCM_object=FCM(data,k)
        FCM_object.FCM(FCM_object)
        FCM_object.hard_clustering()
        DBI_object=DBI(FCM_object.centroids,FCM_object.clusters,k)
        db_index_validity=DBI_object.validityMeasure_DaviesBouldin_Index()
        validity_measure_3feature_FCM[k]=db_index_validity
        #print("DBvalue for cluster with 5 feature with k = ",k,db_index_validity)
    plt.plot(validity_measure_3feature_FCM.keys(),validity_measure_3feature_FCM.values(),marker='o',label="FCM - Three features")
    plt.plot(validity_measure.keys(),validity_measure.values(),marker='.',label="Kmeans - Three features")
    plt.xlabel('K - Number of clusters')
    plt.ylabel('DaviesBouldin_Index')
    plt.title("Validity Measure - DaviesBouldin_Index(FCM vs Kmeans)")
    plt.legend()
    plt.show()

def Fuzzy_c():
    data=pd.read_csv(data_path)
    data_norm=data[['all_NBME_avg_n4','all_PIs_avg_n131','HD_final']]
    K_Object=KMeans(2)
    data = data_norm.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    FCM_object=FCM(data,2)
    FCM_object.FCM(FCM_object)
    FCM_object.hard_clustering()
    print("FCM centroids for k = 2 ",FCM_object.centroids)
    K_Object.clusters_visualize(FCM_object)
    plt.title("FCM - Cluster Visualization for k = 2")
    plt.show()
    
def Fuzzy_c_additionalfeature():
    data=pd.read_csv(data_path)
    data_norm=data[['all_NBME_avg_n4','all_PIs_avg_n131','HD_final','all_irats_avg_n34']]
    data = data_norm.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    
    for k in range(2,11):
        FCM_object=FCM(data,k)
        FCM_object.FCM(FCM_object)
        FCM_object.hard_clustering()
        DBI_object=DBI(FCM_object.centroids,FCM_object.clusters,k)
        db_index_validity=DBI_object.validityMeasure_DaviesBouldin_Index()
        validity_measure_4feature_FCM[k]=db_index_validity
        #print("DBvalue for cluster with 5 feature with k = ",k,db_index_validity)
    plt.plot(validity_measure_4feature_FCM.keys(),validity_measure_4feature_FCM.values(),marker='o',label="FCM - four features(additional feature)")
    plt.plot(validity_measure_3feature_FCM.keys(),validity_measure_3feature_FCM.values(),marker='.',label="FCM - Three features")
    plt.xlabel('K - Number of clusters')
    plt.ylabel('DaviesBouldin_Index')
    plt.title("Validity Measure - DaviesBouldin_Index(FCM- initial feature vs FCM - additional features)")
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    data_path='BSOM_DataSet_revised.csv'
    data=pd.read_csv(data_path)
    print("Raw Data",data[['all_NBME_avg_n4','all_PIs_avg_n131','HD_final']].describe())
    'Data Normalization'
    data_norm=data[['all_NBME_avg_n4','all_PIs_avg_n131','HD_final','all_irats_avg_n34','HA_final']]
    data = data_norm.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    print("Normalized Data",data[['all_NBME_avg_n4','all_PIs_avg_n131','HD_final']].describe())
    
    'Question 1 a'
    #K_clusters3()
    'Question 1 b'
    #K_clusters2_10()
    'Question 1 c'
    #validity_measure={}
    #Davies_Bouldin2_10() #best cluster is k =2 DB=0.7990049818342608 0.7929808986627392
    #print(validity_measure)
    '2 a '
    #validity_measure_4feature={}  
    #Davies_Bouldin_4Features()
    #print(validity_measure_4feature)
    '2b'
    #validity_measure_5feature={}  
    #Davies_Bouldin_5Features()
    #print(validity_measure_5feature)
    #test_dbi()
    #test_dbi_fcm()
    '3a best number of clusters from from problem 1 and best features from problem2'
    Fuzzy_c()
    '3b'
   # validity_measure_3feature_FCM={}
   # Fuzzy_c_Kmeans()
    #print("FCM 3 features",validity_measure_3feature_FCM)
    
    '3c'
    #validity_measure_4feature_FCM={}
    #Fuzzy_c_additionalfeature()
    #print("FCM 4 features",validity_measure_4feature_FCM)
   
    
    
    

    
    
    
    


    
        