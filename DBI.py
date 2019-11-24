'''
Created on Sep 9, 2019

@author: vaishnaviv
'''
import pandas as pd
import numpy as np
import itertools

    
class DBI:
    
    def __init__(self,centroids,clusters,k):
        self.centroids=centroids
        self.clusters=clusters
        self.k=k
        self.s_IntraclusterDistance={}
        self.d_InterclusterDistance={}
        
    '''(Intra-cluster distance) the sum of distances between objects in the same cluster are minimized, 
        (Inter-cluster distance) while the distances between different clusters are maximized '''    
            
    'Calculating intra cluster distances'
    def intra_Cluster_Distance(self):
        for centroidindex in self.centroids.keys():
            for clusterid in self.clusters.keys():
                if centroidindex==clusterid:
                    self.s_IntraclusterDistance[clusterid]=np.average(np.linalg.norm(np.array(self.centroids[centroidindex])-np.array(self.clusters[clusterid]),axis=1))
                    #print(np.array(self.centroids[centroidindex]))
                    #print("clsters",np.array(self.clusters[clusterid]))
                    #print(self.s_IntraclusterDistance[clusterid])
        #print(self.s_IntraclusterDistance)
    'Calculating inter cluster distances'
    def inter_Cluster_Distance(self):
        for cluster_pair1,cluster_pair2 in itertools.product(self.centroids,self.centroids):
            if cluster_pair1!=cluster_pair2:
                euclidean_inter_distances=np.linalg.norm(np.array(self.centroids[cluster_pair1])-np.array(self.centroids[cluster_pair2]))
                self.d_InterclusterDistance[(cluster_pair1,cluster_pair2)]=euclidean_inter_distances
        #print(self.d_InterclusterDistance)
    'Calculating DBI index by calling inter and intra cluster distance calculation modules'              
    def validityMeasure_DaviesBouldin_Index(self):
        
        self.intra_Cluster_Distance()
        self.inter_Cluster_Distance()
        R_ij={}
        R_i={}
        for clusterpairs_key in self.d_InterclusterDistance.keys():
            R_ij[clusterpairs_key]=(self.s_IntraclusterDistance[clusterpairs_key[0]]+self.s_IntraclusterDistance[clusterpairs_key[1]])/self.d_InterclusterDistance[clusterpairs_key]
        for centroid_id in self.centroids:
            rtemp=[]
            for R_ij_keys in R_ij.keys():
                if R_ij_keys[0] == centroid_id:
                    rtemp.append(R_ij[R_ij_keys])
            R_i[centroid_id]=max(rtemp)
        db_index=sum(R_i.values())/self.k
        #print("r_i_j",R_ij,"R_i",R_i)
        return db_index
        
    '''
    'Module to plot the DB index w.r.t to each cluster size from 2 to 10'
    def validity_measure_graph(self):
        plt.plot(validity_measure.keys(), validity_measure.values(),marker='o',)
        plt.plot(validity_tool.keys(),validity_tool.values(),marker='.')
        plt.xlabel('K - Number of clusters')
        plt.ylabel('DaviesBouldin_Index')
        plt.title("Validity Measure - DaviesBouldin_Index")
        plt.show()
    
   '''
if __name__ == "__main__":
    
    
    validity_measure={}
    data_path='BSOM_DataSet_revised.csv'
    #data_path='Synthetic_test_data.csv'
    data=pd.read_csv(data_path)
    'Data normalization'
    data_norm=data[['all_NBME_avg_n4','all_PIs_avg_n131','HD_final']]
    data = data_norm.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    
    
    validity_tool={2: 0.8141395317464027, 3: 0.819358466913027, 4: 0.8539982796451195, 5: 0.9967999811159839, 6: 0.9583471025912981, 7: 1.0105040859835814, 8: 1.1069097097835603, 9: 0.9981179734983576, 10: 1.0835348931997557}
    for k in range(2,11):
        K_Object=KMeans(k)
        K_Object.fit(data,'all_NBME_avg_n4','all_PIs_avg_n131','HD_final')
        #K_Object.fit(data,'x1','x2')
        DBI_object=DBI(K_Object.centroids,K_Object.clusters,k)
        for i in K_Object.centroids:
            print(k,K_Object.centroids[i])

        db_index_validity=DBI_object.validityMeasure_DaviesBouldin_Index()
        validity_measure[k]=db_index_validity
        print(db_index_validity)
    DBI_object.validity_measure_graph()
   