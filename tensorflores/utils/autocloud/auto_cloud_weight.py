import numpy as np
from tensorflores.utils.autocloud.data_cloud_weight import DataCloudWeight

class AutoCloudWeight:
	c= np.array([DataCloudWeight(0)],dtype=DataCloudWeight)
	alfa= np.array([0.0],dtype=float)
	intersection = np.zeros((1,1),dtype=int)
	listIntersection = np.zeros((1),dtype=int)
	matrixIntersection = np.zeros((1,1),dtype=int)
	relevanceList = np.zeros((1),dtype=int)
	k=1
	def __init__(self, m):
		AutoCloudWeight.m = m
		AutoCloudWeight.c= np.array([DataCloudWeight(0)],dtype=DataCloudWeight)
		AutoCloudWeight.alfa= np.array([0.0],dtype=float)
		AutoCloudWeight.intersection = np.zeros((1,1),dtype=int)
		AutoCloudWeight.listIntersection = np.zeros((1),dtype=int)
		AutoCloudWeight.relevanceList = np.zeros((1),dtype=int)
		AutoCloudWeight.matrixIntersection = np.zeros((1,1),dtype=int)
		AutoCloudWeight.classIndex = []
		AutoCloudWeight.k = 1		

	def mergeClouds(self):
		i=0
		while(i<len(AutoCloudWeight.listIntersection)-1):
			merge=False
			j=i+1
			while(j<len(AutoCloudWeight.listIntersection)):
				if(AutoCloudWeight.listIntersection[i] == 1 and AutoCloudWeight.listIntersection[j] == 1):
					AutoCloudWeight.matrixIntersection[i,j] = AutoCloudWeight.matrixIntersection[i,j] + 1;
				nI = AutoCloudWeight.c[i].n
				nJ = AutoCloudWeight.c[j].n
				meanI = AutoCloudWeight.c[i].mean
				meanJ = AutoCloudWeight.c[j].mean
				varianceI = AutoCloudWeight.c[i].variance
				varianceJ = AutoCloudWeight.c[j].variance
				nIntersc = AutoCloudWeight.matrixIntersection[i,j]
				if (nIntersc > (nI - nIntersc) or nIntersc > (nJ - nIntersc)):
					merge = True
					#update values
					n = nI + nJ - nIntersc
					mean = ((nI * meanI) + (nJ * meanJ))/(nI + nJ)
					variance = ((nI - 1) * varianceI + (nJ - 1) * varianceJ)/(nI + nJ - 2)
					newCloud = DataCloudWeight(mean)
					newCloud.updateDataCloud(n,mean,variance)
					#updating intersection list
					AutoCloudWeight.listIntersection = np.concatenate((AutoCloudWeight.listIntersection[0 : i], np.array([1]), AutoCloudWeight.listIntersection[i + 1 : j],AutoCloudWeight.listIntersection[j + 1 : np.size(AutoCloudWeight.listIntersection)]),axis=None)
					#updating data clouds list 
					AutoCloudWeight.c = np.concatenate((AutoCloudWeight.c[0 : i ], np.array([newCloud]), AutoCloudWeight.c[i + 1 : j],AutoCloudWeight.c[j + 1 : np.size(AutoCloudWeight.c)]),axis=None)
					#update  intersection matrix
					M0 = AutoCloudWeight.matrixIntersection
					#remove lines
					M1=np.concatenate((M0[0 : i , :],np.zeros((1,len(M0))),M0[i + 1 : j, :],M0[j + 1 : len(M0), :]))
					#remove columns
					M1=np.concatenate((M1[:, 0 : i ],np.zeros((len(M1),1)),M1[:, i+1 : j],M1[:, j+1 : len(M0)]),axis=1)
					#calculating new column
					col = (M0[:, i] + M0[:, j])*(M0[: , i]*M0[:, j] != 0)
					col = np.concatenate((col[0 : j], col[j + 1 : np.size(col)]))
					#calculating new line
					lin = (M0[i, :]+M0[j, :])*(M0[i, :]*M0[j, :] != 0)
					lin = np.concatenate((lin[ 0 : j], lin[j + 1 : np.size(lin)]))
					#updating column
					M1[:,i]=col
					#updating line
					M1[i,:]=lin
					M1[i, i + 1 : j] = M0[i, i + 1 : j] + M0[i + 1 : j, j].T;   
					AutoCloudWeight.matrixIntersection = M1
				j += 1
			if(merge):
				i = 0
			else:
				i += 1
				
	def run(self,X):
		AutoCloudWeight.listIntersection = np.zeros((np.size(AutoCloudWeight.c)),dtype=int)
		if AutoCloudWeight.k==1:
			AutoCloudWeight.c[0]=DataCloudWeight(X)
			AutoCloudWeight.classIndex.append(0)
		elif AutoCloudWeight.k==2:
			AutoCloudWeight.c[0].addDataClaud(X)
			AutoCloudWeight.classIndex.append(0)
		elif AutoCloudWeight.k>=3:
			i=0
			createCloud = True
			AutoCloudWeight.alfa = np.zeros((np.size(AutoCloudWeight.c)),dtype=float)
			for data in AutoCloudWeight.c:
				n= data.n +1
				mean = ((n-1)/n)*data.mean + (1/n)*X
				variance = ((n-1)/n)*data.variance +(1/n)*((np.linalg.norm(X-mean))**2)
				epsilon = 1e-8  # Valor pequeno para evitar divisão por zero
				eccentricity = (1 / n) + ((mean - X).T.dot(mean - X)) / (n * (variance + epsilon))
				typicality = 1 - eccentricity
				norm_eccentricity = eccentricity/2
				norm_typicality = typicality/(AutoCloudWeight.k-2)
				data.eccAn = eccentricity
				if(norm_eccentricity<=(AutoCloudWeight.m**2 +1)/(2*n)):
					data.updateDataCloud(n,mean,variance)
					AutoCloudWeight.alfa[i] = norm_typicality
					createCloud= False
					AutoCloudWeight.listIntersection.itemset(i,1)
				else:
					AutoCloudWeight.alfa[i] = 0
					AutoCloudWeight.listIntersection.itemset(i,0)
				i+=1
			
			if(createCloud):
				AutoCloudWeight.c = np.append(AutoCloudWeight.c,DataCloudWeight(X))
				AutoCloudWeight.listIntersection = np.insert(AutoCloudWeight.listIntersection,i,1)
				AutoCloudWeight.matrixIntersection = np.pad(AutoCloudWeight.matrixIntersection, ((0,1),(0,1)), 'constant', constant_values=(0)) 
			self.mergeClouds()

			if AutoCloudWeight.alfa.size > 0:
				total_sum = np.sum(AutoCloudWeight.alfa)
				if total_sum != 0:
					AutoCloudWeight.relevanceList = AutoCloudWeight.alfa / total_sum
				else:
					AutoCloudWeight.relevanceList = np.zeros_like(AutoCloudWeight.alfa)  # Handle zero-sum case
			else:
				print("Error: alfa is empty.")


			#AutoCloudWeight.relevanceList = AutoCloudWeight.alfa /np.sum(AutoCloudWeight.alfa)
			classIndex = np.argmax(AutoCloudWeight.relevanceList)
			AutoCloudWeight.classIndex.append(classIndex)

        
		AutoCloudWeight.k=AutoCloudWeight.k+1