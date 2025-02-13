import numpy as np
from tensorflores.utils.autocloud.data_cloud_bias import DataCloudBias


class AutoCloudBias:
	c= np.array([DataCloudBias(0)],dtype=DataCloudBias)
	alfa= np.array([0.0],dtype=float)
	intersection = np.zeros((1,1),dtype=int)
	listIntersection = np.zeros((1),dtype=int)
	matrixIntersection = np.zeros((1,1),dtype=int)
	relevanceList = np.zeros((1),dtype=int)
	k=1
	def __init__(self, m):
		AutoCloudBias.m = m
		AutoCloudBias.c= np.array([DataCloudBias(0)],dtype=DataCloudBias)
		AutoCloudBias.alfa= np.array([0.0],dtype=float)
		AutoCloudBias.intersection = np.zeros((1,1),dtype=int)
		AutoCloudBias.listIntersection = np.zeros((1),dtype=int)
		AutoCloudBias.relevanceList = np.zeros((1),dtype=int)
		AutoCloudBias.matrixIntersection = np.zeros((1,1),dtype=int)
		AutoCloudBias.classIndex = []
		AutoCloudBias.k = 1		

	def mergeClouds(self):
		i=0
		while(i<len(AutoCloudBias.listIntersection)-1):
			merge=False
			j=i+1
			while(j<len(AutoCloudBias.listIntersection)):
				if(AutoCloudBias.listIntersection[i] == 1 and AutoCloudBias.listIntersection[j] == 1):
					AutoCloudBias.matrixIntersection[i,j] = AutoCloudBias.matrixIntersection[i,j] + 1;
				nI = AutoCloudBias.c[i].n
				nJ = AutoCloudBias.c[j].n
				meanI = AutoCloudBias.c[i].mean
				meanJ = AutoCloudBias.c[j].mean
				varianceI = AutoCloudBias.c[i].variance
				varianceJ = AutoCloudBias.c[j].variance
				nIntersc = AutoCloudBias.matrixIntersection[i,j]
				if (nIntersc > (nI - nIntersc) or nIntersc > (nJ - nIntersc)):
					merge = True
					#update values
					n = nI + nJ - nIntersc
					mean = ((nI * meanI) + (nJ * meanJ))/(nI + nJ)
					variance = ((nI - 1) * varianceI + (nJ - 1) * varianceJ)/(nI + nJ - 2)
					newCloud = DataCloudBias(mean)
					newCloud.updateDataCloud(n,mean,variance)
					#updating intersection list
					AutoCloudBias.listIntersection = np.concatenate((AutoCloudBias.listIntersection[0 : i], np.array([1]), AutoCloudBias.listIntersection[i + 1 : j],AutoCloudBias.listIntersection[j + 1 : np.size(AutoCloudBias.listIntersection)]),axis=None)
					#updating data clouds list 
					AutoCloudBias.c = np.concatenate((AutoCloudBias.c[0 : i ], np.array([newCloud]), AutoCloudBias.c[i + 1 : j],AutoCloudBias.c[j + 1 : np.size(AutoCloudBias.c)]),axis=None)
					#update  intersection matrix
					M0 = AutoCloudBias.matrixIntersection
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
					AutoCloudBias.matrixIntersection = M1
				j += 1
			if(merge):
				i = 0
			else:
				i += 1
				
	def run(self,X):
		AutoCloudBias.listIntersection = np.zeros((np.size(AutoCloudBias.c)),dtype=int)
		if AutoCloudBias.k==1:
			AutoCloudBias.c[0]=DataCloudBias(X)
			AutoCloudBias.classIndex.append(0)
		elif AutoCloudBias.k==2:
			AutoCloudBias.c[0].addDataClaud(X)
			AutoCloudBias.classIndex.append(0)
		elif AutoCloudBias.k>=3:
			i=0
			createCloud = True
			AutoCloudBias.alfa = np.zeros((np.size(AutoCloudBias.c)),dtype=float)
			for data in AutoCloudBias.c:
				n= data.n +1
				mean = ((n-1)/n)*data.mean + (1/n)*X
				variance = ((n-1)/n)*data.variance +(1/n)*((np.linalg.norm(X-mean))**2)
				epsilon = 1e-8  # Valor pequeno para evitar divisão por zero
				eccentricity = (1 / n) + ((mean - X).T.dot(mean - X)) / (n * (variance + epsilon))
				typicality = 1 - eccentricity
				norm_eccentricity = eccentricity/2
				norm_typicality = typicality/(AutoCloudBias.k-2)
				data.eccAn = eccentricity
				if(norm_eccentricity<=(AutoCloudBias.m**2 +1)/(2*n)):
					data.updateDataCloud(n,mean,variance)
					AutoCloudBias.alfa[i] = norm_typicality
					createCloud= False
					AutoCloudBias.listIntersection.itemset(i,1)
				else:
					AutoCloudBias.alfa[i] = 0
					AutoCloudBias.listIntersection.itemset(i,0)
				i+=1
			
			if(createCloud):
				AutoCloudBias.c = np.append(AutoCloudBias.c,DataCloudBias(X))
				AutoCloudBias.listIntersection = np.insert(AutoCloudBias.listIntersection,i,1)
				AutoCloudBias.matrixIntersection = np.pad(AutoCloudBias.matrixIntersection, ((0,1),(0,1)), 'constant', constant_values=(0)) 
			self.mergeClouds()
			#AutoCloudBias.relevanceList = AutoCloudBias.alfa /np.sum(AutoCloudBias.alfa)

			if AutoCloudBias.alfa.size > 0:
				total_sum = np.sum(AutoCloudBias.alfa)
				if total_sum != 0:
					AutoCloudBias.relevanceList = AutoCloudBias.alfa / total_sum
				else:
					AutoCloudBias.relevanceList = np.zeros_like(AutoCloudBias.alfa)  # Handle zero-sum case
			else:
				print("Error: alfa is empty.")

			classIndex = np.argmax(AutoCloudBias.relevanceList)
			AutoCloudBias.classIndex.append(classIndex)

        
		AutoCloudBias.k=AutoCloudBias.k+1