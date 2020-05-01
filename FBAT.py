import time as times
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import auc

class FBAT:
	
	data_train_ok = None
	models_train_ok = None
	dist_train_ok = None
	
	data_train_ko = None
	models_train_ko = None
	dist_train_ko = None
	
	data_test_ok = None
	models_test_ok = None
	dist_test_ok = None
	
	data_test_ko = None
	models_test_ko = None
	dist_test_ko = None
	
	threshold = None
	timeTrain = 0
	timeTest = 0
	
	def __init__(self):
		pass;
		
	def set_train(self,ok,ko):
		self.data_train_ok = [a for a in ok]
		self.data_train_ko = [a for a in ko]
		
	def set_test(self,ok,ko):
		self.data_test_ok = [a for a in ok]
		self.data_test_ko = [a for a in ko]
		
	@staticmethod
	def compute_harmonics(TS):
		take = int(len(TS)/2) +1 
		Y = np.fft.fft(TS)
		return Y[:take]
	
	@staticmethod	
	def compute_model(HAR,T,z=1,delay=0):
		nbOfHarmonics = len(HAR) # harmonic 0 included
		harmonics = HAR[:(z+1)]
		freqs = range(z+1)
		out = []
		for i in T:
			time = 0
			for j in range(len(harmonics)):
				if j==0:
					time = time + (abs(harmonics[j])/24.)*np.cos(2*np.pi*i*freqs[j] + np.angle(harmonics[j]) - delay)
				elif j==(nbOfHarmonics-1):
					time = time + (abs(harmonics[j])/24.)*np.cos(2*np.pi*i*freqs[j] + np.angle(harmonics[j]) - delay)
				else:
					time = time + (abs(harmonics[j])/12.)*np.cos(2*np.pi*i*freqs[j] + np.angle(harmonics[j]) - delay)
			out.append(time)
		return [a for a in out]

        # TS (train or test) sequence of length p + q
        # p is number of elements in one period observation 
	# q is the offset of the second curve
	# z is the i-th harmonics to be kept	
	@staticmethod	
	def computeModels(TS,T,p=24,q=12,z=1):
		TS1 = TS[:p]
		TS2 = TS[q:q+p]
		
		HAR1 = FBAT.compute_harmonics(TS1)
		HAR2 = FBAT.compute_harmonics(TS2)
		
		MOD1 = FBAT.compute_model(HAR1,T,z,0)
		MOD2 = FBAT.compute_model(HAR2,T,z,(float(q)/float(p))*2*np.pi)
		
		return MOD1,MOD2
		
	def get_train_performance(self,threshold):
		nbOK = len(self.dist_train_ok)
		nbKO = len(self.dist_train_ko)
		y_pred = []
		y_true = []
		
		for e in self.dist_train_ok:
			y_true.append(0)
			if e>threshold:
				y_pred.append(1)
			else:
				y_pred.append(0)
		for e in self.dist_train_ko:
			y_true.append(1)
			if e>threshold:
				y_pred.append(1)
			else:
				y_pred.append(0)
				
		accG = accuracy_score(y_true,y_pred)
		precG = precision_score(y_true,y_pred,average='macro')
		recG = recall_score(y_true,y_pred,average='macro')
		labels = [0,1]	
		prec = precision_score(y_true,y_pred,average=None,labels=labels)
		rec = recall_score(y_true,y_pred,average=None,labels=labels)	
		
		return accG,precG,recG,prec[0],rec[0],prec[1],rec[1]
		
	def get_test_performance(self, threshold=None):
		nbOK = len(self.dist_test_ok)
		nbKO = len(self.dist_test_ko)
		y_pred = []
		y_true = []
		TN = 0
		TP = 0
		FN = 0
		FP = 0

		if threshold is None:
			threshold = self.threshold
		
		for e in self.dist_test_ok:
			y_true.append(0)
			if e>threshold:
				y_pred.append(1)
				FP = FP +1 
			else:
				y_pred.append(0)
				TN = TN +1
		for e in self.dist_test_ko:
			y_true.append(1)
			if e>threshold:
				y_pred.append(1)
				TP = TP +1
			else:
				y_pred.append(0)
				FN = FN +1
				
		accG = accuracy_score(y_true,y_pred)
		precG = precision_score(y_true,y_pred,average='macro')
		recG = recall_score(y_true,y_pred,average='macro')
		labels = [0,1]	
		prec = precision_score(y_true,y_pred,average=None,labels=labels)
		rec = recall_score(y_true,y_pred,average=None,labels=labels)	
		
		return accG,precG,recG,prec[0],rec[0],prec[1],rec[1],TP,FN,TN,FP
		
	def train_threshold(self):
	
		mini = min([min(self.dist_train_ok),min(self.dist_train_ko)])
		maxi = max([max(self.dist_train_ok),max(self.dist_train_ko)])

		thresholds = np.linspace(mini,maxi,10000)

		bestThres = 0
		bestInd = 0

		for i in range(len(thresholds)):
			sc = self.get_train_performance(thresholds[i])[2] ##acc
			if sc > bestThres:
				bestThres = sc
				bestInd = i

		self.threshold = thresholds[bestInd]

	def get_auc(self, train = True):
		if train:
			mini = min([min(self.dist_train_ok),min(self.dist_train_ko)])
			maxi = max([max(self.dist_train_ok),max(self.dist_train_ko)])
		else:
			mini = min([min(self.dist_test_ok),min(self.dist_test_ko)])
			maxi = max([max(self.dist_test_ok),max(self.dist_test_ko)])
                      
		thresholds = np.linspace(mini,maxi,10000)

		bestThres = 0
		bestInd = 0

		tpr = []
		fpr = []

		for i in range(len(thresholds)):
			if train: 
                                res = self.get_train_performance(thresholds[i])
                                tpr_loc, fpr_loc  = res[4], res[6]
			else:
				res = self.get_test_performance(thresholds[i])
				tpr_loc, fpr_loc  = res[4], res[6]
			tpr.append(tpr_loc)
			fpr.append(fpr_loc)

		return auc(fpr, tpr)

	# z is the i-th harmonics to be kept
	# p is number of elements in one period observation 
	# q is the offset of the second curve
	def fit(self,p=24,q=12,z=1,threshold=None):
		start = times.time()
		
		T = np.linspace(0,1,p)
			
		self.models_train_ok = []
		self.dist_train_ok = []
		self.models_train_ko = []
		self.dist_train_ko = []
		
		for ts in self.data_train_ok:
			MOD1,MOD2 = FBAT.computeModels(ts,T,p,q,z)
			self.models_train_ok.append([MOD1,MOD2])
			distance = sum([(MOD2[i]-MOD1[i])**2 for i in range(len(MOD1))])**(0.5) ##Euclidian distance
			self.dist_train_ok.append(distance)

		for ts in self.data_train_ko:
			MOD1,MOD2 = FBAT.computeModels(ts,T,p,q,z)
			self.models_train_ko.append([MOD1,MOD2])
			distance = sum([(MOD2[i]-MOD1[i])**2 for i in range(len(MOD1))])**(0.5) ##Euclidian distance
			self.dist_train_ko.append(distance)
		
		if threshold:
			self.threshold == threshold
		else:	
			self.train_threshold()
			
			
		stop = times.time()
		self.timeTrain = stop-start
		
	
	def eval_test(self,p=24,q=12,z=1,threshold=None):
		start = times.time()
		
		if threshold:
			self.threshold = threshold
		
		T = np.linspace(0,1,p)
			
		self.models_test_ok = []
		self.dist_test_ok = []
		self.models_test_ko = []
		self.dist_test_ko = []
			
		for ts in self.data_test_ok:
			MOD1,MOD2 = FBAT.computeModels(ts,T,p,q,z)
			self.models_test_ok.append([MOD1,MOD2])
			distance = sum([(MOD2[i]-MOD1[i])**2 for i in range(len(MOD1))])**(0.5) ##Euclidian distance
			self.dist_test_ok.append(distance)
			
		for ts in self.data_test_ko:
			MOD1,MOD2 = FBAT.computeModels(ts,T,p,q,z)
			self.models_test_ko.append([MOD1,MOD2])
			distance = sum([(MOD2[i]-MOD1[i])**2 for i in range(len(MOD1))])**(0.5) ##Euclidian distance
			self.dist_test_ko.append(distance)
			
		res = self.get_test_performance()
		
		stop = times.time()
		self.timeTest = stop-start
		
		return res,self.threshold,self.timeTrain,self.timeTest
		
	def get_test_dist_OK(self):
		return([a for a in self.dist_test_ok])
		
	def get_test_dist_KO(self):
		return([a for a in self.dist_test_ko])
