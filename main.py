import sys,os
import time as times
import argparse
from FBAT import FBAT



def main(argv):

	start = times.time()

	parser=argparse.ArgumentParser()
	parser.add_argument("--fileTrain",default=None,help="file with TRAIN TS")
	parser.add_argument("--fileTest",default=None,help="file with TEST TS")
	parser.add_argument("--fileOut",default=None,help="file out")
	param=parser.parse_args()

	fileTrain = param.fileTrain
	fileTest = param.fileTest
	fileOut = param.fileOut

	TS_train_ok = []
	TS_train_ko = []
	TS_test_ok = []
	TS_test_ko = []

	#train
	f = open(fileTrain,'r')
	for line in f.readlines():
		TS = [float(a) for a in line.split()[1:]]
		if int(line.split()[0]) == 0:
			TS_train_ok.append(TS)
		if int(line.split()[0]) != 0:
			TS_train_ko.append(TS)
	f.close()
	
	#test
	f = open(fileTest,'r')
	for line in f.readlines():
		TS = [float(a) for a in line.split()[1:]]
		if int(line.split()[0]) == 0:
			TS_test_ok.append(TS)
		if int(line.split()[0]) != 0:
			TS_test_ko.append(TS)
	f.close()
	

	fbat = FBAT()
	print("set train")
	fbat.set_train(TS_train_ok, TS_train_ko)
	print("set test")
	fbat.set_test(TS_test_ok,TS_test_ko)
	
	print("fit train")
	fbat.fit(p=24,q=12,z=1)
	print("eval test")
	results = fbat.eval_test(p=24,q=12,z=1)
	
	f = open(fileOut,'w')
			
	# accG,precG,recG,precOK,recOK,precKO,recKO
	stop = times.time()
	totalTime = stop-start
	
	f.write("Z THRESHOLD TIME_TRAIN TIME_TEST TIME_TOTAL ACC PREC_TOT REC_TOT PREC_OK REC_OK PREC_KO REC_KO AUC_TRAIN AUC_TEST TP TN FP FN\n")
	f.write("%d %f %f %f %f %f %f %f %f %f %f %f %f %f %d %d %d %d\n"%(1,results[1],results[2],results[3],totalTime,results[0][0],results[0][1],results[0][2],results[0][3],results[0][4],results[0][5],results[0][6], fbat.get_auc(True), fbat.get_auc(False),results[0][7],results[0][9],results[0][10],results[0][8]))
	print("%d %f %f %f %f %f %f %f %f %f %f %f %f %f %d %d %d %d\n"%(1,results[1],results[2],results[3],totalTime,results[0][0],results[0][1],results[0][2],results[0][3],results[0][4],results[0][5],results[0][6], fbat.get_auc(True), fbat.get_auc(False),results[0][7],results[0][9],results[0][10],results[0][8]))
	f.close()
	
	
	
if __name__ == '__main__':
	main(sys.argv[1:])
