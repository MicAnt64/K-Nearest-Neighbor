# ML HW 1

import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

#################################################################################
#################################################################################
#####################       DOWNLOAD MODULE     #################################
#################################################################################
#################################################################################

def download_MNIST(MNIST_DATA):
	"""
	This function downloads the compressed files of the MNIST DATA from Yan LeCunn et al, MNIST
	handwritten database. The training set has 60,000 examples, and the test set has 10,000 examples. 
	"""
	# Check if directory exists, if not, create it.
	if not os.path.exists(MNIST_DATA):
		os.makedirs(MNIST_DATA)

	# Check of .gz file exist for training and test data,
	# if not, then download.
	if not os.path.isfile(MNIST_DATA + 'train-images-idx3-ubyte.gz'):
		os.system('wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz' + ' -P  MNIST_DATA')
	if not os.path.isfile(MNIST_DATA + 'train-labels-idx1-ubyte.gz'):
		os.system('wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz' + ' -P  MNIST_DATA')
	if not os.path.isfile(MNIST_DATA + 't10k-images-idx3-ubyte.gz'):
		os.system('wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz' + ' -P  MNIST_DATA')
	if not os.path.isfile(MNIST_DATA + 't10k-labels-idx1-ubyte.gz'):
		os.system('wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz' + ' -P  MNIST_DATA')

	# Unpack files and return: train_img, train_labels, test_img, test_labels

	# Train images
	with gzip.open(MNIST_DATA_LOC + 'train-images-idx3-ubyte.gz') as f:
		f.read(16)
		buf = f.read(28*28*60000)
		f.close()
		train_img = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
		train_img = train_img.reshape(60000, 28, 28)

	# Test images
	with gzip.open(MNIST_DATA_LOC + 't10k-images-idx3-ubyte.gz') as f:
		f.read(16)
		buf = f.read(28*28*10000)
		f.close()
		test_img = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
		test_img = test_img.reshape(10000, 28, 28)
	
	# Training labels
	with gzip.open(MNIST_DATA_LOC + 'train-labels-idx1-ubyte.gz') as f:
		f.read(8)
		buf = f.read(1*60000)
		f.close()
		train_labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
	# Test lables
	with gzip.open(MNIST_DATA_LOC + 't10k-labels-idx1-ubyte.gz') as f:
		f.read(8)
		buf = f.read(1*10000)
		f.close()
		test_labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

	return train_img, test_img, train_labels, test_labels

#################################################################################
#################################################################################
#####################       DOWNLOAD DATA.      #################################
#################################################################################
#################################################################################

MNIST_DATA_LOC = "/Users/michaelantia/Desktop/KNN_HW1/MNIST_DATA/"
train_img, test_img, train_labels, test_labels = download_MNIST(MNIST_DATA=MNIST_DATA_LOC)

train_img = train_img.reshape(60000, 28*28)
test_img  = test_img.reshape(10000, 28*28)

#################################################################################
#################################################################################
################      PLOT DISTRIBUTION OF LABELS    ############################
#################################################################################
#################################################################################

plot_predata = 1

if plot_predata:
	plt.figure(figsize=(10,6))
	plt.subplot(1,2,1)
	plt.hist(train_labels, bins=np.arange(11), align='left', rwidth = 0.8,density=True, label=np.arange(0,10), color='c')
	plt.title('Distribution of Training Set Labels \n' + 'Total Examples: ' + str(train_img.shape[0]))
	plt.xlabel('Training Set labels')
	plt.axis([-0.5,9.5,0,0.2])
	plt.xticks(np.arange(0,10))

	plt.subplot(1,2,2)
	plt.hist(test_labels, bins=np.arange(11), align='left', rwidth = 0.8,density=True, label=np.arange(0,10), color='g')
	plt.title('Distribution of Test Set Labels \n' + 'Total Examples: ' + str(test_img.shape[0]))
	plt.xlabel('Test Set labels')
	plt.axis([-0.5,9.5,0,0.2])
	plt.xticks(np.arange(0,10))
	plt.savefig(MNIST_DATA_LOC + "DataHist.png")

#################################################################################
#################################################################################
################        CREATE A K-NN CLASSIFIER     ############################
#################################################################################
#################################################################################

# SNIPET BELOW ALLOWS IS TO FIND INDECES GIVEN CLASS LABELS SO WE CAN FILTER FOR X GIVEN Y
def SubXgivenC(LabelData,label, Xdata):
	# EX: x_5 = SubXgivenC(LabelData=train_labels, label=5, Xdata=train_img)
	return Xdata[np.where(LabelData == label)]

# Snipet Below return a vector of Euclidean distance of shape (n,1) where n is num of examples
def EuclidDist(X_dist, X_sample):
	return np.sqrt(np.sum((X_dist - X_sample)**2, axis=1))

def Avg_X(X_sub_C):
	x_average = np.mean(X_sub_C, axis = 0)
	return x_average

def Error_rate(y, y_pred):
	return np.sum(np.where(y == y_pred , 0., 1.)) / y_pred.shape[0]

def K_NN(k, Xdata, Sample, Xlabel):
	"""
	Finds distances between Xdata and sample: output shape (60000,),Then we return the indices of sorted values
	from min to max. Only return the first K indices. We use theses indices to print the labels corresponding to
	the NN in the train_img data
	"""
	y_pred_array = np.zeros((Sample.shape[0],k), dtype=np.int64)
	y_pred_labels = np.zeros(Sample.shape[0], dtype=np.int64)


	for i in range(Sample.shape[0]):
	#	if (i % (Sample.shape[0]//10)) == 0:
	#		print("%2d percent complete" % (i * 10 // (Sample.shape[0]//10))) 
		#elif(i == Sample.shape[0]-1):
		#	print("Complete")

		Dist     = EuclidDist(Xdata, Sample[i])
		Dist_ind = np.argsort(Dist)[0:k]
		y_pred_array[i] = Xlabel[Dist_ind]

		values, counts = np.unique(y_pred_array[i], return_counts=True)
		modes = values[np.where(counts == np.amax(counts))]
		
		if modes.shape[0] == 1:
			y_pred_labels[i] = modes[0]

		elif modes.shape[0] > 1:
			modeDist = []
			modeLabel = []

			for j in modes:
				X_mode = Xdata[Dist_ind[np.where(y_pred_array[i] == j)]]
				X_mode_avg = Avg_X(X_mode).reshape((1,784))
				X_mode_Dist = EuclidDist(X_mode_avg, Sample[i])
				modeDist.append(X_mode_Dist[0])
				modeLabel.append(j)

			minModeDistIdx = np.argsort(modeDist)[0]
			minModeLable = modeLabel[minModeDistIdx]
			y_pred_labels[i] = minModeLable


	return y_pred_labels

#################################################################################
#################################################################################
################        CREATE K-fold CV.            ############################
#################################################################################
#################################################################################


def kFoldCv(K_NN_val, TrainData, MNIST_DATA_LOC, TrainLabel,k_fold = 10):
	
	meanError = {}
	step = TrainData.shape[0]/k_fold
	splitIdx = np.random.choice(np.arange(TrainData.shape[0]), size = TrainData.shape[0], replace=False)
	
	for u in range(1,K_NN_val+1):
		#meanError = {}
		#step = TrainData.shape[0]/k_fold
		#splitIdx = np.random.choice(np.arange(TrainData.shape[0]), size = TrainData.shape[0], replace=False)
		errorsArray = []
		for q in range(k_fold):

			validIdx = splitIdx[q*step:step + q*step]
			trainIdx = np.asarray(list(set(splitIdx) - set(validIdx)))

			TrainImgSet   = TrainData[trainIdx]
			TrainLabelSet = TrainLabel[trainIdx]
			ValidImgSet   = TrainData[validIdx]
			ValidLabelSet = TrainLabel[validIdx]

			KNN = K_NN(k=u, Xdata=TrainImgSet, Sample=ValidImgSet, Xlabel=TrainLabelSet)

			errorsArray.append(Error_rate(ValidLabelSet, KNN))
		meanError['k='+str(u)] = (u,np.mean(errorsArray))

	errorPlot = []
	kVal = []
	for v in range(1, K_NN_val+1):
		errorPlot.append(meanError['k='+str(v)][1])
		kVal.append(1.0/meanError['k='+str(v)][0])

	testErrEst = errorPlot[np.argsort(errorPlot)[0]]
	bestK      = (1./np.asarray(kVal)).astype(np.int16)[np.argsort(errorPlot)[0]]

	plot = 1
	if plot:
         plt.figure(figsize=(10,6))
         plt.plot(kVal, errorPlot, color = 'green', zorder=1)
         plt.scatter(kVal, errorPlot, s=100, zorder=2)
         plt.title("10-fold CV Error Rate on MNIST Data Using K-NN", fontsize=20)
         plt.ylabel("Error Rate", fontsize=15)
         plt.xlabel("Complexity (1/K)", fontsize=15)
         plt.tick_params(axis='both', which='major', labelsize=14)
         plt.ylim(0.022, 0.033)
         plt.savefig(MNIST_DATA_LOC + "CV_ERR_RT.png")
        

	print ("Best K is %d with and 10-fold CV test error estimate of %.3f" % (bestK, testErrEst))
	
	return bestK, testErrEst, errorPlot, kVal

#################################################################################
#################################################################################
################      CREATE A K-NN SLIDING WINDOW     ##########################
#################################################################################
#################################################################################

def K_NN_WIN(k, Xdata, Sample, Xlabel):
     XdataW=Xdata.reshape(Xdata.shape[0], 28, 28)
     npad = ((0, 0), (1, 1), (1, 1))
     XdataW_pad = np.pad(XdataW, pad_width = npad, mode="constant", constant_values=0)
     y_pred_array = np.zeros((Sample.shape[0],k), dtype=np.int64)
     y_pred_labels = np.zeros(Sample.shape[0], dtype=np.int64)
     
     for i in range(Sample.shape[0]):
          if (i % (Sample.shape[0]//10)) == 0:
               print("%2d percent complete" % (i * 10 // (Sample.shape[0]//10)))
          elif (i == Sample.shape[0] - 1):
               print("COMPLETE")
          
          Dist2  = []
          
          for j in range(Xdata.shape[0]):
               
               DistWindow = []
          
               for u in range(3):
                    for v in range(3):
                         temp = XdataW_pad[j,u:u+28,v:28+v]
                         temp = temp.reshape(28*28)
                         tempDist = np.sqrt(np.sum((temp - Sample[i])**2))
                         DistWindow.append(tempDist)
          
               minima = np.min(DistWindow)
               Dist2.append(minima)
     
          Dist2 = np.asarray(Dist2)
          Dist_ind = np.argsort(Dist2)[0:k]
     
          y_pred_array[i] = Xlabel[Dist_ind]
     
          values, counts = np.unique(y_pred_array[i], return_counts=True)
          modes = values[np.where(counts == np.amax(counts))]
     
          if modes.shape[0] == 1:	
               y_pred_labels[i] = modes[0]
          elif modes.shape[0] > 1:
               modeDist = []
               modeLabel = []
          
               for p in modes:
                    X_mode = Xdata[Dist_ind[np.where(y_pred_array[i] == p)]]
                    X_mode_avg = Avg_X(X_mode).reshape((1,784))
                    X_mode_Dist = EuclidDist(X_mode_avg, Sample[i])
                    
                    modeDist.append(X_mode_Dist[0])
                    modeLabel.append(p)
          

               minModeDistIdx = (np.argsort(modeDist))[0]
               minModeLable = modeLabel[minModeDistIdx]
               y_pred_labels[i] = minModeLable
          
     return y_pred_labels


#################################################################################
#################################################################################
################        CONFUSION MATRIX             ############################
#################################################################################
#################################################################################
    
def plot_confusion_matrix(cm,
                          target_names,
                          MNIST_DATA_LOC,
                          figSz,
                          k = ' ', 
                          cmap=None,
                          normalize=True,
                          wind = ""):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    title = 'K-NN Results for MNIST Dataset \n Unnormalized confusion matrix (k=' + str(k) + ')'
    plt.figure(figsize=figSz)

    if cmap is None:
        cmap = plt.get_cmap('Greens')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Sliding Window K-NN Results for MNIST Dataset \n Normalized confusion matrix (k='+str(k)+')'
        
    from matplotlib import rcParams
    rcParams.update({'figure.autolayout': True})
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize = 10)
        plt.yticks(tick_marks, target_names, fontsize = 10)
        
    plt.title(title, fontsize=20)
    plt.colorbar()


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.3f}".format(cm[i, j]),
                     horizontalalignment="center", fontsize = 16,
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center", fontsize = 16,
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label', fontsize = 18)
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass), fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    if normalize:
        plt.savefig(MNIST_DATA_LOC + wind+ "NormConfMat.png")
    else:
         plt.savefig(MNIST_DATA_LOC + wind + "UnNormConfMat.png")


###################################################################
################        PLAY AREA      ############################
###################################################################


#train_img = train_img[0:1000,:]
#train_labels = train_labels[0:1000]
#test_img = test_img[0:20,:]
#test_labels = test_labels[0:20]

         
# 10-fold CV and return best K, and predicted test error rate
# also an array of pred test error rate vs k in order to save as .npy      
bK, teErEst, errPlt, kVPlt  = kFoldCv(K_NN_val=10, TrainData=train_img, 
                                      MNIST_DATA_LOC=MNIST_DATA_LOC,
                                      TrainLabel=train_labels,k_fold = 10)

print("Best k is: %d" % bK)
print("Test Error estimate is: %.3f" % teErEst)


# Run KNN with best K and print error rate
K_NN_Final = K_NN(k=bK, Xdata=train_img, Sample=test_img, Xlabel=train_labels)
errRt = Error_rate(test_labels, K_NN_Final)

print("ACCURACY: %.4f" % (1-errRt))
print("ERROR RATE: %.4f" % errRt)


# Here we save the final out put of K_NN_Final which are the predicted values
# given the test images. The file format is .npy which is convenient for 
# Storing numpy arrays.
np.save(file=MNIST_DATA_LOC + "KNN_FINAL_PREDICTIONS.npy", arr=K_NN_Final)
# Same as above, but we now save the K values and corresponding error
# rates obtained from k-fold CV
np.save(file=MNIST_DATA_LOC + "kFoldCV_k.npy", arr= np.asarray(kVPlt))
np.save(file=MNIST_DATA_LOC + "kFoldCV_ErrorRate.npy", arr= np.asarray(errPlt))
#SAVE METRICS
metrics = {"accuracy":1-errRt,"error_rate":errRt}
np.save(file=MNIST_DATA_LOC + "metrics.npy", arr=metrics)


#Plot Confusion Matrix for KNN - 4
cnf_matrix = confusion_matrix(test_labels, K_NN_Final)
class_names = np.arange(10)
Normalize_ConfMat = False
plot_confusion_matrix(cnf_matrix, target_names=class_names,
                      MNIST_DATA_LOC=MNIST_DATA_LOC,
                      figSz=(16,12),
                      normalize=Normalize_ConfMat, k=bK)

####################################################################
####################################################################
######################## K NN WIN ##################################
####################################################################


# Run KNN with best K and print error rate
K_NN_Window = K_NN_WIN(k=bK, Xdata=train_img, Sample=test_img, Xlabel=train_labels)
errRtWind = Error_rate(test_labels, K_NN_Window)

print("ACCURACY: %.7f" % (1-errRtWind))
print("ERROR RATE: %.7f" % errRtWind)


np.save(file=MNIST_DATA_LOC + "KNN_WIND_FINAL_PREDICTIONS.npy", arr=K_NN_Window)
metricsWind = {"accuracy":1-errRtWind,"error_rate":errRtWind}
np.save(file=MNIST_DATA_LOC + "metricsWindow.npy", arr=metricsWind)


#Plot Confusion Matrix for KNN - WINDOWS
cnf_matrixWind = confusion_matrix(test_labels, K_NN_Window)
class_names = np.arange(10)
Normalize_ConfMat = True
plot_confusion_matrix(cnf_matrixWind, target_names=class_names,
                      MNIST_DATA_LOC=MNIST_DATA_LOC,
                      figSz=(16,12),
                      normalize=Normalize_ConfMat, k=bK, wind="Wind")


print " --- PROCESS COMPLETE --- "
