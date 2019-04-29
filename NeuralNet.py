import sys , random
import numpy as np
from copy import deepcopy
import math
hid_vals = 50
np.random.seed(20)
random.seed(20)
def sigmoid(x):
	answer = 1/(1+np.exp(-x))
	return answer
def readImages(FileName):
    print ("Reading Images from File...")
    file = open(FileName)
    line = "ABC"
    LineCount = 0
    IMAGES = []
    Image = ""
    while line != '':
        line = file.readline()
        line = line.strip('[')
        line = line.replace(']', "")
        line = line.strip('\n')
        Image = Image + line
        if(LineCount == 43):
            Image = Image.split()
            Image = list(map(int , Image))
            IMAGES.append(Image)
            Image = ''
            LineCount = 0
        else:
            LineCount = LineCount + 1
    # print(IMAGES[:1])
    IMAGES = np.array(IMAGES)
    return IMAGES
def readLabels(TrainLabels):
    print('Reading Labels...')
    # return 1
    f = open(TrainLabels)
    LABLES = ""
    x = "YOTA YOTA YOTA"
    while x != "":
        x = f.readline()
        LABLES = LABLES + x
    LABLES = list(map(int , LABLES.split()))
    # print (LABLES[:10])
    return LABLES 
def Eculedian(a , b , ax = 1):
    return np.linalg.norm(a - b , axis=ax )

def get_clusters(IMAGES):
    clusters = []
    taken = []
    for i in range( 0 , hid_vals):
        while 1:
            index = np.random.randint(0 , 783)
            if index not in taken:
                taken.append([index])
                clusters.append(IMAGES[index])
            break
    clusters = np.array(clusters)
    return clusters
def get_activations(CLUSTERS, IMAGE):
    activations = []
    for i in range(0 , len(CLUSTERS)):
        dist = Eculedian(CLUSTERS[i] , IMAGE , None)
        activations.append( math.exp((-0.00000018*dist*dist)))
    activations = np.array(activations)
    return activations
    # pass
def Train(TrainFile , TrainLabels, LearningRate):
    global hid_vals
    IMAGES =  readImages(TrainFile)
    LABELS = readLabels(TrainLabels)
    NEW_CLUSTERS = get_clusters(IMAGES)
    # error = 15000.0
    # print("Defining Clusters...")
    # while error > 2000.0:
    #     CLUSTERS = np.zeros(len(IMAGES))
    #     for i in range(0 , len(IMAGES)):
    #         distances = Eculedian(IMAGES[i] , NEW_CLUSTERS)
    #         closest = np.argmin(distances)
    #         CLUSTERS[i] = closest
        
    #     OLD_CLUSTERS = deepcopy(NEW_CLUSTERS) 
        
    #     for i in range (0 , hid_vals):
    #         Cluster_Points = []
    #         for K in range(0 , len(IMAGES)):
    #             if int(CLUSTERS[K] == i):
    #                 Cluster_Points.append(IMAGES[K])
    #         if len(Cluster_Points) is 0:
    #             pass
    #             # print ("Cluster ",i," Has No Points")
    #         else:
    #             NEW_CLUSTERS[i] = np.mean(Cluster_Points , axis = 0)
    #     error = Eculedian(NEW_CLUSTERS, OLD_CLUSTERS, None)
    #     print("ERROR Reduced to:", error)
    WEIGHTS = 2*np.random.random((hid_vals,10)) -1
    for epoch in [1 , 2]:
        count = 0
        for i in range(0 , len(IMAGES)):
            Expected_array = np.zeros((10,), dtype = float)  
            Expected_array [LABELS[i]] = 1.0
            activations = get_activations(NEW_CLUSTERS , IMAGES[i])
            output_activations = sigmoid(np.dot( activations.T, WEIGHTS))
            m = np.argmax(output_activations)
            if m == LABELS[i]:
                count = count + 1
            Error = output_activations*(1-output_activations)*(Expected_array - output_activations)
            
            for k in range(0 , 10):
                WEIGHTS[:,k] += LearningRate * Error[k] * activations

        accuracy = float((count/len(IMAGES))*100)
        accuracy = "%.2f" % accuracy
        print ("Epoch ", epoch, " Finished, Accuracy: ", accuracy, "%")
    f = open("netWeights.txt","w+")
    np.savetxt(f, WEIGHTS)
    f.close()
    f = open("netfeatures.txt","w+")
    np.savetxt(f, NEW_CLUSTERS)
    f.close()
def Test(TestFile , TestLabels , Weights):
    IMAGES =  readImages(TestFile)
    LABELS = readLabels(TestLabels)
    WEIGHTS = np.genfromtxt("netWeights.txt" )
    NEW_CLUSTERS = np.genfromtxt("netfeatures.txt")
    count = 0
    for i in range(0, len(IMAGES)):
        Expected_array = np.zeros((10,), dtype = float)  
        Expected_array [LABELS[i]] = 1.0
        activations = get_activations(NEW_CLUSTERS , IMAGES[i])
        output_activations = sigmoid(np.dot( activations.T, WEIGHTS))
        m = np.argmax(output_activations)
        print("Output: ", m , " Actual: ", LABELS[i])
        if m == LABELS[i]:
            count = count + 1
    accuracy = float((count/len(IMAGES))*100)
    accuracy = "%.2f" % accuracy
    print ("Accuracy: ", accuracy, "%")
def main():
    INPUTS = sys.argv
    if(INPUTS[1] == 'train'):
        Train( INPUTS[2] , INPUTS[3] , float(INPUTS[4]) )
    else:
        Test( INPUTS[2] , INPUTS[3] , float(INPUTS[4]) )
if __name__ == "__main__":
    main()