#Packages
import numpy as np
import attributeMethods as AM
import matplotlib.pyplot as plt

import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#CONSTANTS
N = 569                                                         #Breast cancer dataset contains 569 entries
TEST_PROP = 0.2                                                 #Proportion of data split into test set
TRAINING_QUANT = int(np.round((1-TEST_PROP)*N))                 #Number of training samples
TEST_QUANT = int(np.round(TEST_PROP*N))                         #Number of test samples
D = 30                                                           #Number of attributes in input vector
ATTRIBUTES = ["ID","Diagnosis",
              "radiusMean","radiusSE","radiusWorst",
              "textureMean","textureSE","textureWorst",
              "perimeterMean","perimeterSE","perimeterWorst",
              "areaMean","areaSE","areaWorst",
              "smoothMean","smoothSE","smoothWorst",
              "compactMean","compactSE","compactWorst",
              "concavityMean","concavitySE","concavityWorst",
              "conpointMean","conpointSE","conpointWorst",
              "symmetryMean","symmetrySE","symmetryWorst",
              "fractalMean","fractalSE","fractalWorst"]         #Attribute names
np.random.seed(39217531)                                        #Set seed to student ID

#Load in cancer dataset from CSV into np array
#4-byte floating point for real numbers: around 6-7 decimal places, sufficient for our demonstration
data = np.loadtxt("wdbc.data",delimiter=",",
                              dtype={"names": ATTRIBUTES,
                                     "formats": ("i4","S1","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4","f4")})

#Randomise & Split dataset into 80-20 train/test proportion
randomise = np.random.permutation(N)                            #Generate random permutation of indices
data = data[randomise]                                          #Randomise dataset order using permutation
trainData = data[:TRAINING_QUANT]                               #Split into training data
testData = data[TRAINING_QUANT:]                                #Split into test data
del data                                                        #Delete intermediate variable

#Refactor and isolate diagnosis attribute as 1/0: malignant=1, benign=0
Y_train = (trainData['Diagnosis'] == b'M').astype(np.int8)
Y_test = (testData['Diagnosis'] == b'M').astype(np.int8)

#Isolate x input vectors(exclude ID and Diagnosis)
feature_names = [a for a in ATTRIBUTES if a not in ("ID", "Diagnosis")]
X_train = np.column_stack([trainData[name] for name in feature_names])
X_test = np.column_stack([testData[name] for name in feature_names])

del trainData, testData

print("Length of training data:", len(X_train))
m_train = np.count_nonzero(Y_train == 1)
b_train = np.count_nonzero(Y_train == 0)
print(" - Of which malignant:", m_train)
print(" - Of which benign:", b_train)
print("M/B ratio in training data: %.2f" % (m_train / b_train if b_train > 0 else float('inf')))

print("\nLength of testing data:", len(X_test))
m_test = np.count_nonzero(Y_test == 1)
b_test = np.count_nonzero(Y_test == 0)
print(" - Of which malignant:", m_test)
print(" - Of which benign:", b_test)
print("M/B ratio in testing data: %.2f" % (m_test / b_test if b_test > 0 else float('inf')))

MAX_DEPTH = 5              #Maximum depth of decision tree

#Return array of possible thresholds using mean, median, some intervals SD away from mean for category
def possibleThreshold(xVals):
    mean = np.mean(xVals)
    sd = np.std(xVals)
    thresholds = [np.median(xVals)]
    intervals = np.arange(-4,4,0.2).tolist()
    for i in range(0,len(intervals)):
        thresholds.append(mean+intervals[i]*sd)
    return thresholds

#Counts types in xData subsection and returns them in [number of benign, number of malignant] format
#i.e. [1,3] means that there is one benign datapoint and three malignant datapoints within the given subset
def countTypes(xData):
    sN = []
    for i in range (0,2):
        type = np.where(xData==i)
        #type = np.where(xData[:,-1]==i)
        sN.append(len(type[0]))
    return(sN)

#Calculate gini impurity from list of proportions pN
def gini(pN):
    sum = 1
    for i in range(0,len(pN)):
        sum -= (pN[i])**2
    return(sum)

#Given numeric threshold, specified attribute and xData
#Return lower split and upper split [xDataLower, yDataLower, xDataUpper, yDataUpper]
def splitDatabyThreshold(xData,yData,attrCol,threshold):
    upperFilter = xData[:,int(attrCol)]>=threshold
    return(xData[upperFilter==False],yData[upperFilter==False],xData[upperFilter],yData[upperFilter])

#Decide best threshold for given attribute
def CART(xData,yData):
    thresholds = possibleThreshold(xData)    #Get possible thresholds from method
    thresholdSplits = []
    thresholdGinis =[]

    #Split data by threshold
    for i in range(0,len(thresholds)):
        splitPairs = splitDatabyThreshold(xData,yData,0,thresholds[i])
        thresholdSplits.append(splitPairs)

    #Eliminate thresholds that do not split any datapoints
    for i in range(len(thresholds)-1,-1,-1):
        if ((len(thresholdSplits[i][1]) == 0) or (len(thresholdSplits[i][3]) == 0)): #If either partition is empty, remove this threshold and split from consideration
            thresholdSplits.pop(i)
            thresholds.pop(i)
    
    if not thresholdSplits:                 #If empty (no suitable thresholds with this attribute), return nothing
        return(None)             

    #Calculate pN for each type n, calculate overall Gini Impurity for each threshold
    for i in range(0,len(thresholds)):
        """         lowerX = thresholdSplits[i][0]
        lowerY = thresholdSplits[i][1]
        upperX = thresholdSplits[i][2]
        upperY = thresholdSplits[i][3] """

        lowerX,lowerY,upperX,upperY = thresholdSplits[i]

        pNLower = np.divide(countTypes(lowerY),len(lowerY))
        pNUpper = np.divide(countTypes(upperY),len(upperY))
        giniOverall = (len(lowerX)*gini(pNLower) + len(upperX)*gini(pNUpper))/(len(upperX)+len(lowerX))
        thresholdGinis.append(giniOverall)

    #Return threshold producing lowest gini impurity
    #Best threshold value (lowest gini) and gini impurity value returned as tuple
    chosenIndex = thresholdGinis.index(min(thresholdGinis))
    return(thresholds[chosenIndex],thresholdGinis[chosenIndex])

#Identify the best attribute at a certain partition in the data
def bestAttribute(xCurrSplit,yCurrSplit):
    ginis = []

    #Compare the best thresholds and ginis for every attribute
    for i in range(0,D):
        bestThreshold = CART(xCurrSplit[:,[i]],yCurrSplit)
        if (bestThreshold != None):
            ginis.append([i,bestThreshold[0],bestThreshold[1]])
        else:
            return None
        
    #Picks attribute with lowest gini
    ginis = np.array(ginis)
    minIndex = ginis[:,2].argmin()
    return(ginis[minIndex])
    #Returns [attribute,threshold value,gini]

#As tree is being populated from root to leaf, queue is used to keep track of new nodes
#Construct decision tree
def constructDT(xTrain,yTrain):
    #First create the root node and templates for left and right child
    rootInfo = bestAttribute(xTrain,yTrain)
    lowerX,lowerY,upperX,upperY = splitDatabyThreshold(xTrain,yTrain,rootInfo[0],rootInfo[1])
    rootNode = AM.treeNode(attribute=int(rootInfo[0]),value=rootInfo[1],gini=rootInfo[2],xData=lowerX,yData=lowerY,depth=0)
    leftChild = AM.treeNode(xData=lowerX,yData=lowerY,parent=rootNode,depth=1)
    rightChild = AM.treeNode(xData=upperX,yData=upperY,parent=rootNode,depth=1)
    rootNode.setLeftChild(leftChild)
    rootNode.setRightChild(rightChild)

    tree = [rootNode,leftChild,rightChild]
    treeIndex = 3
    #Queue to hold template nodes before their thresholds have been calculated
    nextNode = AM.myQueue()
    nextNode.enqueue(leftChild)
    nextNode.enqueue(rightChild)

    #Until all nodes have been added and computed
    while (nextNode.isEmpty() == False):
        #Dequeue the next node from the queue
        current = nextNode.dequeue()
        xData,yData = current.getData()                         #Get its depth and partition of the data
        depth = current.getDepth()
        threshold = bestAttribute(xData,yData)                  #Compute its best threshold and attribute

        print(threshold)
        if (threshold is None):                                 #No suitable threhold found (i.e. high purity), make leaf
            current.setAttribute("LEAF")
            current.setValue(np.argmax(countTypes(yData)))
        elif (np.any(threshold)==False):                        #Triggered when data is all of one type (max purity), make leaf
            current.setAttribute("LEAF")
            current.setValue(np.argmax(countTypes(yData)))
        elif (depth == MAX_DEPTH):                              #If max depth has been reached, stop and make leaf
            current.setAttribute("LEAF")
            current.setValue(np.argmax(countTypes(yData)))
        else:
            current.setAttribute(threshold[0])                  #Else use the best threshold and attribute computed
            current.setValue(threshold[1])
            current.setGini(threshold[2])
            lowerX,lowerY,upperX,upperY = splitDatabyThreshold(xData,yData,threshold[0],threshold[1])
            print(countTypes(lowerY),countTypes(upperY))
            leftChild = AM.treeNode(xData=lowerX,yData=lowerY,parent=current,depth=depth+1)         
            rightChild = AM.treeNode(xData=upperX,yData=upperY,parent=current,depth=depth+1)
            tree.append(leftChild)                              #And add its left and right child to queue
            tree.append(rightChild)
            current.setLeftChild(leftChild)
            current.setRightChild(rightChild)
            nextNode.enqueue(leftChild)
            nextNode.enqueue(rightChild)
            treeIndex += 2
    
    return(tree)

#Provided with single test datapoint and decision tree
#Return class estimate
def estimateDT(xTest,tree):
    currentNode = tree[0]
    while (currentNode.getAttribute() != "LEAF"):
        attribute = currentNode.getAttribute()
        threshold = currentNode.getValue()
        if (xTest[int(attribute)]<threshold):
            currentNode = currentNode.getLeftChild()
        else:
            currentNode = currentNode.getRightChild()
    estimate = currentNode.getValue()
    return estimate

def printTree(tree):
    for node in tree:
        if node.getAttribute() == "LEAF":
            counts = countTypes(node.getData()[1])
            print("Node %d: LEAF, Class %d, Parent %d, B Count %d, M Count %d" % (tree.index(node), node.getValue(), tree.index(node.getParent()) if node.getParent() is not None else -1, counts[0], counts[1]))
        else:
            threshold = node.getThreshold()
            if threshold is not None:
                print("Node %d: Attribute %s, Threshold %.2f, Gini %.4f, Parent %d" % (tree.index(node), node.getAttribute(), threshold, node.getGini(), tree.index(node.getParent()) if node.getParent() is not None else -1))
            else:
                print("Node %d: Attribute %s, Threshold None, Gini %s, Parent %d" % (tree.index(node), node.getAttribute(), node.getGini(), tree.index(node.getParent()) if node.getParent() is not None else -1))

trainStart = time.time()
tree = constructDT(X_train,Y_train)
trainStop = time.time()
correct = 0
DTpredictedValues = []

startTime = time.time()
for i in range(0,len(X_test)):
    est = estimateDT(X_test[i],tree)
    #print("Predicted: %d, Actual: %d" % (est, Y_test[i]))
    DTpredictedValues.append(est)
    if (int(est) == int(Y_test[i])):
        correct += 1
accuracy = np.round(100*correct/len(X_test),2)
endTime = time.time()

print("Accuracy: %.2f%%" % accuracy)
print("Training time: %.2f seconds" % (trainStop - trainStart))
print("Testing time: %.2f seconds" % (endTime - startTime))

printTree(tree)