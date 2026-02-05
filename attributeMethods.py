class treeNode:
    def __init__(self,attribute=None,value=None,gini=None,data=None,parent=None,depth=None):
        self.__attribute = attribute
        self.__value = value
        self.__gini = gini
        self.__dataPoints = data
        self.__parent = parent
        self.__leftChild = None
        self.__rightChild = None
        self.__depth = depth

    def setAttribute(self,attr):
        self.__attribute = attr

    def setValue(self,val):
        self.__value = val

    def setGini(self,gini):
        self.__gini = gini

    def setLeftChild(self,nodeID):
        self.__leftChild = nodeID
    
    def setRightChild(self,nodeID):
        self.__rightChild = nodeID

    def getValue(self):
        return self.__value

    def getData(self):
        return self.__dataPoints

    def getIndex(self):
        return self.__index
    
    def getAttribute(self):
        return self.__attribute
    
    def getGini(self):
        return self.__gini

    def getLeftChild(self):
        return self.__leftChild
    
    def getRightChild(self):
        return self.__rightChild
    
    def getThreshold(self):
        return self.__value
    
    def getParent(self):
        return self.__parent
    
    def getDepth(self):
        return self.__depth
    
class myQueue:
    def __init__(self):
        self.__arr = []
        
    def size(self):
        return len(self.__arr)
    
    def peek(self):
        if self.isEmpty():
            return(False)
        else:
            return(self.__arr[0])
    
    def enqueue(self,item):
        self.__arr.append(item)
    
    def dequeue(self):
        if self.isEmpty():
            return(False)
        else:
            return(self.__arr.pop(0))
    
    def isEmpty(self):
        return(len(self.__arr)==0)

def getID(entry):
    return(entry[0])

def getRI(entry):
    return(entry[1])

def getNa(entry):
    return(entry[2])

def getMg(entry):
    return(entry[3])

def getAl(entry):
    return(entry[4])

def getSi(entry):
    return(entry[5])

def getK(entry):
    return(entry[6])

def getCa(entry):
    return(entry[7])

def getBa(entry):
    return(entry[8])

def getFe(entry):
    return(entry[9])

def getType(entry):
    return(entry[10])