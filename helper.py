import numpy as np
import os
import struct


def fvecs_read(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv


def read_fvecs(path):
    
    fileSize = os.path.getsize(path)
    file =  open(path,'rb')
    #first 4 bytes of every vector indicate number od dimensions 
    numOfDimensions = struct.unpack('i', file.read(4))[0]
    #each vector has 4 bytes (float is 32 bits) * numberOfDimensions
    #plus 4 bytes long indicator as mentioned  
    numOfVectors = (int) (fileSize / (4 + 4*numOfDimensions))
    #init empty list for vectors
    #vectors = []
    vectors = np.zeros((numOfVectors,numOfDimensions))
    #return to the beginning
    file.seek(0)
    for vecotr in range(numOfVectors):
        file.read(4) #go trough indicator of dimensions
        #vectors.append(struct.unpack('f' * numOfDimensions, file.read(4*numOfDimensions)))
        vectors[vecotr] = struct.unpack('f' * numOfDimensions, file.read(4*numOfDimensions))
    file.close()
    return vectors


def read_ivecs(path):

    fileSize = os.path.getsize(path)
    file =  open(path,'rb')
    #first 4 bytes of every vector indicate number od dimensions 
    numOfDimensions = struct.unpack('i', file.read(4))[0]
    #each vector has 4 bytes (float is 32 bits) * numberOfDimensions
    #plus 4 bytes long indicator as mentioned  
    numOfVectors = (int) (fileSize / (4 + 4*numOfDimensions))
    #init empty list for vectors
    #vectors = []
    vectors = np.zeros((numOfVectors,numOfDimensions),int)
    #return to the beginning
    file.seek(0)
    for vecotr in range(numOfVectors):
        file.read(4) #go trough indicator of dimensions
        #vectors.append(struct.unpack('f' * numOfDimensions, file.read(4*numOfDimensions)))
        vectors[vecotr] = struct.unpack('i' * numOfDimensions, file.read(4*numOfDimensions))
    file.close()
    return vectors


#compares result set with true neighborst test set
#returns intesection of two lists
def returnTrueNeighbours(result, test):
    TrueNeighbours = []
     #for every result vector we check how many right neighbours were identified
    for i in range(result.shape[0]):
        TN = list(set(result[i].tolist()) & set(test[i].tolist()))
        TrueNeighbours.append(TN)
    return TrueNeighbours

#returns ratio of found trueneighbors / totalnumberofneighbours
def returnRecall(result, test):
    numOfTrueNeighbours = []
    #for every result vector we check how many right neighbours were identified
    for i in range(result.shape[0]):
        numTN = len(set(result[i].tolist()) & set(test[i].tolist()))
        numOfTrueNeighbours.append(numTN)
        recall = sum(numOfTrueNeighbours) /test.size
    return recall

def saveDataframe(dataFrame, path, filename):
    dataFrame.to_csv(path + filename + '.csv' , sep='\t' )
    
def createTestAndQuery(dimensions, trainPoints, queryPoints):
    import random
    testRandomMatrix = []
        
    for i in range(trainPoints):
            vector = [random.gauss(0, 1) for z in range(dimensions)]
            testRandomMatrix.append(vector)
            
    testQuery = random.sample(testRandomMatrix, 100)
    
    return testRandomMatrix, testQuery


def fillIfNotAllAreFound(result):
    for i in range(len(result)):
        if len(result[i]) < 100: 
            result[i].extend((100-len(result[i])) *[-1]) 
    return result


def get_zippedFvecs(pathToGz,memeber):
    import tarfile
    fn = pathToGz
    import struct
    import numpy as np
    t = tarfile.open(fn, 'r:gz') 
    
    
    m = t.getmember(memeber)
    file = t.extractfile(m)
    fileSize = m.size
    #file =  open(path,'rb')
    #first 4 bytes of every vector indicate number od dimensions 
    numOfDimensions = struct.unpack('i', file.read(4))[0]
    #each vector has 4 bytes (float is 32 bits) * numberOfDimensions
    #plus 4 bytes long indicator as mentioned  
    numOfVectors = (int) (fileSize / (4 + 4*numOfDimensions))
    #init empty list for vectors
    #vectors = []
    vectors = np.zeros((numOfVectors,numOfDimensions))
    #return to the beginning
    file.seek(0)
    for vecotr in range(numOfVectors):
        file.read(4) #go trough indicator of dimensions
        #vectors.append(struct.unpack('f' * numOfDimensions, file.read(4*numOfDimensions)))
        vectors[vecotr] = struct.unpack('f' * numOfDimensions, file.read(4*numOfDimensions))
    file.close()
    return vectors
    
p =  'C:/Users/jasap/Downloads/siftsmall.tar.gz'
f = 'siftsmall/siftsmall_base.fvecs'  
v = get_zippedFvecs(p,f)



def get_zippedIvecs(pathToGz,memeber):
    import tarfile
    fn = pathToGz
    import struct
    import numpy as np
    t = tarfile.open(fn, 'r:gz') 
    
    
    m = t.getmember(memeber)
    file = t.extractfile(m)
    fileSize = m.size
    #file =  open(path,'rb')
    #first 4 bytes of every vector indicate number od dimensions 
    numOfDimensions = struct.unpack('i', file.read(4))[0]
    #each vector has 4 bytes (float is 32 bits) * numberOfDimensions
    #plus 4 bytes long indicator as mentioned  
    numOfVectors = (int) (fileSize / (4 + 4*numOfDimensions))
    #init empty list for vectors
    #vectors = []
    vectors = np.zeros((numOfVectors,numOfDimensions))
    #return to the beginning
    file.seek(0)
    for vecotr in range(numOfVectors):
        file.read(4) #go trough indicator of dimensions
        #vectors.append(struct.unpack('f' * numOfDimensions, file.read(4*numOfDimensions)))
        vectors[vecotr] = struct.unpack('i' * numOfDimensions, file.read(4*numOfDimensions))
    file.close()
    return vectors

