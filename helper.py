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


path = 'C:/Users/jasap/Downloads/siftsmall.tar/siftsmall/siftsmall_base.fvecs'
v = read_fvecs(path)
test = fvecs_read(path)

v[11][3]
test[11][3]



def calculateRacl(tOfTrueNeighbours):
    recall = sum(numOfTrueNeighbours) /result.size

def returnTrueNeighbours(result, test):
    TrueNeighbours = []
    for i in range(result.shape[0]):
        TN = list(set(result[i].tolist()) & set(groundTruth[i].tolist()))
        TrueNeighbours.append(TN)
    return TrueNeighbours

def returnNumerOfTrueNeighboursPerQuery(result, test):
    listOfTrueNeighbors = returnTrueNeighbours(result,test)
    numOfTrueNeighbours = []
    for i in range(len(listOfTrueNeighbors)):
        numTN = len(listOfTrueNeighbors[i])
        numOfTrueNeighbours.append(numTN)
    return numOfTrueNeighbours

def calculateRacl(result, test):
    trueNNperQuery = returnNumerOfTrueNeighboursPerQuery(result, test)
    recall = sum(trueNNperQuery) /len(trueNNperQuery)
    return recall
    


