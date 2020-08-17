import h5py
import numpy
import os
import random
import sys



#k je broj dimenzija - 128, n je broj vektora, float je 4 bajta 
def _load_texmex_vectors(f, n, k):
    import struct

    v = numpy.zeros((n, k))
    for i in range(n):
        f.read(4)  # ignore vec length
        v[i] = struct.unpack('f' * k, f.read(k * 4))

    return v


def _get_irisa_matrix(t, fn):
    import struct
    m = t.getmember(fn)
    f = t.extractfile(m)
    k, = struct.unpack('i', f.read(4))#k je broj dimenzija
    n = m.size // (4 + 4 * k)#n je broj vektora
    f.seek(0)
    return _load_texmex_vectors(f, n, k)


def sift(out_fn):
    import tarfile

    url = 'ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz'
    fn = os.path.join('data', 'sift.tar.tz')
    download(url, fn)
    with tarfile.open(fn, 'r:gz') as t:
        train = _get_irisa_matrix(t, 'sift/sift_base.fvecs')
        test = _get_irisa_matrix(t, 'sift/sift_query.fvecs')
        #write_output(train, test, out_fn, 'euclidean')
        
import pandas as pd
pd.read('C:/Users/jasap/Downloads/siftsmall.tar/siftsmall/siftsmall_base.fvecs')

import struct
with open('C:/Users/jasap/Downloads/siftsmall.tar/siftsmall/siftsmall_base.fvecs','rb') as f:
    #a = f.read(4)
    f.size()
    f.read(516)
    a = struct.unpack('i', f.read(4))[0]
    print(a)
    prvaDimDrugiVektor = struct.unpack('f', f.read(4))[0]
    print(prvaDimDrugiVektor)
    f.close()
    
df = pd.read_hdf('C:/Users/jasap/Downloads/sift-128-euclidean.hdf5')


import h5py
filename = 'C:/Users/jasap/Downloads/sift-128-euclidean.hdf5'

with h5py.File(filename, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[1]

    # Get the data
    data = list(f[a_group_key])

    
data[1][1]
len(data)        
len(data[1])

import numpy as np


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

file = 'C:/Users/jasap/Downloads/siftsmall.tar/siftsmall/siftsmall_base.fvecs'

'C:/Users/jasap/Downloads/siftsmall.tar/siftsmall/siftsmall_base.fvecs'


def read_fvecs
    import os
    import struct
    path = 'C:/Users/jasap/Downloads/siftsmall.tar/siftsmall/siftsmall_base.fvecs'
    
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
        vectors[vecotr] = struct.unpack('f' * numOfDimensions, file.read(4*numOfDimensions))[0]
    file.close()
    return vectors


len(vectors[1])
vectors[1][0]



return vectors



