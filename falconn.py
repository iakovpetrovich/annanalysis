import falconn 
par = falconn.LSHConstructionParameters()
param = falconn.get_default_parameters(num_points = len(train), dimension = len(train[0]), distance = falconn.DistanceFunction.EuclideanSquared )
print(param.lsh_family, param.l, param.k)
tables = param.l
hashes = param.k
param.l = int(1.1*tables)
para = []

for k in [hashes,int(hashes*1.5)]:
    param.k = k
    lsh = falconn.LSHIndex(param)
    lsh.setup(train)
      
    startClock = time.clock()
    startTime = process_time()
    indexlsh = lsh.construct_query_object()
    end_time = process_time()
    constructionTime = end_time - startTime
    endClock = time.clock()
    constructionClock= endClock - startClock
  
    for t in [param.l, int(param.l*2), int(param.l*3)]:
        indexlsh.set_num_probes(t)
        
        print('lsh-l'+str(param.l)+'k'+str(param.k)+'t'+str(t))
        
        rez = []
        for q in qry:
            startClock = time.clock()
            startTime = process_time()
            res = indexlsh.find_k_nearest_neighbors(q, 100)
            end_time = process_time()
            searchTime = end_time - startTime
            endClock = time.clock()
            searchClock= endClock - startClock
            rez.append(res)
          
        result = fillIfNotAllAreFound(rez) 
        result = np.asanyarray(result)
        lshReacll = returnRecall(result, groundTruth)
        avgDist = 0
      
        para.append(flannparams)
        reacll.append(lshReacll)
        algorithm.append('lsh-l'+str(param.l)+'k'+str(param.k)+'t'+str(t))
        construciotnTimes.append(constructionTime)
        searchTimes.append(searchTime)
        avgdistances.append(avgDist)
        searchClocks.append(searchClock)
        constructionClocks.append(constructionClock)
       