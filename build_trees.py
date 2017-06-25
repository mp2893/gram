import sys, copy
import cPickle as pickle

if __name__ == '__main__':
    infile = sys.argv[1]
    seqFile = sys.argv[2]
    typeFile = sys.argv[3]
    outFile = sys.argv[4]

    infd = open(infile, 'r')
    _ = infd.readline()

    seqs = pickle.load(open(seqFile, 'rb'))
    types = pickle.load(open(typeFile, 'rb'))

    startSet = set(types.keys())
    hitList = []
    missList = []
    cat1count = 0
    cat2count = 0
    cat3count = 0
    cat4count = 0
    for line in infd:
        tokens = line.strip().split(',')
        icd9 = tokens[0][1:-1].strip()
        cat1 = tokens[1][1:-1].strip()
        desc1 = 'A_' + tokens[2][1:-1].strip()
        cat2 = tokens[3][1:-1].strip()
        desc2 = 'A_' + tokens[4][1:-1].strip()
        cat3 = tokens[5][1:-1].strip()
        desc3 = 'A_' + tokens[6][1:-1].strip()
        cat4 = tokens[7][1:-1].strip()
        desc4 = 'A_' + tokens[8][1:-1].strip()
        
        if icd9.startswith('E'):
            if len(icd9) > 4: icd9 = icd9[:4] + '.' + icd9[4:]
        else:
            if len(icd9) > 3: icd9 = icd9[:3] + '.' + icd9[3:]
        icd9 = 'D_' + icd9

        if icd9 not in types: 
            missList.append(icd9)
        else: 
            hitList.append(icd9)

        if desc1 not in types: 
            cat1count += 1
            types[desc1] = len(types)

        if len(cat2) > 0:
            if desc2 not in types: 
                cat2count += 1
                types[desc2] = len(types)
        if len(cat3) > 0:
            if desc3 not in types: 
                cat3count += 1
                types[desc3] = len(types)
        if len(cat4) > 0:
            if desc4 not in types: 
                cat4count += 1
                types[desc4] = len(types)
    infd.close()

    rootCode = len(types)
    types['A_ROOT'] = rootCode
    print rootCode

    print 'cat1count: %d' % cat1count
    print 'cat2count: %d' % cat2count
    print 'cat3count: %d' % cat3count
    print 'cat4count: %d' % cat4count
    print 'Number of total ancestors: %d' % (cat1count + cat2count + cat3count + cat4count + 1)
    #print 'hit count: %d' % len(set(hitList))
    print 'miss count: %d' % len(startSet - set(hitList))
    missSet = startSet - set(hitList)

    #pickle.dump(types, open(outFile + '.types', 'wb'), -1)
    #pickle.dump(missSet, open(outFile + '.miss', 'wb'), -1)


    fiveMap = {}
    fourMap = {}
    threeMap = {}
    twoMap = {}
    oneMap = dict([(types[icd], [types[icd], rootCode]) for icd in missSet])

    infd = open(infile, 'r')
    infd.readline()

    for line in infd:
        tokens = line.strip().split(',')
        icd9 = tokens[0][1:-1].strip()
        cat1 = tokens[1][1:-1].strip()
        desc1 = 'A_' + tokens[2][1:-1].strip()
        cat2 = tokens[3][1:-1].strip()
        desc2 = 'A_' + tokens[4][1:-1].strip()
        cat3 = tokens[5][1:-1].strip()
        desc3 = 'A_' + tokens[6][1:-1].strip()
        cat4 = tokens[7][1:-1].strip()
        desc4 = 'A_' + tokens[8][1:-1].strip()

        if icd9.startswith('E'):
            if len(icd9) > 4: icd9 = icd9[:4] + '.' + icd9[4:]
        else:
            if len(icd9) > 3: icd9 = icd9[:3] + '.' + icd9[3:]
        icd9 = 'D_' + icd9

        if icd9 not in types: continue
        icdCode = types[icd9]

        codeVec = []

        if len(cat4) > 0:
            code4 = types[desc4]
            code3 = types[desc3]
            code2 = types[desc2]
            code1 = types[desc1]
            fiveMap[icdCode] = [icdCode, rootCode, code1, code2, code3, code4]
        elif len(cat3) > 0:
            code3 = types[desc3]
            code2 = types[desc2]
            code1 = types[desc1]
            fourMap[icdCode] = [icdCode, rootCode, code1, code2, code3]
        elif len(cat2) > 0:
            code2 = types[desc2]
            code1 = types[desc1]
            threeMap[icdCode] = [icdCode, rootCode, code1, code2]
        else:
            code1 = types[desc1]
            twoMap[icdCode] = [icdCode, rootCode, code1]
    
    # Now we re-map the integers to all medical codes.
    newFiveMap = {}
    newFourMap = {}
    newThreeMap = {}
    newTwoMap = {}
    newOneMap = {}
    newTypes = {}
    rtypes = dict([(v, k) for k, v in types.iteritems()])

    codeCount = 0
    for icdCode, ancestors in fiveMap.iteritems():
        newTypes[rtypes[icdCode]] = codeCount
        newFiveMap[codeCount] = [codeCount] + ancestors[1:]
        codeCount += 1
    for icdCode, ancestors in fourMap.iteritems():
        newTypes[rtypes[icdCode]] = codeCount
        newFourMap[codeCount] = [codeCount] + ancestors[1:]
        codeCount += 1
    for icdCode, ancestors in threeMap.iteritems():
        newTypes[rtypes[icdCode]] = codeCount
        newThreeMap[codeCount] = [codeCount] + ancestors[1:]
        codeCount += 1
    for icdCode, ancestors in twoMap.iteritems():
        newTypes[rtypes[icdCode]] = codeCount
        newTwoMap[codeCount] = [codeCount] + ancestors[1:]
        codeCount += 1
    for icdCode, ancestors in oneMap.iteritems():
        newTypes[rtypes[icdCode]] = codeCount
        newOneMap[codeCount] = [codeCount] + ancestors[1:]
        codeCount += 1

    newSeqs = []
    for patient in seqs:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in visit:
                newVisit.append(newTypes[rtypes[code]])
            newPatient.append(newVisit)
        newSeqs.append(newPatient)

    pickle.dump(newFiveMap, open(outFile + '.level5.pk', 'wb'), -1)
    pickle.dump(newFourMap, open(outFile + '.level4.pk', 'wb'), -1)
    pickle.dump(newThreeMap, open(outFile + '.level3.pk', 'wb'), -1)
    pickle.dump(newTwoMap, open(outFile + '.level2.pk', 'wb'), -1)
    pickle.dump(newOneMap, open(outFile + '.level1.pk', 'wb'), -1)
    pickle.dump(newTypes, open(outFile + '.types', 'wb'), -1)
    pickle.dump(newSeqs, open(outFile + '.seqs', 'wb'), -1)
