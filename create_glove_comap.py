import sys
import cPickle as pickle
import numpy as np

def augmentVisit(visit, code, treeList):
	for tree in treeList:
		if code in tree:
			visit.extend(tree[code][1:])
			break
	return

def countCooccurrenceProduct(visit, coMap):
	codeSet = set(visit)
	for code1 in codeSet:
		for code2 in codeSet:
			if code1 == code2: continue

			product = visit.count(code1) * visit.count(code2)
			key1 = (code1, code2)
			key2 = (code2, code1)

			if key1 in coMap: coMap[key1] += product
			else: coMap[key1] = product

			if key2 in coMap: coMap[key2] += product
			else: coMap[key2] = product

if __name__=='__main__':
	seqFile = sys.argv[1]
	treeFile = sys.argv[2]
	outFile = 'cooccurrenceMap.pk'

	maxLevel = 5
	seqs = pickle.load(open(seqFile, 'rb'))
	treeList = [pickle.load(open(treeFile+'.level'+str(i)+'.pk', 'rb')) for i in range(1,maxLevel+1)]

	coMap = {}
	count = 0
	for patient in seqs:
		if count % 1000 == 0: print count
		count += 1
		for visit in patient:
			for code in visit: 
				augmentVisit(visit, code, treeList)
			countCooccurrenceProduct(visit, coMap)
	
	pickle.dump(coMap, open(outFile, 'wb'), -1)
