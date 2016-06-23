from bitstring import BitStream, BitArray


def getEqualInnerNeighbours(locationalCode):
	if int(locationalCode[-1]) > 3:
		return [''] # ugly, but works
	# Set up the different values of the input materialised path
	# print "Current node: ", locationalCode
		# The final digit of the materialised path for calculating inner neighbours
	leafDecimal = int(locationalCode[-1])

	# Check inner neighbours 
		# take the mask to switch on/off every bit (dimension) of the highst level of resolution
	zBinary = BitArray(bin="{0:03b}".format(leafDecimal)) ^ BitArray(bin='100')

	# 	# results in 3 inner neighbours (a, b and c) --> it converts the binary coordinate back to integers

	zMP = "{0}{1}".format(locationalCode[0:-1], int(zBinary.bin, 2))

	# return [xMP, yMP, zMP]

	return [zMP]


def getK(locationalCode, dimension):
	dimension = dimension.lower()
	# NOTE: this is only the code for x in the paper (and z in our case - the paper has the binary numbers flipped in the Matrices of J, B and E)
	# print "Get K for node: {0} in dimension: {1}\n".format(locationalCode, dimension)

	if dimension == 'x':
		dimension = 2
	elif dimension == 'y':
		dimension = 1
	elif dimension == 'z':
		dimension = 0
	else:
		return "Please input a dimension!"

	n = len(locationalCode)

	prevNode = None
	for i, node in enumerate(reversed(locationalCode)):
		if i == 0: # Here we define Xn (or Yn or Zn for that matter), which doesn't get checked any further because it's the first position
			Xn = "{0:03b}".format(int(node))[dimension]		
		else:
			if "{0:03b}".format(int(node))[dimension] == Xn: #X(n-i)
				pass
			else:
				if int("{0:03b}".format(int(prevNode))[dimension]) == int(Xn): #X(n-k)+1
					return i #in this case i == k
		prevNode = node

		if i+1 == n: #in this case there's no neigbor in this direction
			# print "No neigbor found!" 
			return False
	

def getEqualOuterNeighbours(locationalCode):

	Kz = getK(locationalCode, 'z')
		# take the complement of the entire materialised for every dimension

	if int(locationalCode[-1]) < 4:
		return [('', Kz)]# ugly, but works

	zMP = ""

	for i, node in enumerate(reversed(locationalCode)):
		if Kz == False:
	 		pass
	 	elif i > Kz:
	 		zMP += str(node)
	 	else:
			nodez = BitArray(bin="{0:03b}".format(int(node)))
			zDigit = nodez ^ BitArray(bin='100')
			zMP += str(int(zDigit.bin, 2))

	neighbours = []
	neighbours.append((zMP[::-1], Kz))
	# print neighbours
	return neighbours


def getLargerNeighbours(equalNeighbours):
	# We create an empty set that will contain all larger sized neighbours
	largerNeighbours = set()
	# We loop through all the equal sized neighbours
	for neighbour in equalNeighbours:
		# For every equal sized neighbour we loop through the digits
		for i, digit in enumerate(neighbour[0]):
			# If the level is smaller than K we append it to the set with larger sized neighbours
			if i < neighbour[1]:
				i += 1
				largerNeighbours.add(neighbour[0][:-i])
	
	return largerNeighbours, [equalNeighbours[0][0]]

def createMPs(curDict, prevNeighbours):
	newNeighbours = set()
	for prev in prevNeighbours:
		for each in curDict:
			new = prev + each
			newNeighbours.add(new)

	return newNeighbours

def getSmallerNeighbours(EqualInnerNeighbours, EqualOuterNeighbours, maxLevels, currentNode):

	neighbours = set()
	Zn = int(BitArray(bin="{0:03b}".format(int(currentNode[-1])))[0])

	compZn = int((bin(Zn) ^ BitArray(bin='1')).bin)


	Dict = {
		'innerz' : [],
		'outerz' : []
	}

	for i in range(4):
		d1 = int(BitArray(bin="{0:02b}".format(i))[0])
		d2 = int(BitArray(bin="{0:02b}".format(i))[1])

		Dict['innerz'].append(str(int(BitArray(bin="{0}{1}{2}".format(Zn, d1, d2)).bin, 2)))
		Dict['outerz'].append(str(int(BitArray(bin="{0}{1}{2}".format(compZn, d1, d2)).bin, 2)))

	prevInnerZ, prevOuterZ = ([EqualInnerNeighbours[0]]), ([EqualOuterNeighbours[0]])
	
	neighboursZi = set()
	neighboursZo = set()

	for i in range(maxLevels-len(currentNode)):

		if int(currentNode[-1]) < 4:
			newInnerZ = createMPs(Dict['innerz'], prevInnerZ)	
			for each in newInnerZ:
				neighboursZi.add(each)
			prevInnerZ = newInnerZ
		else:
			if prevOuterZ != ['']:
				newOuterZ = createMPs(Dict['outerz'], prevOuterZ)	
				for each in newOuterZ:
					neighboursZo.add(each)
				prevOuterZ = newOuterZ

	neighbours = neighboursZo | neighboursZi

	return neighbours


def giveMeAllZNeighbours(currentNode, maxLevels=8):
	EqualOuterNeighbours = getEqualOuterNeighbours(currentNode)
	EqualInnerNeighbours = getEqualInnerNeighbours(currentNode)
	LargerNeighbours, EqualOuterNeighbours = getLargerNeighbours(EqualOuterNeighbours)

	
	SmallerNeighbours = getSmallerNeighbours(EqualInnerNeighbours, EqualOuterNeighbours, maxLevels, currentNode)
	allNeighbours = set()


	allNeighbours.add(EqualOuterNeighbours[0])
	allNeighbours.add(EqualInnerNeighbours[0])

	allNeighbours = allNeighbours.union(LargerNeighbours)
	allNeighbours = allNeighbours.union(SmallerNeighbours)

	return allNeighbours

if (__name__ == '__main__'):
	neighbours = giveMeAllZNeighbours('011056')
	print neighbours
	 