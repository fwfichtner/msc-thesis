from bitstring import BitStream, BitArray


def getEqualInnerNeighbours(locationalCode):
	# Set up the different values of the input materialised path
	# print "Current node: ", locationalCode
		# The final digit of the materialised path for calculating inner neighbours
	leafDecimal = int(locationalCode[-1])

	# Check inner neighbours 
		# take the mask to switch on/off every bit (dimension) of the highst level of resolution
	xBinary = BitArray(bin="{0:03b}".format(leafDecimal)) ^ BitArray(bin='001')
	yBinary = BitArray(bin="{0:03b}".format(leafDecimal)) ^ BitArray(bin='010')
	zBinary = BitArray(bin="{0:03b}".format(leafDecimal)) ^ BitArray(bin='100')

	# 	# results in 3 inner neighbours (a, b and c) --> it converts the binary coordinate back to integers
	xMP = "{0}{1}".format(locationalCode[0:-1], int(xBinary.bin, 2))
	yMP = "{0}{1}".format(locationalCode[0:-1], int(yBinary.bin, 2))
	zMP = "{0}{1}".format(locationalCode[0:-1], int(zBinary.bin, 2))

	return [xMP, yMP, zMP]


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
	Kx = getK(locationalCode, 'x')
	Ky = getK(locationalCode, 'y')
	Kz = getK(locationalCode, 'z')
		# take the complement of the entire materialised for every dimension
	xMP = ""
	yMP = ""
	zMP = ""

	for i, node in enumerate(reversed(locationalCode)):
	 	if Kx == False:
	 		pass
	 	elif i > Kx:
	 		xMP += str(node)
	 	else:
			nodex = BitArray(bin="{0:03b}".format(int(node)))
			xDigit = nodex ^ BitArray(bin='001')
			xMP += str(int(xDigit.bin, 2))
		
		if Ky == False:
	 		pass
	 	elif i > Ky:
	 		yMP += str(node)
	 	else:
			nodey = BitArray(bin="{0:03b}".format(int(node)))
			yDigit = nodey ^ BitArray(bin='010')
			yMP += str(int(yDigit.bin, 2))
		
		if Kz == False:
	 		pass
	 	elif i > Kz:
	 		zMP += str(node)
	 	else:
			nodez = BitArray(bin="{0:03b}".format(int(node)))
			zDigit = nodez ^ BitArray(bin='100')
			zMP += str(int(zDigit.bin, 2))

	neighbours = []
	neighbours.append((xMP[::-1], Kx))
	neighbours.append((yMP[::-1], Ky))
	neighbours.append((zMP[::-1], Kz))

	return neighbours

def giveMeAllEqualNeighbours(currentNode):
	EqualOuterNeighbours = getEqualOuterNeighbours(currentNode)
	EqualInnerNeighbours = getEqualInnerNeighbours(currentNode)
	allNeighbours = set()

	for i in range(3):
		allNeighbours.add(EqualOuterNeighbours[i][0])
		allNeighbours.add(EqualInnerNeighbours[i])

	return allNeighbours

if __name__ == '__main__':
	giveMeAllEqualNeighbours('11033423')
	# set(['11033067', '11033601', '11033422', '11033421', '11033427', '11033432'])