# import libraries
import psycopg2
import os
# import datetime
from liblas import file
from cStringIO import StringIO
import webbrowser
import csv
# from flask import flash

# define the binary function to get the material path of a leaf node in an octree
def getLocationalCode(depth, x, y, z):
	ocKey = ""
	for i in range(depth, 0, -1):
		digit = 0
		mask = 1 << (i-1)
		if (x & mask) != 0:
			digit += 1
		if (y & mask) != 0:
			digit += 2
		if (z & mask) != 0:
			digit += 4
		ocKey += str(digit)
	return ocKey


def LasToOctree(depth, lasFile, dbms_name, user, password):
	# print "read las file: " + str(datetime.datetime.now()) 
	# Open .Las file
	f = file.File(lasFile, mode='r')
	h = f.header

	print "Calculate BBox translation and scaling"
	# check bounding to calculate scaling factor
	lenX = h.max[0] - h.min[0]   # length in x direction
	lenY = h.max[1] - h.min[1]   # length in y direction
	lenZ = h.max[2] - h.min[2]   # length in Z direction
	
	# Putting the BBox min coordinates at 0
	if (h.min[0] < 0):
		translateX = abs(h.min[0])
	elif (h.min[0] > 0): 
		translateX = 0 - h.min[0]
	else:
		translateX = 0

	if (h.min[1] < 0):
		translateY = abs(h.min[1])
	elif (h.min[1] > 0): 
		translateY = 0 - h.min[1]
	else:
		translateY = 0

	if (h.min[2] < 0):
		translateZ = abs(h.min[2])
	elif (h.min[0] > 0): 
		translateZ = 0 - h.min[2]
	else:
		translateZ = 0

	if (lenX != 2**depth) or (lenY != 2**depth) or (lenZ != 2**depth):
		# if scaling is required we take the maximum domain of the point cloud for scaling
		maxDomain = max([lenX, lenY, lenZ])
		scale = 2**depth / maxDomain
	else: 
		# if no scaling is required we set the scale to 1
		scale = 1

	# calculate the dimension of the minimum leaf size
	minleafDomain = maxDomain / (2**depth)
	print 'minleafDomain is ', minleafDomain, ' and the scale is ', scale

	# calculate the height of the floor to place path of scanner in there
	floorheight = h.min[2]

	# open a text file to write the points to and a text file to write the materialised paths to
	# PointsWriter = open("temp.txt", "w")

	# Load and iterate through the points  
	i = 0
	stringPoints =  StringIO()

	conn = psycopg2.connect("host='localhost' dbname='"+dbms_name+"' user='"+ user + "' password='"+ password + "'")
	cur = conn.cursor()

	w = csv.writer(stringPoints) 
	data = []

	# print "Start writing points to octree file: "+ str(datetime.datetime.now()) 

	# We assume there is no colour in the point cloud, unless there is any colour detected in the first 10 points 
	colour = False
	for point in f:
		if i < 10:
			if point.color.red != 0:
				colour = True
			elif point.color.green != 0:
				colour = True
			elif point.color.blue != 0: 
				colour = True

		#scale point 
		x = (point.x + translateX) * scale
		y = (point.y + translateY) * scale       
		z = (point.z + translateZ) * scale

		# Snap point to leaf node by converting float to integer and truncate towards 0 
		LeafNode = (int(x),int(y),int(z))

		# retrieve Material path from box
		LocationalCode = getLocationalCode(depth, LeafNode[0], LeafNode[1], LeafNode[2])

		# origin coordinates necessary for histograms of shape grammar method, etc., changes also later when written to db or getCoordinates()??

		# Check if there is color or not in this las file
		if colour == False:
			# Set the color to 0 0 0 
			# stringPoints.write(",0,0,0\n")
			point_list = [i,LocationalCode,x,y,z,0,0,0]
		else:
			# retrieve the color of the point from the las file
			# converting 16 bit colors: point_list = [i,LocationalCode,x,y,z,((point.color.red >> 16) & 255) + ((point.color.red >> 8) & 255) + ((point.color.red) & 255),((point.color.green >> 16) & 255) + ((point.color.green >> 8) & 255) + ((point.color.green) & 255), ((point.color.blue >> 16) & 255) + ((point.color.blue >> 8) & 255) + ((point.color.blue) & 255)]
			point_list = [i,LocationalCode,x,y,z, point.color.red,point.color.green, point.color.blue]
		
		data.append(point_list)

		# store material points to file after every 100.000 records
		if i % 100000 == 0 and i > 0:

			w.writerows(data)

			stringPoints.seek(0)
			cur.copy_from(stringPoints, 'pointcloud', sep=',', columns=('index', 'locationalcode', 'x', 'y', 'z', 'red', 'green', 'blue') ) 
			# conn.commit()

			stringPoints.close()
			stringPoints = StringIO()
			w = csv.writer(stringPoints) 
			data = []

			print i, "points written"

		i += 1
	
	# store the remaining points 
	w.writerows(data)
	stringPoints.seek(0)
	cur.copy_from(stringPoints, 'pointcloud', sep=',', columns=('index', 'locationalcode', 'x', 'y', 'z', 'red', 'green', 'blue') ) 
	conn.commit()

	print i, "points written"
	
	# Close the writer and the database connection
	conn.close()
	stringPoints.close()
	
	# print "Done writing points to octree file: " + str(datetime.datetime.now()) 
	return minleafDomain, floorheight

def create_dbms(dbms_name, user, password):
	print "Create database"
	con = psycopg2.connect("host='localhost' user='"+ user + "' password='"+ password + "'")
	con.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
	cur = con.cursor()

	try:
		cur.execute('drop DATABASE ' + dbms_name)		
		print "Database already exists, replacing database"
	except:
		pass
	cur.execute('CREATE DATABASE ' + dbms_name)
	cur.close()
	con.close()

	conn = psycopg2.connect("host='localhost' dbname='"+dbms_name+"' user='"+ user + "' password='"+ password + "'")
	cur = conn.cursor()
	# cur.execute('CREATE EXTENSION POSTGIS;')
	# PRIMARY KEY constraint added, think about removing colors or make this dependent
	cur.execute('CREATE TABLE pointcloud(index varchar(100) PRIMARY KEY, locationalcode varchar(100), x varchar(100), y varchar(100), z varchar(100), red varchar(10), green varchar(10), blue varchar(10));')
	# cur.execute('CREATE TABLE pointcloud(index varchar(100) PRIMARY KEY, locationalcode varchar(100), x varchar(100), y varchar(100), z varchar(100));')
	cur.execute('CREATE TABLE emptyspace(locationalcode varchar(100) PRIMARY KEY, x varchar(100), y varchar(100), z varchar(100), leafSize int);')
	conn.commit()

def call_dbms(dbms_name, user, password, threshold):
	# This function fetches the table pointcloud with full boxes from the database
	# print "Retrieve full leaf nodes from database: " + str(datetime.datetime.now()) 

	conn = psycopg2.connect("host='localhost' dbname='"+dbms_name+"' user='"+ user + "' password='"+ password + "'")
	cur = conn.cursor()
	#this determines that a box has to include at least threshold points in order to be taken into consideration for the octree of empty space
	if threshold is '' or threshold is '1':
		cur.execute('SELECT locationalcode FROM pointcloud;')
	else:
		cur.execute('SELECT locationalcode FROM pointcloud GROUP BY locationalcode HAVING (COUNT(index) >= '+ threshold +')')
	table = cur.fetchall()

	# print "Retrieved full leaf nodes from database: " + str(datetime.datetime.now()) 

	return table

def find_empty(table, maximumLevels):
	# This function finds the empty nodes for every level in the tree
	# print "Calculate the empty leaf nodes in the point cloud" + str(datetime.datetime.now()) 

	tree = map(str, range(8))
	empty = set()
	nonempty = [{str()}]

	# Iterates over the depth of the tree
	for level in range(maximumLevels):					

		# Stores all the nonempty nodes of the current tree level in a set, and adds the set to a list
		nonempty_cur_level = set()
		for entry in table:
			nonempty_cur_level.add(entry[0][0:level+1])
		nonempty.append(nonempty_cur_level)

		# Finds all the empty nodes in the current level of the tree, and stores these in a set
		for node in nonempty[level]:
			for number in tree:
				child = node + number
				if child not in nonempty[level+1]:
					empty.add(child)


	# print "Done calculating empty leaf nodes: " + str(datetime.datetime.now()) 
	return empty	

def getCoord(depth, LocationalCode):
	#This function finds the coordinates of the empty space
    #checks the size of the voxel according to the lenght of the path
    if len(LocationalCode) < depth:
    	leafSize = 2 ** (depth - len(LocationalCode))
    else:
        leafSize = 1

    x = ''
    y = ''
    z = ''

    # check digit per digit
    for digit in range(len(LocationalCode)):
        # check which 0s/1s should be added to the binary coordinate strings  
        if int(LocationalCode[digit]) > 3:
            if int(LocationalCode[digit]) > 5:
                if int(LocationalCode[digit]) > 6:   #7
                    x +='1'
                    y +='1'
                    z +='1'
                else:           #6
                    x +='0'
                    y +='1'
                    z +='1'                   
            elif int(LocationalCode[digit]) > 4:     #5
                x +='1'
                y +='0'
                z +='1'
            else:               #4
                x +='0'
                y +='0'
                z +='1'
        elif int(LocationalCode[digit]) > 1:
            if int(LocationalCode[digit]) > 2:       #3
                x +='1'
                y +='1'
                z +='0'           
            else:               #2
                x +='0'
                y +='1'
                z +='0'
        else:
            if int(LocationalCode[digit]) > 0:       #1 
                x +='1'
                y +='0'
                z +='0'
            else:               #0
                x +='0'
                y +='0'
                z +='0'

    remainingLevels = "0"*(depth - len(LocationalCode))
    x += remainingLevels
    y += remainingLevels
    z += remainingLevels

    # convert the binary string to decimal integers
    x = int(x, 2)
    y = int(y, 2)
    z = int(z, 2)

    return str(LocationalCode)+" "+str(x)+" "+str(y)+" "+str(z)+" "+str(leafSize)

def emptyLeaf2DBMS(LocationalCodes, dbms_name, user, password, maximumLevels):
	# print "Start writing empty leafs to database: " + str(datetime.datetime.now()) 
	i = 0

	conn = psycopg2.connect("host='localhost' dbname='"+dbms_name+"' user='"+ user + "' password='"+ password + "'")
	cur = conn.cursor()

	stringEmpty = StringIO()
	w = csv.writer(stringEmpty) 
	data = []

	for path in LocationalCodes:
		data.append([getCoord(maximumLevels, path)])
		
       	# store empty leafs to database after every 100.000 records
		if (i % 100000 == 0) and (i > 0):
			w.writerows(data)

			stringEmpty.seek(0)
			cur.copy_from(stringEmpty, 'emptyspace', sep=' ', columns=('locationalcode', 'x', 'y', 'z', 'leafSize') ) 
			# conn.commit()

			stringEmpty.close()
			stringEmpty = StringIO()
			w = csv.writer(stringEmpty) 
			data = []

			print i, " empty leafs written"

		i += 1

	# store the remaining points
	w.writerows(data) 
	stringEmpty.seek(0)
	cur.copy_from(stringEmpty, 'emptyspace', sep=' ', columns=('locationalcode', 'x', 'y', 'z', 'leafSize') ) 
	conn.commit()

	print i, "Empty leafs written"
	
	# Close the writer and the database connection
	conn.close()
	stringEmpty.close()

	# print "Done writing empty leafs to database: " + str(datetime.datetime.now()) 

def CheckInput(lasFile, dbms_name, user, password, maximumLevels=8):
	error = False 
	# Check if the input filename is of type '.las'
	if lasFile.lower().endswith(('.las')):
		# Check if the file exists
		if (os.path.isfile(lasFile) == True ):
			# Test the database connection with user's username and password combination
			try: 
				con = psycopg2.connect("host='localhost' user='"+ user + "' password='"+ password + "'")
				con.close()
			except:
				# Raise error for wrong username/password combination
				print "Please provide the correct username and password combination for Postgres"
				# flash("Please provide the correct username and password combination for Postgres")
				error = True
		else:
			print lasFile + " does not exist at the specified location"
			# flash(lasFile + " does not exist at the specified location")
			error = True
	else:
		# Raise error for wrong input files type
		print "The input file has to be of type '.las' " 
		# flash("The input file has to be of type '.las' ")
		error = True

	if maximumLevels > 10:
		# The maximum number of octree levels that is allowed is 8  
		print "The octree level exceeds the allowed maximum level. Please set level to 8 or lower"
		# flash("The octree level exceeds the allowed maximum level. Please set level to 8 or lower")
		error = True
		maximumLevels = 10

	dbms_name = dbms_name.lower()
	return dbms_name, error

def createConfigPY(dbms_name, user, password):
	file = open(os.getcwd() +"/config.py", "w")
	file.write('SQLALCHEMY_DATABASE_URI = "postgresql://'+user+':'+password+'@localhost/'+dbms_name+'"\n')
	file.close()
	return

def writeLAS(table, scale, translateX, translateY, translateZ):
	# in case a las file has to be written, this translates back to original scale etc.
	# added later by Tom Broersen
	h = header.Header()
	f = file.File('denoised_test.las',mode='w', header= h) # to denoise delete all leafs with less than threshold points in it
	
	print "writing points"
	for entry in table:
		pt = point.Point()
		pt.x = (float(entry[2]) / scale) - translateX
		pt.y = (float(entry[3]) / scale) - translateY
		pt.z = (float(entry[4]) / scale) - translateZ
		f.write(pt)
	f.close()

def Pointless(lasFile, threshold="", dbms_name="",  maximumLevels=8, user="postgres", password=""):
	# This is the master file calling all the other functions based on the user's input
	print "Running the Pointless Converter > build octrees, find empty space"

	# Check if the user input is valid
	dbms_name, error = CheckInput(lasFile, dbms_name, user, password, maximumLevels)

	if error == False:
		print "Start loading "+lasFile+" file "# + str(datetime.datetime.now())
	else:
		return 

	# Call the different functions with the user's input parameters
	create_dbms(dbms_name, user, password) 

	minleafDomain, floorheight = LasToOctree(maximumLevels, lasFile, dbms_name, user, password)
	
	EmptyLocationalCodes = find_empty(call_dbms(dbms_name, user, password, threshold), maximumLevels)
	emptyLeaf2DBMS(EmptyLocationalCodes, dbms_name, user, password, maximumLevels)

	print "Finished writing '"+lasFile+"' to octree database '"+dbms_name#+"': " + str(datetime.datetime.now())
	# flash("Finished writing '"+lasFile+"' to octree database '"+dbms_name+"': " + str(datetime.datetime.now()))

	# createConfigPY(dbms_name, user, password)

	# print 'the smallest leaf has an edge length of ' + str(minleafDomain)

	return minleafDomain, floorheight

if (__name__ == "__main__"):
	Pointless('firebrigade_zeb1.las', "1", "thesis", 8)