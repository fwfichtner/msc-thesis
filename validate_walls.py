"""
optional

This script fits planes in all leafs, builds 2D histograms of the vertical planes, checks whether there's a line (corresponds to wall) 
and then removes wrong assignments (within a threshold) from the database.
"""

import math
import numpy as np
import psycopg2
from collections import defaultdict
import matplotlib.pyplot as plt
from skimage.transform import probabilistic_hough_line # scikit-image
from skimage.feature import canny # scikit-image

from libs.findEqualNeighbours import giveMeAllEqualNeighbours
from find_stairs import make2dhistogram, svd_magic, is_vertical, angle_to_z

def retrieveDB(dbms_name, user, password, storey):
	# this function retrieves all points from the database within the current storey and puts them into a dictionary

	# connect to database
	con = psycopg2.connect("host='localhost' dbname='"+dbms_name+"' user='"+user+"' password='"+password+"'")
	cur = con.cursor()
	cur.execute("SELECT locationalcode, x, y, z FROM pointcloud WHERE locationalcode in (SELECT locationalcode FROM semantics WHERE storey = "+str(storey)+");")
	# if it's only for validating I could also only retrieve leafs which are already labelled as a wall
	data = cur.fetchall()

	con.commit()
	cur.close()

	storey_dict = defaultdict(list) # dictionary of all remaining locationalcodes with lists of points' coordinates

	xmax = 0 # max value in this direction, needed later for the bins...
	ymax = 0 # ...all histograms need to have the same shapes
	lookup_dict = {} # dictionary to reversely get the locational code I need
	for i in range(len(data)):
		storey_dict[data[i][0]].append([float(data[i][1]), float(data[i][2]), float(data[i][3])])

		if float(data[i][1]) > xmax: 
			xmax = float(data[i][1]) # find xmax
		if float(data[i][2]) > ymax:
			ymax = float(data[i][2]) # find ymax

	return storey_dict, int(xmax), int(ymax)

def hough_transform(H):
	# this function takes the 2D histogram, finds and returns lines

	print "let's do the hough and find some lines here"
	# http://scikit-image.org/docs/dev/auto_examples/plot_line_hough_transform.html
	# https://nabinsharma.wordpress.com/2012/12/26/linear-hough-transform-using-python/
	# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html


	builder = []
	for row in H:
		temprow = []
		for i in row:
			if i < 8: # clean data so it only finds edges when above a certain threshold
				i = 0
			temprow.append(i)
		builder.append(temprow)
	H = np.asanyarray(builder) # turn this row off to use normal H, there's too much noise in floor 1 though

	edges = canny(H) # for noise set sigma=1.8; Edges also very interesting for stairs also! < not absolutely necessary

	lines = probabilistic_hough_line(H, threshold=50, line_length=30, line_gap=5) # parameters to be set # threshold=50, line_length=5, line_gap=20

	"""showme"""
	# fig, (plt1, plt2, plt3) = plt.subplots(1, 3, sharex=True, sharey=True)
	# plt1.imshow(H,cmap='spectral')
	# plt1.set_title('vert hist')
	# plt2.imshow(edges,cmap=plt.cm.gray)
	# plt2.set_title('canny edges')
	# for line in lines:
	#     startpt, endpt = line
	#     plt3.plot((startpt[0], endpt[0]), (startpt[1], endpt[1]))
	# plt3.set_title('hough lines')
	# plt.show() # can't get rid of stupid white space

	return lines

def check_if_on_line(px, py, lines):
	# this function checks whether the leaf is directly on a line, it is not up-to-date and doesn't get called

	for line in lines:
		x_coords, y_coords = zip(*line) # get x0,x1 & y0,y1
		print x_coords, y_coords
		if x_coords[0]==x_coords[1]:
			if px==x_coords[1]: # slope y-intercept form doesn't exist as x0 is x1 <-- but why don't I test that from a Y-perspective?
				return True
		# test following part, add threshold!
		temp = np.vstack([x_coords,np.ones(len(x_coords))]).T
		m, c = np.linalg.lstsq(temp, y_coords)[0]
		if np.allclose((py-m*px), c): # it's not equal due to some rounding problems, but that should be fine
			return True
	return False

def distance_to_line(px, py, lines, t):
	# This function meassures the distance of the leaf to all lines, when withing Threshold returns True

	# http://mathworld.wolfram.com/Point-LineDistance2-Dimensional.html or Wikipedia

	for line in lines:
		x1 = line[0][0]
		y1 = line[0][1]
		x2 = line[1][0]
		y2 = line[1][1]

		distance = abs((x2-x1)*(y1-py)-(x1-px)*(y2-y1))/(math.sqrt(math.pow((y2-y1),2) + math.pow((x2-x1),2)))	
		if distance <= t:
			return True
		else:
			continue
	return False

def updatedb(dbms_name, user, password, storey, lines, maxwall, minleafDomain):
	# this function updates the results in the database, wrong assignments get set back to 0

	con = psycopg2.connect("host='localhost' dbname='"+dbms_name+"' user='"+user+"' password='"+password+"'")
	cur = con.cursor()

	cur.execute("SELECT x, y FROM semantics WHERE storey = "+str(storey)+" AND attribute = 1;")
	data = cur.fetchall()

	t = int((maxwall/minleafDomain)/2+0.5) # rounded up threshold for wall thickness

	temp = []
	i = 0
	for leaf in data:
		px = leaf[0]
		py = leaf[1]

		if i % 10000 == 0 and i > 0:
			print str(i)+' leafs checked'
		i += 1

		if distance_to_line(px, py, lines, t):
			continue
		else:
			temp.append((px, py))
			# cur.execute('UPDATE semantics SET attribute = NULL WHERE x = '+str(px)+' AND y = '+str(py)+' AND storey = '+str(storey)+';') # too slow, fastest is makeing one big string

	if not temp:
		return

	print "I'm updating the database now"
	first = True # it's ugly, but a hell lot faster to build the whole query first
	query_builder = ''
	for i in temp:
		if first:
			query_builder += 'x = '+str(i[0])+' AND y = '+str(i[1])+' AND storey = '+str(storey)+' AND attribute = 1'
			first = False
		else:
			query_builder += ' OR x = '+str(i[0])+' AND y = '+str(i[1])+' AND storey = '+str(storey)+' AND attribute = 1'

	cur.execute('UPDATE semantics SET attribute = NULL WHERE '+query_builder+';')
	con.commit()
	cur.close()

def vali(storey, dbms_name="thesis", user="postgres", password="", minleafDomain=0.05277552607, maxwall=0.3):
	"""
	This function calls all other functions and structures the workflow
	"""

	print 'validate walls and remove wrong assignments in the storey '+str(storey)+', this can take a while ...'
	# minleafDomain = 0.05277552607 # takes less long for testing, this has to come from PointlessConverter or add_semantics, zeb1
	# # minleafDomain = 0.0520286270439 # leica
	# maxwall = 0.3 # has to come from add_semantics or settings
	storey_dict, gridx, gridy = retrieveDB(dbms_name, user, password, storey)
	vert = []

	i = 0
	for each in storey_dict:
		neighbours_lst = [each] # puts also itself in there
		for neighbour in giveMeAllEqualNeighbours(each): # returns set with all neighbours of current node, from findNeighbours
			if neighbour in storey_dict:
				neighbours_lst.append(neighbour)
		N = svd_magic(neighbours_lst, storey_dict) # I fit planes to some of the leafs later again, maybe I could keep the results
		if N is not None: # if it's a non planar voxel with not enough points
			angle = angle_to_z(N)
			if is_vertical(angle) is True:
				vert.append(each)

	H = make2dhistogram(storey_dict, vert, gridx, gridy)

	lines = hough_transform(H.T) # take transpose of H
	
	updatedb(dbms_name, user, password, storey, lines, maxwall, minleafDomain)


if __name__ == '__main__':
	vali(1, "thesis_main", "postgres", "")