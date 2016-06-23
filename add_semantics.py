"""
This script enriches the octree semantically with the classification storey, floor (0), wall (1) and writes these to the database
"""

import numpy as np
import psycopg2

from histograms import returnpeaksZ, returnpeaksXY, returnpeaksZ_las
from libs.PointlessConverter import Pointless

def thickness(maxwall, minleafDomain, edge='rising'): # edge parameter should be selected in main function

	"""
	for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both')
	"""
	# scale = 1/minleafDomain # in case scaling gets necessary, but relations stay the same, so maybe never
	bincount = round((maxwall/2)/minleafDomain) # <-- parameter

	# left/right are bins next to peak that will also be assigned wall/floor
	if edge is 'rising': 
		if bincount < 2:
			left = 0
			right = 0
		if bincount >= 2:
			left = bincount - 2 
			right = bincount - 1
	if edge is 'falling':
		print 'setting for left and right still need to be set'
		pass # check if this ever plays a role or is selected
	if edge is 'both':
		print 'setting for left and right still need to be set'
		pass # check if this ever plays a role or is selected

	return left, right

def build_wall(dbms_name, user, password, direction, storey, show):
	# this function takes the X,Y direction of the histogram, seperated by storey and return histograms and peaks
	# connect to database
	con = psycopg2.connect("host='localhost' dbname='"+dbms_name+"' user='"+user+"' password='"+password+"'")
	cur = con.cursor()

	# Number of bins in direction (without empty), this is different here for every direction
	cur.execute("SELECT max(cast("+direction+" AS float)) FROM semantics WHERE storey = "+str(storey)+";")
	b = cur.fetchone()[0]
	b = int(b)+1 # it will be used in range and we want to include the highest bin as well

	# cur.execute("SELECT min(cast("+direction+" AS float)) FROM semantics WHERE storey = "+str(storey)+";")
	# lower = cur.fetchone()[0] # <-- needed to translate ind back later
	# int(b) = upper - lower

	# fetch data for histogram
	cur.execute("SELECT "+direction+" FROM semantics WHERE storey = "+str(storey)+";")
	temp = cur.fetchall()
	data = []
	for each in temp:
		data.append(int(float(each[0])))

	con.commit()
	cur.close()

	# make histogram for x & y
	ind, histo = returnpeaksXY(direction, data, b, show)
	return ind, histo

	# write to db in project_to_octree()

def close_peaks(ind, right):
	# tests whether two peaks are too close to have empty space in between. If yes it removes the second peak as this peak doesn't indicate another storey or wall.
	ind = np.msort(ind) # necessary to get real distances
	close_peaks = False
	distance = [] # list with bin distance of peaks
	for i in range(len(ind)-1):
	 	distance.append(ind[i+1]-ind[i])
	temp = [] # list with index of items which have ot be deleted, because distance to low
	for i in distance:
		if i <= right: # right parameter identifies it as one wall
			temp.append(distance.index(i)+1)
			close_peaks = True
	if close_peaks is True:
		print 'two close peaks --> one floor'
		return np.delete(ind,temp) # second peak deleted, continue with newind, should right be made bigger???
	else:
		return None

def walls_obstacle(peaksdict, maxwall, minleafDomain):
	# the main walls are more likely to appear in multiple storeys, that's why it is important to identify them.
	# peaks above each other are more likely mainwalls
	# peaks close to main walls are more likely obstacle and not another wall.

	# make mainwalls
	mainwall = set()
	allpeaks = []
	for each in peaksdict:
		# make first and last peak a mainwall
		mainwall.add(peaksdict[each][0])
		mainwall.add(peaksdict[each][-1])	
		# make peaks above each other mainwall
		"""
		If the building is bigger there should be a constraint so that it only compares to adjacent floors and not the entire building
		Is this logical?
		"""
		for i in peaksdict[each]:
			if i in allpeaks or i+2 in allpeaks or i-2 in allpeaks: # in Leica scan 1 should be enough
				mainwall.add(i)
			allpeaks.append(i)

	# make close peak to mainwall obstacle
	f = int(0.5 / minleafDomain + maxwall / minleafDomain) # obstacle constraint, an item within half a meter from a mainwall <-- parameter
	obstacle = {}
	for each in peaksdict:
		distance = []
		for i in range(len(peaksdict[each])-1):
			distance.append(peaksdict[each][i+1]-peaksdict[each][i])
		for i in distance:
			if i <= f:
				if peaksdict[each][distance.index(i)] in mainwall: # if detected item is mainwall, next (close) peak will be obstacle
					if each not in obstacle:
						obstacle[each]=[peaksdict[each][distance.index(i)+1]]
					else:
						obstacle[each].append(peaksdict[each][distance.index(i)+1])
				elif peaksdict[each][distance.index(i)+1] in mainwall: # else if next (close) peak in mainwall, detected item will be obstacle
					if each not in obstacle:
						obstacle[each]=[peaksdict[each][distance.index(i)]]
					else:
						obstacle[each].append(peaksdict[each][distance.index(i)])
	# remove obstacle from peaksdict
	for each in obstacle:
		if each in peaksdict:
			peaksdict[each]=list(set(peaksdict[each])-set(obstacle[each])) # order gets lost!!!

	# does it make sense to return mainwalls and obstacle, because all I have left will be assigned obstacle
	# do something with obstacle information and write it to DB as obstacle?
	return peaksdict

def project_to_octree(dbms_name, user, password, ind, left, right, direction, storey):
	# this function projects the attribute '1' for wall to the octree in the semantics table

	# when two walls (peaks) are closer together than the maximum wall thickness, the space in between is not a room
	newind = close_peaks(ind, right)
	if newind is None:
		pass
	else:
		return project_to_octree(dbms_name, user, password, newind, left, right, direction, storey)

	# connect to database
	con = psycopg2.connect("host='localhost' dbname='"+dbms_name+"' user='"+user+"' password='"+password+"'")
	cur = con.cursor()

	# write wall attribue (1) to semantics database
	for i in ind:
		lefter = str(i - int(left)) # preparation to set attribute for wall (1), lefter and righter end
		righter = str(i + int(right))
		cur.execute('UPDATE semantics SET attribute = 1 WHERE locationalcode IN (SELECT locationalcode FROM semantics WHERE '+direction+' >= '+lefter+' AND '+direction+' <= '+righter+' AND storey = '+str(storey)+' AND attribute IS NULL);')
	con.commit()
	cur.close()
	
	print 'in '+direction+'-direction '+str(len(ind))+' walls are found in floor level '+str(storey)
	# return ind

def real_storey(ind, hist, minleafDomain, floorheight, scannerpathfile=None):
	# this function tests whether a peak relly represents a storey

	if scannerpathfile is not None: # this obviously only works when there is a path of the scanner available, why not just if scannerpathfile ???
		# this function returns the peaks with which are located within a threshold under the path of the scanner

		# returns the peaks, histogram and smallest height of the path of the scanner, peaks represent where the scanned has moved the longest, thus is a storey
		pathind, pathhisto, pathminheight = returnpeaksZ_las(scannerpathfile, minleafDomain, show=False)
		k = ind

		path_to_floor = abs(int((floorheight - pathminheight)/minleafDomain)) # floor is represented by the lowest point rather than by the lowest peak, has more integrity
		pathind = [ i + path_to_floor for i in pathind]

		dupl = set(ind).intersection(pathind) # find duplicates
		if len(dupl) > 0:
			print "floorpeak and pathpeak at the same height, we'll remove it for you"
			for i in dupl:
				ind.remove(i)

		# average shoulder height is about 140cm +30cm for Zeb1
		t_min = 1.4/minleafDomain
		t_max = 1.7/minleafDomain

		merged = sorted(list(ind) + pathind)
		newind = [] # new list with final storey peaks
		for i in pathind:
			pos = merged.index(i)
			dist = merged[pos] - merged[pos-1]
			for floorpeak in ind:
				if floorpeak < merged[pos]: # should be smaller obviously as you walk on it
					if merged[pos] - floorpeak > t_min: # should be in between threshold
						if merged[pos] - floorpeak < t_max:
							newind.append(floorpeak)
		newind.append(ind[-1]) # add roof again, historic reasons and we want those points to be labelled
		return newind

	else:
		# this function tests if a peak really represents a storey, if the height of the storey is not big enough the lower peak gets removed
		# fire brigade situation less likely then a table, how about hanging ceilings?
		# this only works with one occurance of this phenomena, if more the others get ignored. Could go into a loop like close_peaks

		distance = []
		for i in range(len(ind)-1):
		 	distance.append(ind[i+1]-ind[i])
		for i in distance:
			if i * minleafDomain < 1.8: # here we set the minimum storey height to 1.8 meters <-- parameter
				if hist[ind[distance.index(i)]] > hist[ind[distance.index(i)+1]]:
					return np.delete(ind, distance.index(i)+1)
				else:
					return np.delete(ind, distance.index(i))	
		return ind

def seperate_storeys(dbms_name, user, password, ind, left, right, hist, minleafDomain, scannerpathfile, floorheight):
	# this function seperates the storeys and projects the attribute '0' for floor to the octree in the semantics table

	# when two walls (peaks) are closer together than the maximum wall thickness, the space in between is not a storey
	newind = close_peaks(ind, right)
	if newind is None:
		pass
	else:
		return seperate_storeys(dbms_name, user, password, newind, left, right, hist, minleafDomain, scannerpathfile, floorheight)

	ind = real_storey(ind, hist, minleafDomain, floorheight, scannerpathfile) # it's questionable whether this function is always necessary, if more than once it needs to be in a loop

	# connect to database
	con = psycopg2.connect("host='localhost' dbname='"+dbms_name+"' user='"+user+"' password='"+password+"'")
	cur = con.cursor()

	# write semantics to database, seperate by storey and put attribute floor (0) in there
	first = True
	for i in range(len(ind)):
		if i == len(ind)-1: # roof, doesn't start a new storey (& for pathfinding purposes unnecessary, if semantics wanted > make special query out of loop)
			break
		s = str(i) # storey
		f = str((ind[i])-int(left)) # floor
		c = str((ind[i+1])-int(left)) # ceiling
		lower = str(int(ind[i]) - int(left)) # preparation to set attribute for floor (0), lower and higher end
		upper = str(int(ind[i]) + int(right))
		cur.execute('UPDATE semantics SET storey = '+s+' WHERE z IN (SELECT z FROM semantics WHERE z >= '+f+' AND z < '+c+');')
		cur.execute('UPDATE semantics SET attribute = 0 WHERE z IN (SELECT z FROM semantics WHERE z >= '+lower+' AND z <= '+upper+');')
	con.commit()
	cur.close()

	n = int(s)+1 # len(ind)-1
	print n, "storeys found"
	return n

def main(dbms_name="thesis", user="postgres", password="", file='firebrigade_zeb1.las', maxwall=0.3):
	# main function to run and control the other ones
	
	if 'zeb1' in file: # exists only for Zeb1, please also put it in folder
		scannerpathfile = file[:-4]+'_traj'+file[-4:]
	else:
		scannerpathfile = None
	minleafDomain, floorheight = Pointless(file, "1", dbms_name, 8) # retrieve from PointlessConverter.py
	# floorheight = -7.98559331894
	# minleafDomain = 0.0545399934053
	# minleafDomain = 0.05277552607 # takes less long for testing, this has to come from PointlessConverter, zeb1
	# floorheight = -7.9910607338 # zeb1, for path of the scanner
	# minleafDomain = 0.0520286270439 # leica
	# maxwall = 0.3 # maximum wall thickness (not scaled), parameter to set manually
	left, right = thickness(maxwall, minleafDomain)

	indz, histz = returnpeaksZ(dbms_name, user, password, "z", show=False)
	
	# indz = np.array([ 10,70,71,117,119,161]) # test array
	storeys = seperate_storeys(dbms_name, user, password, indz, left, right, histz, minleafDomain, scannerpathfile, floorheight)

	# ind = np.array([ 21, 30, 37, 179, 216]) # test array
	# for i in xrange(0,storeys): # without wallsfuniture()
	# 	ind, hist = build_wall(dbms_name, user, password, "x", i, show=False)
	# 	project_to_octree(dbms_name, user, password, ind, left, right, "x", i)
	# 	ind, hist = build_wall(dbms_name, user, password, "y", i, show=False)
	# 	project_to_octree(dbms_name, user, password, ind, left, right, "y", i)


	peaksdictx = {}
	peaksdicty = {}
	for i in xrange(0,storeys):
		indx, histx = build_wall(dbms_name, user, password, "x", i, show=False)
		indy, histy = build_wall(dbms_name, user, password, "y", i, show=False)
		peaksdictx[i] = indx
		peaksdicty[i] = indy

	walls_obstacle(peaksdictx, maxwall, minleafDomain)
	walls_obstacle(peaksdicty, maxwall, minleafDomain)

	for i in peaksdictx:
		project_to_octree(dbms_name, user, password, peaksdictx[i], left, right, "x", i)
	for i in peaksdicty:
		project_to_octree(dbms_name, user, password, peaksdicty[i], left, right, "y", i)

	return minleafDomain, storeys

if __name__ == '__main__':
	main("thesis_main", "postgres", "")