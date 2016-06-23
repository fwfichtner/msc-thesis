"""
This script builds the histograms from either the octree (of points) or the point cloud directly and find it's peaks. It also has the possibility to plot (show) the results.
"""

from liblas import file
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
# import datetime
# from scipy.signal import argrelextrema, find_peaks_cwt

from libs.detect_peaks import detect_peaks
# from libs.findpeaks import findpeaks

def create_table(dbms_name, user, password):
	# creates or replaces table semantics in DB

	# create table semantics
	con = psycopg2.connect("host='localhost' dbname='"+dbms_name+"' user='"+user+"' password='"+password+"'")
	con.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT) # necessary to catch error (DROP TABLE) --> http://initd.org/psycopg/docs/usage.html#transactions-control
	cur = con.cursor()
	# print 'start slow', str(datetime.datetime.now())
	try:
		cur.execute('DROP TABLE semantics;')		
		print "table 'semantics' already exists, replacing table"
	except:
		pass
	# table with distinct locationalcode and origin coordinates of leaf
	# this takes really long compared to the rest, INDEX doesn't make it faster
	cur.execute('CREATE TABLE semantics AS SELECT DISTINCT locationalcode, ((x::float-0.5)::int) AS x, ((y::float-0.5)::int) AS y, ((z::float-0.5)::int) AS z FROM pointcloud;')
	# x, y, z are casted and rounded to the next lower int (thus -0.5)
	cur.execute('ALTER TABLE semantics ADD COLUMN storey integer;')
	cur.execute('ALTER TABLE semantics ADD COLUMN attribute integer;')
	con.commit()
	cur.close()
	# print 'stop slow', str(datetime.datetime.now())

	"""
	attribute: floor=0, wall=1, stairs=2
	"""

def retrieveDB(dbms_name, user, password, direction):
	# retrieve octree leafs (per point > pointcloud), alternatively from table semantics (which first needs to be created) with only leafs, result very similar
	"""
	create_table() needs to get turned on after testing, alternatively can be switched off to lower execution time
	"""
	create_table(dbms_name, user, password) # here table semantics gets created
	# connect to database
	con = psycopg2.connect("host='localhost' dbname='"+dbms_name+"' user='"+ user + "' password='"+ password + "'")
	cur = con.cursor()

	# Number of bins in direction (without empty)
	# cur.execute("SELECT max(cast("+direction+" AS float)) FROM pointcloud;")
	cur.execute("SELECT max(cast("+direction+" AS float)) FROM semantics;")

	# Max number of bins in all directions (including empty)
	# cur.execute("SELECT greatest(max(cast(x AS float)), max(cast(y AS float)), max(cast(z AS float))) FROM pointcloud;")
	b = cur.fetchone()[0]
	b = int(b) + 1 # it will be used in range and we want to include the highest bin as well

	# fetch data for histogram
	# cur.execute("SELECT "+direction+" FROM pointcloud;")
	cur.execute("SELECT "+direction+" FROM semantics;")
	temp = cur.fetchall()
	data = []
	for each in temp:
		data.append(int(float(each[0]))) #snap to boundary (to get origin), unnecessary for table semantics
		# data.append(float(each[0]))

	con.commit()
	cur.close()

	return data, b

def retrieveLAS(lasFile, direction, minleafDomain):
	# scaled histogram of path of scanner
	temp = []
	data = []
	f = file.File(lasFile, mode='r')

	if direction is "x": 
		for point in f:
			temp.append(point.x)
	if direction is "y":
		for point in f:
			temp.append(point.y)
	if direction is "z":
		for point in f:
			temp.append(point.z)

	# minimum should not be negative
	if min(temp) < 0:
		t = min(temp) * (-1) # to make it positive
		bins = int((max(temp)+t)/minleafDomain)
		for each in temp:
			data.append((each+t)/minleafDomain)
	else:
		bins = int(max(temp)/minleafDomain)
		for each in temp:
			data.append(each/minleafDomain)

	return data, bins, min(temp)

def makehistogram(data, b):
	"""
	more information and helpful explainations: http://stackoverflow.com/questions/9141732/how-does-numpy-histogram-work
	--------------
	res = np.histogram([1, 2, 1], bins=[0, 1, 2, 3])
	showme([1, 2, 1], 4)
	"""
	res = np.histogram(data, bins=np.arange(b))
	try:
		res = np.histogram(data, bins=np.arange(b))
		# res = np.histogram(data, bins=range(b))
	except TypeError:
		res = np.histogram(data, bins=b) # because in LAS list instead of int (in DB)
	return res

def showme(ldata, lrange, direction, ind, histo, lim, parameter=None):
	# to show histogram with peaks
	
	histoind = histo[ind]
	# if parameter is not None: # data preparation, necessary when point cloud other scale of bins (doesn't happen with octree)
	# 	ind = ind*parameter

	try:
		plt.hist(ldata, bins = range(lrange)) 
	except TypeError:
		plt.hist(ldata, bins = lrange) # because in point cloud list instead of int

	plt.xlabel(direction +'-distance from origin')
	plt.ylabel('obstacles & walls')
	plt.title('point cloud histogram')
	plt.plot(ind, histoind, '+', mfc=None, mec='r', mew=2, ms=8)
	if lim is not None: #added line to show above where peaks are found
	    plt.axhline(lim, color='r')
	plt.show()

def peaks(histarray, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, show=False): # put standard settings here and at the end make those variables settings to choose
	"""
	https://github.com/MonsieurV/py-findpeaks <-- many different methodologies to check out
	---------------
	* ScyPy find_peaks_cwt does weird stuff, but generally works
	* findpeaks lightweight and fast, limited settings, show not integrated
	""" 
	# square histarray to remove outliers (bogus detections)?

	# this needs to be adaptle according to the input data, above this height (95% of the data values) peaks are found
	# maybe loop until number of walls are found? <-- if number known before
	lim = np.percentile(histarray,95) # <-- parameter

	# this only gives all maxima, no settings possible
	# maxima = argrelextrema(temp[0], np.greater)[0]

	# findpeaks --> "Janko Slavic, https://github.com/jankoslavic/py-tools/tree/master/findpeaks"
	# ind = findpeaks(histarray, spacing=5, limit=5000)

	# detect_peaks --> "Marcos Duarte, https://github.com/demotu/BMC"
	"""
	possible settings:
	mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
	"""

	# add 0 on both sides of histarray to prevent edge peaks not getting detected
	histarray = np.hstack((0, histarray))
	histarray = np.append([histarray],[0])

	indtemp = detect_peaks(histarray, mph=lim, mpd=3, threshold=0, edge='rising') # <-- parameter

	# move everything 1 to the left again because of 0 added at the beginning to detect edge peaks
	temp = []
	for each in indtemp:
		temp.append(each - 1)
	ind = np.array(temp)

	return ind, lim

def returnpeaksZ_las(filename, minleafDomain, show=False, direction='z'):
	# to be called from other files, kind of like main if not main, called from add_semantics.py
	data, b, minheight = retrieveLAS(filename, direction, minleafDomain)
	res = makehistogram(data, b)
	histo = res[0]
	ind, lim = peaks(histo)
	if show:
		showme(data, b, direction, ind, histo, lim)
	return ind, histo, minheight


def returnpeaksZ(dbms_name="thesis", user="postgres", password="", direction="z", show=False):
	# to be called from other files, kind of like main if not main, called from add_semantics.py
	data, b = retrieveDB(dbms_name, user, password, direction) 
	res = makehistogram(data, b)
	histo = res[0]
	ind, lim = peaks(histo)
	if show:
		showme(data, b, direction, ind, histo, lim)
	return ind, histo

def returnpeaksXY(direction, data, b, show=False):
	# to be called from other files, kind of like main if not main, called from add_semantics.py

	res = makehistogram(data, b)
	histo = res[0]
	ind, lim = peaks(histo)
	if show:
		showme(data, b, direction, ind, histo, lim)
	return ind, histo

def main(dbms_name="thesis", user="postgres", password="", direction="z", db="", minleafDomain = 0.0527755282819):
	# main function for testing purposes

	if db:
		parameter = 1 # overwrite in case forgotton to change
		data, b = retrieveDB(dbms_name, user, password, direction)
	else:
		data, b, minheight = retrieveLAS('firebrigade_tango.las', direction, minleafDomain)

	res = makehistogram(data, b)
	histo = res[0]

	ind, lim = peaks(histo)

	showme(data, b, direction, ind, histo, lim)

if __name__ == '__main__':
	# main("thesis", "postgres", "", "x", db=True, parameter=0.02)
	# main("thesis", "postgres", "", "y", db=True, parameter=0.02)
	main("thesis", "postgres", "", "z", db=True, minleafDomain = 0.0581947583705)