"""
This script locates the stairs and writes the octree classification (2) to the database.
"""

import math
import numpy as np
import psycopg2
from collections import defaultdict
# from matplotlib.mlab import PCA
import matplotlib.pyplot as plt
# import scipy.ndimage as ndi
from scipy import signal, stats

from libs.findEqualNeighbours import giveMeAllEqualNeighbours

def retrieveDB(dbms_name, user, password, storey):
	"""
	this function retrieves necessary data from the database
	it returns a dictionary storey_dict {locationalcode: [x,y,z]}, max value for x/y-direction, a dictionary lookup_dict for the projection to the octree later {(x,y):(locationalcode,z)}
	"""
	
	# connect to database
	con = psycopg2.connect("host='localhost' dbname='"+dbms_name+"' user='"+user+"' password='"+password+"'")
	cur = con.cursor()
	# cur.execute("SELECT locationalcode, x, y, z FROM pointcloud WHERE locationalcode in (SELECT locationalcode FROM semantics WHERE storey = "+str(storey)+");")
	cur.execute("SELECT locationalcode, x, y, z FROM pointcloud WHERE locationalcode in (SELECT locationalcode FROM semantics WHERE storey = "+str(storey)+" AND attribute is NULL);")

	# for radius and POSTGIS http://postgis.net/docs/ST_MakePoint.html & http://postgis.net/docs/ST_3DDWithin.html can be used, but the whole thing might be super expensive to bring in form first

	data = cur.fetchall()


	con.commit()
	cur.close()

	storey_dict = defaultdict(list) # dictionary of all remaining locationalcodes with lists of points' coordinates
	xmax = 0 # max value in this direction, needed later for the bins...
	ymax = 0 # ...all histograms need to have the same shapes
	lookup_dict = {} # dictionary to reversely get the locational code I need
	for i in range(len(data)):
		storey_dict[data[i][0]].append([float(data[i][1]), float(data[i][2]), float(data[i][3])]) #  {locationalcode: [x,y,z]}

		if float(data[i][1]) > xmax: 
			xmax = float(data[i][1]) # find xmax
		if float(data[i][2]) > ymax:
			ymax = float(data[i][2]) # find ymax

		if (int(float(data[i][1])),int(float(data[i][2]))) not in lookup_dict: # lookup_dict gets created with tuple of XY-coordinates as key, it is necessary to project the information from the 2d histograms back to the octree
			lookup_dict[(int(float(data[i][1])),int(float(data[i][2])))] = (data[i][0], int(float(data[i][3]))) # (x,y):(locationalcode,z)

	return storey_dict, int(xmax), int(ymax), lookup_dict

def covarianceMatrix(storey_dict):
	# doesn't get called, but might be useful to detect whether the estimation matrix is badly conditions to filter out non planar leafs
	"""
	Takes the leaf's points from storey_dict
	Returns normal (N) of the plane through those points

	~ should have same result as svd_magic(), but slower ~
	"""

	for i in storey_dict: # this needs to be changed and just this function called
		if len(storey_dict[i]) >= 3:
			# data = np.asarray(storey_dict[i])
			data = np.asarray([[0,0,0],[0,1,0],[0,0,1],[1,0,1]])

			covi = np.cov(data.T) # transpose?
			eigenValues,eigenVectors = np.linalg.eig(covi)
			N = eigenVectors[:, eigenValues.argmin()] # eigenVector corresponding to least eigenValues, should be the Normal
			
			# idx = eigenValues.argsort()[::-1] # sort from largest to smallest, same result as above
			# eigenValues = eigenValues[idx]
			# eigenVectors = eigenVectors[:,idx]

			return N
			break

def svd_magic(neighbours_lst, storey_dict):
	"""
	This function returns the normal N of the plane fitted through the points in the leaf and its neighbours. 
	"""

	data = []
	for i in neighbours_lst: # collecting also the neighbours' points
		if len(storey_dict[i]) >= 1: # maybe leave this thing out
			for each in storey_dict[i]:
				data.append(each)

	data = np.asarray(data)

	if len(data) < 4: # could also be set to 2, checks if there are enough points to fit a plane
		return None

	centroid = np.average(data, axis=0) # Find the average of points, could also take the centroid of the main voxel?

	centroid_to_point = data - centroid # centroid to point vector matrix

	u,s,v = np.linalg.svd(centroid_to_point) # Singular value decomposition
	N = v[-1] # The last row of V matrix indicate the eigenvectors of smallest eigenvalues.
	# N = v.conj().transpose()[:,-1] # same

	# print N
	return N

def angle_to_z(v1):
    """ 
    Returns the angle in radians between vectors 'v1' and 'v2'
    http://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    """
    v2 = np.asarray([0,0,1]) # z-axis

    # math.degrees(math.acos(np.dot(v1, v2)/(math.sqrt(np.dot(v1,v1))*math.sqrt(np.dot(v2,v2))))) # doesn't work when parallel

    uv1 = v1 / np.linalg.norm(v1) # unit vector of v1
    uv2 = v2 / np.linalg.norm(v2) # unit vector of v2
    angle = np.arccos(np.clip(np.dot(uv1, uv2),-1,1)) # calculates angle, avoids rounding issues

    return math.degrees(angle) # convert angle from radians to degrees

def is_vertical(angle):
	# plane is parallel to z-axis <-- normal, parallel
	if angle > 80:
		if angle < 100:
			return True
	else:
		return False

def is_horizontal(angle):
	# plane is perpendicular z-axis <-- normal, perpendicular
	if angle < 10:
		return True
	elif angle > 170:
		return True
	else:
		return False

def make2dhistogram(storey_dict, orientation_lst, gridx, gridy):
	"""
	returns 2D histogram
	"""

	x_array = []
	y_array = []
	for each in orientation_lst:
		if each in storey_dict: # necessary?
			for i in storey_dict[each]:
				x_array.append(int(i[0]))
				y_array.append(int(i[1]))

	H, xedges, yedges = np.histogram2d(x_array, y_array, bins=(gridx, gridy))

	""" zoom to stairs firebrigade zeb1, story 0 """
	# x_array = []
	# y_array = []
	# for each in orientation_lst:
	# 	if each in storey_dict: # necessary?
	# 		for i in storey_dict[each]:
	# 			if int(i[0]) < 195 and int(i[0]) >= 113 and int(i[1]) >= 42:
	# 				x_array.append(int(i[0]))
	# 				y_array.append(int(i[1]))
	# gridx = np.linspace(min(x_array),max(x_array),max(x_array)-min(x_array))
	# gridy = np.linspace(min(y_array),max(y_array),max(y_array)-min(y_array))

	# H, xedges, yedges = np.histogram2d(x_array, y_array, bins=(gridx, gridy))

	""" show histogram """
	# im = plt.imshow(H, cmap="spectral") # origin='low' turns it around, spectral sets colours --> http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.imshow imshow turns axes around, doesn't matter as long as real values stay > transpose data & make origin lower
	# plt.colorbar()
	# plt.show()

	return H

def create_matched_filter(minleafDomain):
	"""
	This function creates the matched filter for the stairs. As we look from above the step_tread plays a role, even though not mentioned in the paper.
	I returns a filter in the following form and shape to represent the stairs (leaf_count 5, minstepcount 3, width 6):
	[[ 1. -1. -1. -1. -1. -1.  1. -1. -1. -1. -1. -1.  1. -1. -1. -1. -1. -1.]
	 [ 1. -1. -1. -1. -1. -1.  1. -1. -1. -1. -1. -1.  1. -1. -1. -1. -1. -1.]
	 [ 1. -1. -1. -1. -1. -1.  1. -1. -1. -1. -1. -1.  1. -1. -1. -1. -1. -1.]
	 [ 1. -1. -1. -1. -1. -1.  1. -1. -1. -1. -1. -1.  1. -1. -1. -1. -1. -1.]
	 [ 1. -1. -1. -1. -1. -1.  1. -1. -1. -1. -1. -1.  1. -1. -1. -1. -1. -1.]
	 [ 1. -1. -1. -1. -1. -1.  1. -1. -1. -1. -1. -1.  1. -1. -1. -1. -1. -1.]]

	The question remains if this filter adds anyhting to the result or it just works even though I use it
	"""
	step_tread = 0.25 # <-- parameter, average is usually a bit more than in this example (29 cm)
	leaf_count = round(step_tread/minleafDomain) # round, number of leafs in octree per step
	minstepCount = 3 # <-- parameter
	# leaf_count = 1 # this would make the filter closer to the one described in the paper
	onestep = []
	for each in xrange(minstepCount):
		onestep.append(1.0)
		for i in xrange(int(leaf_count)): # hier sind es 5
			onestep.append(-1.0)

	temp = []
	for i in xrange(6): # width of filter
		temp.append(onestep)

	filt = np.asarray(temp)

	# Prewitt Filter
	# filt=np.array([[1.0, 0.0, -1.0],[1.0, 0.0, -1.0],[1.0, 0.0, -1.0],])
	return filt

def filtervertical(H, minleafDomain):
	"""
	This function applies the matched filter on the horizontal 2D histogram.
	Returns filter response for both directions, as stairs can face both ways.
	https://www.youtube.com/watch?v=S7qbelm_4Y8 --> explains matched filter good, couldn't make spectralpython matched filter work
	"""
	# from skimage.feature import canny
	# edges = canny(H)
	# plt.imshow(edges,cmap=plt.cm.gray)
	# plt.show()

	filt = create_matched_filter(minleafDomain)
	# Note that the convolution of the time-reversed wavelet is identical to cross-correlation of the wavelet with the wavelet (autocorrelation) in the input signal --> http://crewes.org/ForOurSponsors/ResearchReports/2002/2002-46.pdf
	
	filty=np.transpose(filt) # transpose to also get stairs in other direction
	fr1=signal.convolve(H,filt, mode='same')

	# plt.subplot(1,3,1)
	# plt.imshow(fr1,cmap='spectral',interpolation='none')
	# plt.title('vert matched filter')
	# plt.colorbar()

	fr2=signal.convolve(H,filty, mode='same')
	# fr3=signal.convolve2d(H,filty, mode='same') # should givve the same result as fr2
	# fr3 = fr3**2 # doesn't really work as there are very high 'outliers' that just take everything away

	# plt.subplot(1,3,2)
	# plt.imshow(fr2,cmap='spectral',interpolation='none')
	# plt.title('vert matched filter transpose')
	# plt.colorbar()

	return fr1, fr2

def filterhorizontal(H):
	"""
	This function creates and applies a boxcar filter on the horizontal 2D histogram.
	Returns filter response
	"""
	# https://www.hdm-stuttgart.de/~maucher/Python/ComputerVision/html/Filtering.html#id1 <-- explains a lot about filters!

	filt=np.array([[0.0, 0.0, 0.0, 0.0],[0.0, 1.0, 1.0, 0.0],[0.0, 1.0, 1.0, 0.0],[0.0, 0.0, 0.0, 0.0]]) # boxcar filter
	frbc=signal.convolve(H,filt, mode='same')

	# plt.subplot(1,3,3)
	# plt.imshow(frbc,cmap="spectral",interpolation='none')
	# plt.title('hori boxcar filter')
	# plt.colorbar()
	# plt.show()

	return frbc

def combine_fr(frbc, fr1, fr2):
	# combines all filter responses to find possible stair areas
	"""with gmean 0 in one fr makes it all 0 > raises RuntimeWarning""" 
	import warnings
	warnings.filterwarnings("ignore")

	builder = []
	for row in xrange(frbc.shape[0]): # all filter responses have the same shape
	# row = 0
		temprow = []
		for column in xrange(frbc.shape[1]):
			temp = stats.gmean([abs(frbc[row,column]),abs(fr1[row,column]),abs(fr2[row,column])]) # gmean of all arrays, this would raise a warning
			if temp > 100: # threshold for binary
				temp = 0
			if temp > 0:
				temp = 1
			temprow.append(temp) # take abs of corresponding value of each fr and calculate geometric mean
		builder.append(temprow)
 	combi_fr = np.asanyarray(builder) # returns binary array

 	"""show combined filter responses"""
	# plt.imshow(combi_fr.T,cmap='spectral',interpolation='none', origin='lower') # <-- this would show the actual orientation of the data

	# plt.imshow(combi_fr,interpolation='none')
	# plt.title('combined filter responses')
	# # plt.colorbar()
	# plt.show()
	return combi_fr

def grow_region(combi_fr, minleafDomain):
	"""
	grows regions of filter responses to treat possible stairs as one area
	returns regions if their area or length is above threshold
	"""
	# http://www.scipy-lectures.org/packages/scikit-image/#binary-segmentation-foreground-background
	from skimage import measure, filters

	row = combi_fr.shape[0]
	column = combi_fr.shape[1]
	combi_fr = filters.gaussian_filter(combi_fr, sigma=column / (4. * row))
	regions = combi_fr > 0.7 * combi_fr.mean()

	# http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label
	regions_labels = measure.label(regions, background=0, return_num=False) # background=0, so only foreground gets labelled

	l_regions = []
	
	properties = measure.regionprops(regions_labels) # http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
	t_area = 5 / minleafDomain # threshold of 5 sqm as minimum for stairs
	t_len = 0.7 / minleafDomain # threshold of at least 0.6 meters in all directions of stairs
	for prop in properties:
		if prop.area < t_area:
			regions_labels[regions_labels == prop.label] = -1 # delete too small areas
		if prop.bbox[2]-prop.bbox[0] < t_len or prop.bbox[3]-prop.bbox[1] < t_len:
			regions_labels[regions_labels == prop.label] = -1 # delete too small areas
		else:
			temp = (prop.bbox, prop.coords)
			# prop.filled_image # -> cut to region
			l_regions.append(temp) # coords gives me all coordinates for one reagion as ndarray

	"""showme"""
	# plt.imshow(regions_labels, cmap='spectral')
	# plt.axis('off')
	# plt.tight_layout()
	# plt.show()

	return l_regions

def find_outliers(data, m=2.):
	"""
	finds an returns outliers
	--> http://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
	linear regression and least squares gave worse result or data was not used properly
	"""
	d = np.abs(data - np.median(data)) # absolut distance to median
	d_med = np.median(d) # median of the distances

	if d_med < 0.1: # almost no slope, so can't be stairs. When most distances to median are very small
		return False

	s = d/d_med if d_med else 0. # distances scaled by their median

	if isinstance(data[s>m],(int,long)):
		return set([data[s>m]]) # if only one of them to prevent iteration (type-)error

	return set(data[s>m]) # return outliers

def leastsquares4outliers(data):
	"""
	alternative to find_outliers --> doesn't get called
	to be called with the slope in each direction, but didn't give expected results when not enough points available
	"""

	# prepare data
	x = np.arange(len(data))
	y = data

	# Imports 
	from statsmodels.formula.api import ols # ordinary least squares

	# Make fit
	regression = ols("data ~ x", data=dict(data=y, x=x)).fit()
	
	# Find outliers
	test = regression.outlier_test()
	outliers = ((x[i],y[i]) for i,t in enumerate(test.iloc[:,2]) if t < 0.5)
	return list(outliers)

def rows_w_0(data, p=90.0):
	"""this part removes all rows and columns with almost only 0, not stair shaped"""

	tempX = []
	for idx, row in enumerate(data):
		count = np.unique(row, return_counts=True)
		if count[0][0] == 0:
			if count[1][0] > (data.shape[1]/100.0)*p: # 0 has to be in at least p% of all column in this row
				tempX.append(idx)

	tempY = []
	for idx, row in enumerate(data.T):
		count = np.unique(row, return_counts=True)
		if count[0][0] == 0:
			if count[1][0] > (data.shape[0]/100.0)*p: # 0 has to be in at least p% of all column in this row
				tempY.append(idx)

	data = np.delete(data, tempX, axis=0)
	data = np.delete(data, tempY, axis=1)

	return data

def calculate_slope(l_regions, lookup_dict):
	"""
	this function takes the regions, it then removes the outliers and calculates the slope and aspect of the areas. 
	If they're found to be stairs they get returned

	calculating np.gradient does not give a clear slope
	"""
	toDB = []
	if len(l_regions)<1:
		print 'there are no stairs in this storey'
		return

	for region in l_regions:
		"""prepare data"""
		l_reg = []
		x = []
		y = []
		z = []
		x_ll = region[0][0]
		x_ur = region[0][2]
		y_ll = region[0][1]
		y_ur = region[0][3]
		w = x_ur-x_ll
		h = y_ur-y_ll
		data = np.zeros((w, h)) # empty array in shapes of bbox of region

		to_find_outliers = [] # identify and remove outliers
		for coords in region[1]:
			coord = tuple(coords)
			if coord in lookup_dict:
				data.itemset((coord[0]-x_ll, coord[1]-y_ll), lookup_dict[coord][1]) # fill empty array with z-data
				to_find_outliers.append(lookup_dict[coord][1]) # list with all z-values of region to identify and remove outliers


		"""remove outliers and 0 values"""
		outliers = find_outliers(np.asarray(to_find_outliers)) # find outliers
		if outliers == False: # remove when no slope, so flat parts, terminology is a bit misleading
			print "this is not stairs, it's flat as the Netherlands"
			continue

		medi = np.median(data) # make outliers median value, das Problem ist, dass einfach alle ersetzt werden egal wo sie sind. Das ist doch bisschen Pfusch...
		for i in outliers:
			data[data == i] = medi

		data = rows_w_0(data, 80.0) # remove rows and columns with mostly (80%) 0 values

		if 0 in data.shape: 
			print "this is not stairs, it's too unevenly spread"
			continue

		mi = np.amin(data[np.nonzero(data)]) # make the smallest pixel minimum instead of 0, to visualize slope better
		data[data == 0] = mi

		"""show raster 2d"""
		# plt.imshow(data, cmap='spectral')
		# plt.colorbar()
		# plt.tight_layout()
		# plt.show()

		"""show raster 3d"""
		# from mpl_toolkits.mplot3d import Axes3D
		# from matplotlib import cm
		# x = np.arange(0,data.shape[1],1)		# x = np.arange(0,w,1)
		# y = np.arange(0,data.shape[0],1)		# y = np.arange(0,h,1)
		# X, Y = np.meshgrid(x, y)

		# fig = plt.figure()
		# ax = fig.gca(projection='3d')
		# surf = ax.plot_surface(X,Y,data, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
		# fig.colorbar(surf, shrink=0.5, aspect=5)
		# plt.show()

		""" calculate the slope """
		# also here remove outliers? / trim data or list > http://stats.stackexchange.com/questions/194783/extreme-values-in-the-data

		l = []
		for i in data: # prepare data for slope in Y-direction
			l.append(np.mean(i))

		# slopeY, brY = np.polyfit(np.arange(len(l)),np.asarray(l),1) # linregress is the same
		# stats.mstats.theilslopes(l) # results weren't better, should be more robust

		slopeY, interceptY, r_valueY, p_valueY, std_errY = stats.linregress(np.arange(len(l)),np.asarray(l))
		# print 'slope y', slopeY
		# print 'r-value y', r_valueY # Pearson's coefficient, that tell's how linear it is
		# print 'p-value', p_valueY
		# print 'standard deviation', std_errY

		"""show linear regression"""
		# plt.title('Linear Regression Y-direction, slope is ca. '+str(int(math.degrees(slopeY))))
		# plt.plot(np.arange(len(l)),np.asarray(l),'g.--')
		# plt.plot(np.arange(len(l)),np.polyval([slopeY, interceptY],np.arange(len(l))),'r.-')
		# plt.legend(['data', 'regression'])
		# plt.show()

		lT = []
		for i in data.T: # prepare data for slope in X-direction
			lT.append(np.mean(i))

		slopeX, interceptX, r_valueX, p_valueX, std_errX = stats.linregress(np.arange(len(lT)),np.asarray(lT))
		# print 'slope X', slopeX
		# print 'r-value X', r_valueX # Pearson's coefficient, that tell's how linear it is
		# print 'p-value', p_valueX
		# print 'standard deviation', std_errX		

		"""show linear regression"""
		# plt.title('Linear Regression X-direction, slope is ca. '+str(int(math.degrees(slopeX))))
		# plt.plot(np.arange(len(lT)),np.asarray(lT),'g.--')
		# plt.plot(np.arange(len(lT)),np.polyval([slopeX, interceptX],np.arange(len(lT))),'r.-')
		# plt.legend(['data', 'regression'])
		# plt.show()

		slopeX = math.degrees(slopeX)
		slopeY = math.degrees(slopeY)
		if abs(r_valueY) > 0.6 and abs(slopeY) < 50 and abs(slopeY) >= 20 and std_errY <= 0.1: # should be 44
			if abs(slopeX) < 14: # random threshold to not get completely sloped areas, should be 0 in theory
				print "hey we got stairs in Y-direction"
				for i in region[1]:
					toDB.append(tuple(i))
		if abs(r_valueX) > 0.6 and abs(slopeX) < 50 and abs(slopeX) >= 20 and std_errX <= 0.1: # should be 44
			if abs(slopeY) < 14: # random threshold to not get completely sloped areas, should be 0 in theory
				print "hey we got stairs in X-direction"
				for i in region[1]:
					toDB.append(tuple(i))

	return toDB

def writetoDB(dbms_name, user, password, toDB, storey):	
	# writes results to the databaset
	print "writing the stairs to the database"

	# connect to database
	con = psycopg2.connect("host='localhost' dbname='"+dbms_name+"' user='"+user+"' password='"+password+"'")
	cur = con.cursor()

	# write stair attribute (2) to semantics database
	for i in toDB:
		x = str(i[0])
		y = str(i[1])
		cur.execute('UPDATE semantics SET attribute = 2 WHERE x = '+x+' AND y = '+y+' AND storey = '+str(storey)+' AND attribute is NULL;') # Zusammenhaengende anders behandeln?
	con.commit()
	cur.close()

def stairs(storey, dbms_name="thesis", user="postgres", password="", minleafDomain=0.05277552607):
	"""
	This function calls all other functions in this file and structures the workflow
	"""
	print 'finding the stairs in storey '+str(storey)+', this might also take some time ...'
	# minleafDomain = 0.05277552607 # takes less long for testing, this has to come from PointlessConverter or add_semantics, zeb1
	# minleafDomain = 0.0545399934053 # other zeb1 of firebrigade2/3
	# minleafDomain = 0.0520286270439 # leica
	storey_dict, xmax, ymax, lookup_dict = retrieveDB(dbms_name, user, password, storey)
	vert = []
	hori = []
	# non_planar = []

	i = 0
	for each in storey_dict:
		neighbours_lst = [each] # puts also itself in there
		for neighbour in giveMeAllEqualNeighbours(each): # returns set with all neighbours of current node, from findNeighbours
			if neighbour in storey_dict:
				neighbours_lst.append(neighbour)
		N = svd_magic(neighbours_lst, storey_dict)
		if N is not None: # if it's a non planar voxel with not enough points
			angle = angle_to_z(N)
			if is_vertical(angle) is True:
				vert.append(each)
			elif is_horizontal(angle) is True:
				hori.append(each)
		# 	else:
		# 		non_planar.append(each)
		# else:
		# 	non_planar.append(each)

	Hv = make2dhistogram(storey_dict, vert, xmax, ymax)
	Hh = make2dhistogram(storey_dict, hori, xmax, ymax)
	# make2dhistogram(storey_dict, non_planar, xmax, ymax) # --> not necessary, basically leftovers

	fr1, fr2 = filtervertical(Hv, minleafDomain)
	frbc = filterhorizontal(Hh)

	combi_fr = combine_fr(frbc, fr1, fr2)

	l_regions = grow_region(combi_fr, minleafDomain)
	# print l_regions, minleafDomain, lookup_dict
	if l_regions:
		toDB = calculate_slope(l_regions, lookup_dict)
		if toDB:
			writetoDB(dbms_name, user, password, toDB, storey)
	else:
		print "there's no stairs on this storey"

if __name__ == '__main__':
	stairs(0, "thesis_main", "postgres", "")