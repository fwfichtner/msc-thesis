"""main file to control and run the others"""

# imports main functions of other files
from add_semantics import main
from validate_walls import vali
from find_stairs import stairs


def find_semantics():
	# reconstruction and semantic labelling

	file='firebrigade3_zeb1.las'
	maxwall=0.4
	do_validate_walls = True # be careful to also change the parameters in the hough_transform function
	dbms_name="thesis"
	user="postgres"
	password=""
	# settings for histograms, settings for show?
	
	minleafDomain, storeys = main(dbms_name, user, password, file, maxwall)

	if do_validate_walls:
		print "you are validating the walls, this improves the results a lot, but also takes quite some time ..."
		for storey in xrange(storeys):
			vali(storey, dbms_name, user, password, minleafDomain, maxwall)

	for storey in xrange(storeys):
		stairs(storey, dbms_name, user, password, minleafDomain)

def get_network():
	# not part of the online version
	pass

if __name__ == '__main__':
	find_semantics()