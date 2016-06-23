"""
!!!This script has to be manually changed and fed with data found through CloudCompare!!!
It calculates the rotation angle the point cloud needs to be parallel to the x- & y-axes.

Steps to find Plane and Normal parameters (to put in line 25):
RANSAC --> manually select main plane --> export to XYZ --> open in CloudCompare, find parameters in properties

Steps for transformation in CloudCompare:
Apply transformation --> Axis (0/0/1), Rotation angle: RESULT
"""

from sympy import Point3D, Line3D, Plane
import math

def intersect(a, b):
	return a.intersection(b)[0]

def angle(l1, l2):
	return l1.angle_between(l2)

def main():
	z = Plane(Point3D(0, 0, 0), normal_vector=(0, 0, 1))

	# The coordinates have to be added manually at this point, retrieved from CloudCompare
	wall = Plane(Point3D(3.18607, -4.60194, 2.25742), normal_vector=(-0.998939,0.019319,0.041815))
	# normal: (-0.848817,-0.525202,0.060595) (-0.921847,-0.387546,0.002566)
	# X: 3.05705 , Y: -1.88154 , Z: 3. (-5.83743, -3.17211, -1.77898)

	x = angle(Line3D(Point3D(0, 0, 0), Point3D(1, 0, 0)), intersect(z, wall))
	print 'The point cloud has to be rotated with ' + str(math.degrees(x)) + ' degrees around the z-axis.'

if __name__ == '__main__':
	main()