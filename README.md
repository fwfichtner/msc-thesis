# Semantic enrichment of a point cloud based on an octree for multi-storey pathfinding
Geomatics Master Thesis at Delft University of Technology and CGI Nederland by Florian W. Fichtner

## Overview

Acquiring point clouds of indoor environments became increasingly accessible in recent years. However, the resulting 3D point cloud data is unstructured, and does not contain enough information to be useful for complex tasks like pathfinding. Indoor models which are currently derived from point clouds do not include furniture and stairs. The necessary graph to enable multi-storey pathfinding is not available in the point cloud.

This thesis proposes a workflow to semantically enrich indoor point clouds using an octree data structure. Meaning is added to the point cloud scene that allows to act as a basis for a graph. This graph can then follow navigation constraints of humans through an indoor environment. The approach for semantic enrichment of this study is capable of separating storeys, detecting floors, walls, stairs and obstacles like furniture. Strict preconditions are used, like walls being perpendicular to each other and using noise free point clouds. The implementation works as a proof of concept and the octree proves to be a helpful data structure.

For more information about this project, see the [SIMs3D Website](http://www.sims3d.net/).

## Getting Started

The necessary software, packages and deployment are described in the thesis document. The code provided here is the one used for the semantic enrichment of the point cloud.

A link to the thesis will be made available soon.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
