# onshape-mjcf

This package contains 2 base areas of functionality, which function fairly independently:
* Data structures, utilities and algorithms for reading and working with data from the Onhhape API.
* Builder for MJCF files.

On top of this, the library also provides:
* Utilities for analyzing the topology of Onshape documents and transforming them into kinematic trees.

A more complete list of features:
* Abstract representation of both the raw CAD model and the inferred robot kinematic tree.
* Infers the topology of your robot kinematic tree using graph analysis.
* Has the ability to recognize primitive shapes in CAD model (sphere, cyllinder for now) and specify those as primitives in the MJCF. This allows you to efficiently specify colliders in your CAD model.
* Powerful transparrent caching layer for interacting with your Onshape document.

Thanks to [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot) for providing the base of a lot of the code in this repository.

## How does this compare to `onshape-to-robot`?

Main areas of difference are:
* Output format
  * `onshape-mjcf` targets the MJCF format (but would be easily extensible to others)
  * `onshape-to-robot` targets SDF/URDF
* Invocation
  * `onshape-mjcf` provides a library you would use in your own conversion script
  * `onshape-to-robot` provides a CLI tool
* Customization
  * `onshape-mjcf` allows you to customize every aspect of the conversion in code. The goal is to make it possible to do conversion without any post processing
  * `onshape-to-robot` provides limited customizability though a config file. Most of the time you need to post process the files manually
