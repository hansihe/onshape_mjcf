# onshape-mjcf
Aims to provide everything you need to build a Onshape -> MJCF pipeline.

This package contains 2 base areas of functionality, which function fairly independently:
* Data structures, utilities and algorithms for reading and working with data from the Onhhape API.
* Builder for MJCF files.

On top of this, the library also provides:
* Utilities for analyzing the topology of Onshape documents and transforming them into kinematic trees.
* Basic Onshape -> MJCF conversion pipeline.

A more complete list of features:
* Abstract representation of both the raw CAD model and the inferred robot kinematic tree.
* Infers the topology of your robot kinematic tree using graph analysis.
* Has the ability to recognize primitive shapes in CAD model (sphere, cyllinder for now) and specify those as primitives in the MJCF. This allows you to efficiently specify colliders in your CAD model.
* Powerful transparrent caching layer for interacting with your Onshape document.

Thanks to [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot) for providing the base of a lot of the code in this repository.

## Basic usage
To get started quickly, you can use the `onshape-to-mjcf-basic` command provided by this library.

```bash
onshape-to-mjcf-basic [document-id] [assembly-name] [root-name]
```

Your MJCF will be built as `robot.xml`. Meshes are exported into the `models` directory.

## Advanced usage
Use the [source code for the `onshape-to-mjcf-basic` command](onshape_mjcf/to_mjcf_basic.py) as a starting point for your own conversion script.

It is fairly simple for what it does, only about 100 LOC.

## How does this compare to `onshape-to-robot`?
Main areas of difference are:
* Output format
  * `onshape-mjcf` targets the MJCF format (but would be easily extensible to others)
  * `onshape-to-robot` targets SDF/URDF
* Invocation
  * `onshape-mjcf` provides a library you would use in your own conversion script. It also provides a basic CLI tool.
  * `onshape-to-robot` provides a CLI tool
* Customization
  * `onshape-mjcf` allows you to customize every aspect of the conversion in code. The goal is to make it possible to do conversion without any post processing
  * `onshape-to-robot` provides limited customizability though a config file. Most of the time you need to post process the files manually
* Kinematic tree
  * `onshape-mjcf` builds your kinematic tree by graph analysis of your CAD document. You just provide a root, joint mate ordering doesn't matter.
  * `onshape-to-robot` uses ordering of mates within a joint in Onshape to specify topology implicitly.
* Topology
  * `onshape-mjcf` enables you to structure your CAD model as you wish, it uses graph analysis to recognize topology independently of assemblies.
  * `onshape-to-robot` relies on you grouping parts within assemblies in your CAD model for topology.
