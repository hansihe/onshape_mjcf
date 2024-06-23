from collections import defaultdict
import hashlib
import numpy as np
import networkx as nx
from dataclasses import dataclass

from onshape_mjcf.onshape_data import AssemblyMateConnectorFeature, AssemblyMateFeature, FeatureId, InstancePath, OnshapeData, Part, PartRef
from onshape_mjcf.onshape_data.joint import getLimits
from onshape_mjcf.onshape_data.topology import findCommonAncestor, findComponents, findStrictSubtrees, makeOccurranceTree
from onshape_mjcf.util import formatName

@dataclass
class EqualityConstraint:
    point: np.array

    left: InstancePath
    right: InstancePath

    leftComp: int
    rightComp: int

@dataclass
class RobotDof:
    name: str
    featureId: FeatureId
    limits: tuple[float, float] | None

    jointType: str
    jointTransform: np.matrix

    child: InstancePath
    parent: InstancePath

    childComp: int
    parentComp: int

@dataclass
class Site:
    name: str
    transform: np.array
    parent: InstancePath
    parentComp: int

def buildDofs(data: OnshapeData) -> tuple[list[RobotDof], list[EqualityConstraint], list[Site]]:
    rootId = data.rootAssemblyId
    rootAssembly = data.assemblies[rootId]

    dofs = {}
    equalConstraints = []
    sites = []

    for feature in rootAssembly.features.values():
        if feature.suppressed:
            continue

        if isinstance(feature, AssemblyMateConnectorFeature):
            if feature.name.startswith("site_"):
                T_world_part = data.occurrences[feature.definition.occurrence].transform
                T_part_mate = feature.definition.partToMateT
                T_world_mate = T_world_part @ T_part_mate

                sites.append(Site(
                    name=feature.name[5:],
                    parent=feature.definition.occurrence,
                    transform=T_world_mate,
                    parentComp=None
                ))

            if feature.name.startswith("link_"):
                # TODO link impl
                pass

        elif isinstance(feature, AssemblyMateFeature):
            entities = feature.matedEntities

            assert len(entities) == 2
            e1 = entities[0]
            e2 = entities[1]

            T_world_part = data.occurrences[e1.occurrence].transform
            T_part_mate = e1.partToMateT
            T_world_mate = T_world_part @ T_part_mate

            if feature.name.startswith("closing_"):
                equalConstraints.append(EqualityConstraint(
                    point=T_world_mate[:3, 3],
                    left=e1.occurrence,
                    right=e2.occurrence,
                    leftComp=None,
                    rightComp=None
                ))
            elif feature.name.startswith("dof_"):
                parts = feature.name.split("_")[1:]
                inverted = False
                if parts[-1] == "inv" or parts[-1] == "inverted":
                    inverted = True
                    parts = parts[:-1]
                name = "_".join(parts)

                if name == "":
                    raise Exception("unnamed DOF")

                assert name not in dofs

                featId = feature.id
                mateType = feature.mateType

                jointType = None
                limits = None

                if mateType in ["REVOLUTE", "CYLINDRICAL"]:
                    if "wheel" in parts or "continuous" in parts:
                        jointType = "continuous"
                    else:
                        jointType = "revolute"
                    limits = getLimits(data, jointType, featId)
                elif mateType == "SLIDER":
                    jointType = "prismatic"
                    limits = getLimits(data, jointType, featId)
                elif mateType == "FASTENED":
                    jointType = "fixed"
                else:
                    raise "unknown mate type: " + mateType

                #matedEntity = entities[0]
                #T_world_part = data.occurrences[matedEntity.occurrence].transform
                #T_part_mate = matedEntity.partToMateT
                
                ### TODO invert flag
                ##if False:
                ##    if limits is not None:
                ##        limits = (-limits[1], -limits[0])
                ##    
                ##    # Flipping the joint around X axis
                ##    flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])                        
                ##    T_part_mate[:3, :3] = T_part_mate[:3, :3] @ flip

                #T_world_mate = T_world_part @ T_part_mate

                dofs[name] = RobotDof(
                    name=name,
                    jointType=jointType,
                    featureId=feature.id,
                    limits=limits,
                    child=e1.occurrence,
                    parent=e2.occurrence,
                    childComp=None,
                    parentComp=None,
                    jointTransform=T_world_mate
                )

        else:
            raise Exception("unknown feature type: " + str(feature))

    return dofs, equalConstraints, sites

@dataclass()
class RobotComponent:
    idx: int
    name: str
    originalName: str

    instances: set[InstancePath]
    """
    The full set of instances within this component.
    """

    parts: list[InstancePath]
    """
    The full set of part instances within this component.
    """
    
    rootInstances: list[InstancePath]
    """
    The minimal set of instances which either:
    * Are parts which are a member of the component.
    * Are assemblies which only dominate parts which are a member
      of the component.

    You can rely on this to directly or indirectly contain every
    part which is a member of this component, and no others.

    For instance, if you take the mass of these instances
    and compose them together, you will end up with the
    same mass number as if you were to call the mass calculation
    on every part independently.
    """

    partTree: dict[InstancePath, list[InstancePath]]

    sites: list[Site]
    """
    Map from site idx to list of sites.
    """

@dataclass(frozen=True)
class ComponentEdge:
    u: int
    v: int
    dofName: str

@dataclass(frozen=True)
class Mesh:
    ref: PartRef
    uniqueName: str
    originalName: str
    partHash: str

@dataclass(frozen=True)
class RobotDescription:
    dofs: dict[str, RobotDof]
    equalities: list[EqualityConstraint]

    rootComponentIdx: int
    componentEdges: dict[int, list[ComponentEdge]]
    components: list[RobotComponent]

    meshes: dict[PartRef, Mesh]

    instanceTree: dict[InstancePath, list[InstancePath]]

    @classmethod
    def from_onshape(cls, data: OnshapeData, rootName: str):
        dofs, equalities, sites = buildDofs(data)
        components = findComponents(data, dofs)
        rootInstanceTree = makeOccurranceTree(data.occurrences.keys())

        instanceTree = {}
        for node in rootInstanceTree.nodes:
            nodeP = InstancePath(data.rootAssemblyId, node)
            instanceTree[nodeP] = list(map(lambda p: InstancePath(data.rootAssemblyId, p), rootInstanceTree.succ[node].keys()))

        # Find component names
        robotComponents = []
        allPartRefs = {}
        for idx, comp in enumerate(components):
            ancestor = findCommonAncestor(comp)
            ancestorLookup = data[ancestor]
            compName = ancestorLookup.instance.name

            parts = []
            for instancePath in comp:
                lookup = data[instancePath]
                if isinstance(lookup.archetype, Part):
                    assert instancePath.root == data.rootAssemblyId
                    parts.append(instancePath)
                    allPartRefs[lookup.archetype.ref] = instancePath
            parts = sorted(parts)

            partTree = makeOccurranceTree(parts)
            commonRoots = findStrictSubtrees(rootInstanceTree, partTree)
            commonRoots = list(map(lambda p: InstancePath(data.rootAssemblyId, p), commonRoots))

            robotComponents.append(RobotComponent(
                idx=idx,
                name=formatName(compName),
                originalName=compName,
                instances=comp,
                parts=parts,
                rootInstances=commonRoots,
                partTree=partTree,
                sites=None
            ))

        # Map DOFs to components
        pathToComponentIdx = {}
        for idx, comp in enumerate(components):
            for item in comp:
                pathToComponentIdx[item] = idx

        for dof in dofs.values():
            dof.childComp = pathToComponentIdx[dof.child]
            dof.parentComp = pathToComponentIdx[dof.parent]

            if dof.childComp == dof.parentComp:
                raise Exception("DOF needs to be between two separate components, was to itself")

        for equality in equalities:
            equality.leftComp = pathToComponentIdx[equality.left]
            equality.rightComp = pathToComponentIdx[equality.right]

            if dof.childComp == dof.parentComp:
                raise Exception("Equality needs to be between two separate components, was to itself")

        # Assign sites to components
        sitesByIdx = defaultdict(list)
        for site in sites:
            site.parentComp = pathToComponentIdx[site.parent]
            sitesByIdx[site.parentComp].append(site)
        for idx, comp in enumerate(robotComponents):
            comp.sites = sitesByIdx[idx]

        # Find root
        rootComponentIdx = None
        for comp in robotComponents:
            if comp.originalName == rootName:
                if rootComponentIdx is not None:
                    raise Exception("Root name matches multiple components!")
                rootComponentIdx = comp.idx
        if rootComponentIdx is None:
            raise Exception("No root component found by name!")

        # Build kinematic tree
        graph = nx.Graph()
        graph.add_nodes_from(pathToComponentIdx.values())
        graph.add_weighted_edges_from(map(lambda dof: (dof.parentComp, dof.childComp, dof.name), dofs.values()))

        # Can have no cycles
        assert len(nx.cycle_basis(graph)) == 0

        # Can have only 1 component
        if nx.number_connected_components(graph) != 1:
            for comp in robotComponents:
                print(comp.idx, comp.name)
            for dof in dofs.values():
                print(dof.name, dof.parentComp, dof.childComp)
            print(list(nx.connected_components(graph)))
            raise Exception("CAD graph can only have 1 connected component")

        # DFS to generate kinematic tree from root
        tree = nx.bfs_tree(graph, rootComponentIdx)
        componentEdges = dict(map(lambda a: (a, list()), graph.nodes))
        for u, v in tree.out_edges():
            componentEdges[u].append(ComponentEdge(u, v, graph.get_edge_data(u, v)["weight"]))

        # Calculate mesh naming
        allPartRefs = sorted(allPartRefs.items())
        partNameCounters = {}
        partMap = {}
        for ref, exampleInstance in allPartRefs:
            lookup = data[exampleInstance]

            key = f"{lookup.archetype.ref}"
            digest = hashlib.sha1(key.encode()).hexdigest()

            uniqueName = formatName(lookup.instance.name)
            if uniqueName in partNameCounters:
                counter = partNameCounters[uniqueName]
                uniqueName = f"{uniqueName}_{counter}"
                partNameCounters[uniqueName] += 1
            else:
                partNameCounters[uniqueName] = 1

            partMap[ref] = Mesh(
                ref=ref,
                uniqueName=uniqueName,
                originalName=lookup.instance.name,
                partHash=digest
            )

        return cls(
            dofs=dofs,
            equalities=equalities,
            rootComponentIdx=rootComponentIdx,
            componentEdges=componentEdges,
            components=robotComponents,
            meshes=partMap,
            instanceTree=instanceTree,
        )