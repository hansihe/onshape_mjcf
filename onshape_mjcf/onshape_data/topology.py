from collections import defaultdict
from copy import copy
import itertools
import networkx as nx
from onshape_mjcf.onshape_data import Assembly, AssemblyMateFeature, AssemblyMateGroupFeature, FeatureId, InstancePath, OnshapeData

def findComponents(data: OnshapeData, excluded: set[FeatureId]) -> list[set[InstancePath]]:
    """
    Identifies components of the instance graph when the given set of mates are excluded.
    A component is defined as a set of instances that have connectivity through mates.
    """

    graph = nx.Graph()

    def addEdges(features, canonicalizePath):
        for feature in features.values():
            if feature.id in excluded:
                continue
            if isinstance(feature, AssemblyMateFeature):
                if feature.mateType == "FASTENED":
                    graph.add_edge(
                        canonicalizePath(feature.matedEntities[0].occurrence),
                        canonicalizePath(feature.matedEntities[1].occurrence)
                    )
            if isinstance(feature, AssemblyMateGroupFeature):
                first = canonicalizePath(feature.occurrences[0])
                for second in feature.occurrences[1:]:
                    graph.add_edge(
                        first,
                        canonicalizePath(second)
                    )

    # Mates for root
    addEdges(data[data.rootAssemblyId].archetype.features, canonicalizePath=lambda p: p)

    # Mates for occurrences
    for occurrence in data.occurrences.values():
        lookup = data[occurrence.path]

        if not isinstance(lookup.archetype, Assembly):
            continue

        addEdges(lookup.archetype.features, canonicalizePath=lambda p: lookup.canonicalizePath(p))

    return sorted(nx.connected_components(graph))

def findCommonAncestor(occurrences: set[InstancePath]):
    occurrences = list(occurrences)

    first = occurrences[0]
    for occ in occurrences:
        assert first.root == occ.root

    elems = first.elements
    count = len(elems)

    for occ in occurrences[1:]:
        for idx, (e1, e2) in enumerate(zip(itertools.islice(elems, count), occ.elements)):
            if e1 != e2:
                count = min(idx, count)
                break

    return InstancePath(
        root=first.root,
        elements=elems[:count]
    )

def makeOccurranceTree(paths: list[InstancePath]):
    graph = nx.DiGraph()
    for path in paths:
        elems = []
        for n2 in path.elements:
            before = tuple(elems)
            elems.append(n2)
            graph.add_edge(before, tuple(elems))
    return nx.depth_first_search.dfs_tree(graph, ())

def findStrictSubtrees(super, sub, rootNode=()):
    # find common subtrees
    common_subtrees = set()
    t1_succ = super.succ
    t2_succ = sub.succ
    for path in nx.dfs_postorder_nodes(sub, source=rootNode):
        if t1_succ[path] != t2_succ[path]:
            continue
        for succ in t1_succ[path]:
            if succ not in common_subtrees:
                continue
        common_subtrees.add(path)
    
    # find common subtree roots
    common_roots = set()
    excluded = set()

    view = nx.subgraph_view(sub, filter_node=lambda n1: n1 not in excluded)
    for path in nx.dfs_preorder_nodes(view, source=rootNode):
        if path in common_subtrees:
            excluded.update(sub.succ[path])
            common_roots.add(path)

    return common_roots