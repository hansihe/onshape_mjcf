from dataclasses import dataclass, field
import os
from tqdm import tqdm
import numpy as np

from onshape_mjcf import RobotDescription
from onshape_mjcf.mjcf.builder import BodyBuilder, MJCFBuilder
from onshape_mjcf.onshape_data import InstancePath, OnshapeData
from onshape_mjcf.onshape_data.inertial import calculateMassPropertiesByParts
from onshape_mjcf.onshape_data.primitive_geom import CylinderGeom, SphereGeom, partToPrimitive
from onshape_mjcf.util import formatName

@dataclass(kw_only=True)
class MJCFBuildOptions:
    geomNames: dict[tuple[str], str] = field(default_factory=dict)
    """
    Map from a path of names to a name in the resulting MJCF.

    Key is a path of the names you see in the left of your Onshape CAD document.
    Example: ("Body <1>", "Some Component <1>")
    """

def transformPos(transform: np.array, pos: np.array):
    return (transform @ np.pad(pos, (0, 1), constant_values=1))[:3]

@dataclass
class BuildData:
    data: OnshapeData
    description: RobotDescription
    pathToGeomName: dict[InstancePath, str]

def buildMJCFBody(bd: BuildData, parent: BodyBuilder, compIdx: int, dofName=None, freeJoint=False):
    comp = bd.description.components[compIdx]
    name = formatName(comp.name)
    body = parent.body(name)

    if freeJoint:
        body.freeJoint()

    if dofName is not None:
        dof = bd.description.dofs[dofName]
        joint = body.joint()
        joint.name = dofName
        #joint.frictionloss = 0.01

        joint.axis = np.asarray(dof.jointTransform[:3, :3]) @ np.array([0, 0, 1])
        joint.pos = np.asarray(dof.jointTransform)[:3, 3]
        match dof.jointType:
            case "revolute":
                # default in mjcf is hinge
                pass
            case _:
                raise Exception("unknown joint type: " + dof.jointType)

        if dof.limits is not None:
            joint.range = dof.limits

    for site in comp.sites:
        siteE = body.site()
        siteE.name = site.name
        siteE.pos = site.transform[:3, 3]

    for path in comp.parts:
        lookup = bd.data[path]
        mesh = bd.description.meshes[lookup.archetype.ref]

        geom = body.geom()

        name = bd.pathToGeomName.get(path)
        if name is not None:
            geom.name = name

        if mesh.originalName.startswith("Collider "):
            primitive = partToPrimitive(bd.data, lookup.archetype.ref)

            if isinstance(primitive, SphereGeom):
                geom.type = "sphere"
                geom.contype = 2
                geom.size = primitive.radius
                geom.pos = (lookup.occurrence.transform @ np.pad(primitive.origin, (0, 1), constant_values=1))[:3]
            elif isinstance(primitive, CylinderGeom):
                geom.type = "cylinder"
                geom.contype = 2
                geom.size = primitive.radius
                geom.fromto = np.array([
                    (lookup.occurrence.transform @ np.pad(primitive.fromto[0], (0, 1), constant_values=1))[:3],
                    (lookup.occurrence.transform @ np.pad(primitive.fromto[1], (0, 1), constant_values=1))[:3]
                ])

        else:
            geom.type = "mesh"
            geom.contype = 0
            geom.conaffinity = 0
            geom.transform = lookup.occurrence.transform
            geom.mesh = mesh.uniqueName

    inertial = calculateMassPropertiesByParts(bd.data.client, bd.data, bd.description, comp.rootInstances) #, overrides=inertialOverrides)

    body.inertial.pos = inertial.centroid
    body.inertial.mass = inertial.mass
    body.inertial.inertia = inertial.inertia

    for edge in bd.description.componentEdges[compIdx]:
        buildMJCFBody(bd, body, edge.v, dofName=edge.dofName)

def toMJCFBasic(data: OnshapeData, description: RobotDescription, options=MJCFBuildOptions(), writeMeshes=True):
    mjcf = MJCFBuilder()

    pathToGeomName = {}
    for namePath, name in options.geomNames.items():
        pathToGeomName[data.occurranceNamePaths[namePath]] = name

    bd = BuildData(
        data=data,
        description=description,
        pathToGeomName=pathToGeomName
    )

    if writeMeshes:
        os.makedirs("models", exist_ok=True)

    # Export meshes and add mesh assets to XML
    for mesh in tqdm(description.meshes.values(), "exporting meshes"):
        stl = data.client.get_part_stl(mesh.ref)

        filePath = f"models/{mesh.uniqueName}.stl"
        if writeMeshes:
            with open(filePath, "wb") as f:
                f.write(stl)

        meshB = mjcf.asset.mesh()
        meshB.name = mesh.uniqueName
        meshB.file = filePath

    # Write out inferred equality constraints
    for equality in description.equalities:
        elem = mjcf.equality.connect()
        elem.anchor = equality.point
        elem.body1 = description.components[equality.leftComp].name
        elem.body2 = description.components[equality.rightComp].name

    # Build robot kinematic tree XML recursively
    buildMJCFBody(bd, mjcf.worldBody, description.rootComponentIdx, freeJoint=True)

    return mjcf