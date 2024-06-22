import os
from tqdm import tqdm
import numpy as np

from onshape_mjcf import RobotDescription
from onshape_mjcf.mjcf.builder import BodyBuilder, MJCFBuilder
from onshape_mjcf.onshape_data import OnshapeData
from onshape_mjcf.onshape_data.inertial import calculateMassPropertiesByParts
from onshape_mjcf.onshape_data.primitive_geom import CylinderGeom, SphereGeom, partToPrimitive
from onshape_mjcf.util import formatName

def buildMJCFBody(data: OnshapeData, description: RobotDescription, parent: BodyBuilder, compIdx: int, dofName=None, freeJoint=False):
    comp = description.components[compIdx]
    name = formatName(comp.name)
    body = parent.body(name)

    if freeJoint:
        body.freeJoint()

    if dofName is not None:
        dof = description.dofs[dofName]
        joint = body.joint()
        joint.name = dofName
        joint.frictionloss = 0.01

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

    for path in comp.parts:
        lookup = data[path]
        mesh = description.meshes[lookup.archetype.ref]

        geom = body.geom()

        if mesh.originalName.startswith("Collider "):
            primitive = partToPrimitive(data, lookup.archetype.ref)

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

    inertial = calculateMassPropertiesByParts(data.client, data, description, comp.rootInstances) #, overrides=inertialOverrides)

    body.inertial.pos = inertial.centroid
    body.inertial.mass = inertial.mass
    body.inertial.inertia = inertial.inertia

    for edge in description.componentEdges[compIdx]:
        buildMJCFBody(data, description, body, edge.v, dofName=edge.dofName)

def toMJCFBasic(data: OnshapeData, description: RobotDescription, writeMeshes=True):
    mjcf = MJCFBuilder()

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
    buildMJCFBody(data, description, mjcf.worldBody, description.rootComponentIdx, freeJoint=True)

    return mjcf