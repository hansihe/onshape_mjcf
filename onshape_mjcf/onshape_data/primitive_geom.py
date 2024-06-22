from dataclasses import dataclass
import pprint
import numpy as np

from onshape_mjcf.onshape_data import OnshapeData, PartRef

@dataclass(frozen=True)
class PrimitiveGeom:
    pass

@dataclass(frozen=True)
class SphereGeom(PrimitiveGeom):
    origin: np.array
    radius: float

@dataclass(frozen=True)
class CylinderGeom(PrimitiveGeom):
    fromto: np.array # (2, 3)
    radius: float

def linePlaneIntersection(planePoint, planeNormal, rayPoint, rayDirection, epsilon=1e-6):
    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")

    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    return Psi

def partToPrimitive(data: OnshapeData, partRef: PartRef):
    lookup = data[partRef]

    details = data.client.get_part_body_details(lookup.archetype.ref)
    bodies = details["bodies"]

    if len(bodies) == 1:
        body = bodies[0]
        faces = body["faces"]
        edges = body["edges"]

        # Sphere
        if body["type"] == "solid" and len(faces) == 1 and len(edges) == 0:
            face = faces[0]
            surface = face["surface"]
            if surface["type"] == "sphere":
                return SphereGeom(
                    origin=np.array(surface["origin"]),
                    radius=surface["radius"]
                )

        # Cylinder
        if body["type"] == "solid" and len(faces) == 3 and len(edges) == 2:
            planes = []
            cylinders = []
            for face in faces:
                typ = face["surface"]["type"]
                if typ == "plane":
                    planes.append(face)
                if typ == "cylinder":
                    cylinders.append(face)

            if len(planes) == 2 and len(cylinders) == 1:
                cylinderOrigin = np.array(cylinders[0]["surface"]["origin"])
                cylinderAxis = np.array(cylinders[0]["surface"]["axis"])
                cylinderRadius = np.array(cylinders[0]["surface"]["radius"])
                plane1Point = np.array(planes[0]["surface"]["origin"])
                plane1Normal = np.array(planes[0]["surface"]["normal"])
                plane2Point = np.array(planes[1]["surface"]["origin"])
                plane2Normal = np.array(planes[1]["surface"]["normal"])

                fromto = np.array([
                    linePlaneIntersection(plane1Point, plane1Normal, cylinderOrigin, cylinderAxis),
                    linePlaneIntersection(plane2Point, plane2Normal, cylinderOrigin, cylinderAxis)
                ])

                return CylinderGeom(
                    fromto=fromto,
                    radius=cylinderRadius
                )

    pprint.pprint(details)
    raise Exception("Could not extract primitive geom from part. Are you sure it's a primitive-like?")
    