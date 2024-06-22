from typing import Self
from dataclasses import dataclass
import numpy as np

@dataclass
class InertialData:
    centroid: np.array
    mass: float
    inertia: np.array

    def normalize(self) -> Self:
        return self

@dataclass
class TransformedInertialData(InertialData):
    transform: np.array

    def normalize(self) -> InertialData:
        rotation = self.transform[:3, :3]

        centroid = (self.transform @ np.pad(self.centroid, (0, 1), constant_values=1))[:3]
        inertia = rotation @ self.inertia @ rotation.T

        return InertialData(
            centroid=centroid,
            mass=self.mass,
            inertia=inertia
        )


def combineInertials(inertials: list[InertialData]) -> InertialData:
    inertials = list(map(lambda i: i.normalize(), inertials))

    if len(inertials) == 0:
        raise Exception("Empty inertials list provided")

    total_mass = sum(inertial.mass for inertial in inertials)
    combined_centroid = sum(inertial.mass * inertial.centroid for inertial in inertials) / total_mass

    combined_inertia = np.zeros((3, 3))
    for inertial in inertials:
        # Use parallel axis theorem to transform inertial tensor into central 
        d = combined_centroid - inertial.centroid
        d_matrix = np.outer(d, d)
        inertia_parallel = inertial.inertia + inertial.mass * (np.dot(d, d) * np.eye(3) - d_matrix)
        combined_inertia += inertia_parallel

    return InertialData(
        centroid=combined_centroid,
        mass=total_mass,
        inertia=combined_inertia
    )


    mass = 0.0
    centroid = np.array([0.0, 0, 0])

    for inertial in inertials:
        mass += inertial.mass
        centroid += inertial.centroid * inertial.mass
    
    if mass > 0:
        centroid /= mass

    inertia = np.zeros((3, 3))

    for inertial in inertials:
        r = inertial.centroid - centroid
        inertia += inertial.inertia + (np.dot(r, r)*np.eye(3) - r.T*r) * inertial.mass

    return InertialData(
        centroid=centroid,
        mass=mass,
        inertia=inertia
    )
