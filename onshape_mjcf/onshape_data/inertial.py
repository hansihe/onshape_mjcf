import numpy as np

from onshape_mjcf import RobotDescription
from onshape_mjcf.onshape_data import Assembly, InstancePath, OnshapeData, Part, QualifiedRef
from onshape_mjcf.util import warn, error
from onshape_mjcf.util.inertial import InertialData, TransformedInertialData, combineInertials

def onshapeMassPropsToInertial(props: dict, frame: np.array=None) -> InertialData:
    inertia, _inertiaMin, _inertiaMax = np.array(props["inertia"]).reshape((3, 3, 3))
    mass, _massMin, _massMax = props["mass"]
    centroid, _centroidMin, _centroidMax = np.array(props["centroid"]).reshape((3, 3))

    kwargs = {
        "inertia": inertia,
        "mass": mass,
        "centroid": centroid
    }

    if frame is None:
        return InertialData(**kwargs)
    else:
        return TransformedInertialData(transform=frame, **kwargs)

def calculateMassProperties(client, data: OnshapeData, description: RobotDescription, rootInstances: list[InstancePath]):
    inertials = []

    for root in rootInstances:
        lookup = data[root]

        if isinstance(lookup.archetype, Assembly):
            props = client.get_assembly_mass_properties(lookup.archetype.ref)
        elif isinstance(lookup.archetype, Part):
            props = client.get_part_mass_properties(lookup.archetype.ref, lookup.archetype.partId, useMassPropertyOverrides=True)
        else:
            assert(False)

        massMissingCount = props["massMissingCount"]
        if massMissingCount > 0:
            warn(f"{lookup.instance.name} is missing mass for {massMissingCount} parts")

        if not props["hasMass"]:
            error(f"{lookup.instance.name} is missing mass")

        inertials.append(onshapeMassPropsToInertial(props, frame=lookup.occurrence.transform))

    return combineInertials(inertials)

def calculateMassPropertiesByParts(client, data: OnshapeData, description: RobotDescription, rootInstances: list[InstancePath], overrides: dict[QualifiedRef, InertialData]={}):
    """
    This is a workaround for the broken onshape assembly mass calculation.

    Instead of calculating mass on roots, we walk down to each individual part
    and combine them.
    """

    inertials = []

    def traverseInstance(instance):
        lookup = data[instance]

        if lookup.archetype.ref in overrides:
            override = overrides[lookup.archetype.ref]

            if override == "getFromAssembly":
                assert isinstance(lookup.archetype, Assembly)
                props = client.get_assembly_mass_properties(lookup.archetype.ref, linkDocumentId=data.rootAssemblyId.documentId)
                inertials.append(onshapeMassPropsToInertial(props, frame=lookup.occurrence.transform))

            elif override is not None:
                override = override.normalize()
                inertials.append(TransformedInertialData(
                    mass=override.mass,
                    centroid=override.centroid,
                    inertia=override.inertia,
                    transform=lookup.occurrence.transform
                ))

            return

        if isinstance(lookup.archetype, Assembly):
            for subInstance in description.instanceTree[instance]:
                traverseInstance(subInstance)

        elif isinstance(lookup.archetype, Part):
            props = data.getPartMassProperties(lookup.archetype.ref)

            if not props["hasMass"]:
                warn(f"instance has no mass: {data.pathPrettyName(instance)}")
                return

            inertials.append(onshapeMassPropsToInertial(props, frame=lookup.occurrence.transform))

        else:
            assert(False)

    for instance in rootInstances:
        traverseInstance(instance)

    return combineInertials(inertials)