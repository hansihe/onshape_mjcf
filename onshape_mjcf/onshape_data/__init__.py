from dataclasses import dataclass
from functools import cached_property
import itertools
import json
from typing import Self
import numpy as np
import math

from .. import ureg
from ..onshape_api.client import Client

# Identifiers:
# * InstanceId:
#     ID of a single instance (part/assembly) within another assembly.
#     Has no meaning outside of a parent assembly, can refer to an instance within
#     any instance of the parent.
#     An instance can be mapped to a QualifiedRef.
# * InstancePath:
#     A unique path for a single instance within a document. Also referred to as occurrance.
#     This can uniquely identify a single instance within a document.
#     Implemented as a list of InstanceIds, starting from the root assembly.
# * QualifiedRef: 
#     Unique identifier for a specific version and configuration of a part studio or assembly.

# Shortly explained:
# * InstanceId: Instance within a parent assembly. Can be mapped to a QualifiedRef.
# * InstancePath: Full instance path within a document. Can be mapped to a QualifiedRef.
# * QualifiedRef: Reference to a specific version and configuration of an object.
# * PartRef: Reference to a specific single part.

# Indexing needs:
# * InstancePath -> Instance, Occurrence, Assembly|Part
# * InstanceId -> Instance, Assembly|Part
# * QualifeidRef -> Assembly|PartStudio
# * PartRef -> Part

@dataclass(frozen=True, order=True)
class InstanceId:
    """
    ID of a single instance within an assembly.
    """

    id: str

    def __str__(self) -> str:
        return "instance;" + self.id
    def __repr__(self) -> str:
        return self.__str__()

@dataclass(frozen=True, order=True)
class QualifiedRef:
    """
    The full unique reference to an assembly or part studio.
    """

    documentId: str
    elementId: str
    microversionId: str
    configuration: str

    @classmethod
    def from_json(cls, data: dict):
        return cls(
            documentId=data["documentId"],
            elementId=data["elementId"],
            microversionId=data["documentMicroversion"],
            configuration=data["configuration"]
        )

    def __str__(self) -> str:
        return f"qualified;{self.documentId};{self.elementId};{self.microversionId};{self.configuration}"
    def __repr__(self) -> str:
        return self.__str__()

@dataclass(frozen=True, order=True)
class PartRef:
    """
    The full unique reference to a part.
    """
    ref: QualifiedRef
    partId: str

    @classmethod
    def from_json(cls, data: dict):
        return cls(
            ref=QualifiedRef.from_json(data),
            partId=data["partId"]
        )

    def __str__(self) -> str:
        return f"qualified_part;{self.ref.documentId};{self.ref.elementId};{self.ref.microversionId};{self.ref.configuration};{self.partId}"
    def __repr__(self) -> str:
        return self.__str__()

@dataclass(frozen=True, order=True)
class InstancePath:
    """
    The full path of a single occurrance of an instance.
    Always relative to the root assembly.
    """

    root: QualifiedRef
    elements: tuple[InstanceId]

    def __init__(self, root: QualifiedRef, elements: list[str | InstanceId] | tuple[str | InstanceId]):
        elementsConv = []
        for element in elements:
            if isinstance(element, str):
                elementsConv.append(InstanceId(element))
            elif isinstance(element, InstanceId):
                elementsConv.append(element)
            else:
                raise Exception("unknown type")

        object.__setattr__(self, "root", root)
        object.__setattr__(self, "elements", tuple(elementsConv))

    @property
    def instance(self):
        return self.elements[-1]

    def __len__(self):
        return len(self.elements)
    def __getitem__(self, idx: int) -> Self:
        return InstancePath(
            root=self.root,
            elements=self.elements[:idx + 1]
        )

    def __str__(self) -> str:
        return "occurrence;" + ";".join(map(lambda i: i.id, self.elements))
    def __repr__(self) -> str:
        return self.__str__()

@dataclass(frozen=True)
class FeatureId:
    """
    The ID of a feature.
    A feature can be mates, mate connectors, mate groups, etc.
    """

    id: str

    def __str__(self):
        return f"feature;{self.id}"
    def __repr__(self) -> str:
        return self.__str__()

@dataclass(frozen=True)
class MateDefinition:
    """
    The definition of a mate.
    This contains both a occurrence and a transform matrix.
    TODO fully describe this.
    """

    occurrence: InstancePath
    partToMateT: np.array

@dataclass(frozen=True)
class AssemblyFeature:
    type: str
    id: FeatureId
    suppressed: bool

@dataclass(frozen=True)
class AssemblyMateFeature(AssemblyFeature):
    name: str
    mateType: str
    matedEntities: list[MateDefinition]

    @classmethod
    def from_json(cls, data: dict, ref: QualifiedRef):
        assert data["featureType"] == "mate"
        sData = data["featureData"]

        mated_entities = []
        for entity in sData["matedEntities"]:
            mated_entities.append(MateDefinition(
                occurrence=InstancePath(ref, entity["matedOccurrence"]),
                partToMateT=readCSMatrix(entity["matedCS"])
            ))

        return cls(
            type="mate",
            id=FeatureId(data["id"]),
            suppressed=data["suppressed"],
            mateType=sData["mateType"],
            name=sData["name"],
            matedEntities=mated_entities
        )

@dataclass(frozen=True)
class AssemblyMateGroupFeature(AssemblyFeature):
    name: str
    occurrences: list[InstancePath]

    @classmethod
    def from_json(cls, data: dict, ref: QualifiedRef):
        assert data["featureType"] == "mateGroup"
        sData = data["featureData"]

        occurrences = []
        for occurrence in sData["occurrences"]:
            occurrences.append(InstancePath(ref, occurrence["occurrence"]))

        return cls(
            type="mateGroup",
            id=FeatureId(data["id"]),
            suppressed=data["suppressed"],
            name=sData["name"],
            occurrences=occurrences
        )

@dataclass(frozen=True)
class AssemblyMateConnectorFeature(AssemblyFeature):
    name: str
    definition: MateDefinition

    @classmethod
    def from_json(cls, data: dict, ref: QualifiedRef):
        assert data["featureType"] == "mateConnector"
        sData = data["featureData"]

        return cls(
            type="mateConnector",
            id=FeatureId(data["id"]),
            suppressed=data["suppressed"],
            name=sData["name"],
            definition=MateDefinition(
                occurrence=InstancePath(ref, sData["occurrence"]),
                partToMateT=readCSMatrix(sData["mateConnectorCS"])
            )
        )

@dataclass(frozen=True)
class Instance:
    type: str
    id: InstanceId
    name: str
    suppressed: bool

    @classmethod
    def data_from_json(cls, data: dict):
        return dict(
            id=InstanceId(data["id"]),
            name=data["name"],
            suppressed=data["suppressed"]
        )

@dataclass(frozen=True)
class AssemblyInstance(Instance):
    ref: QualifiedRef
    @classmethod
    def from_json(cls, data: dict):
        return cls(
            type="Assembly",
            ref=QualifiedRef.from_json(data),
            **Instance.data_from_json(data)
        )

@dataclass(frozen=True)
class PartInstance(Instance):
    ref: PartRef
    isStandardContent: bool
    documentVersion: str

    @classmethod
    def from_json(cls, data: dict):
        return cls(
            type="Part",
            ref=PartRef.from_json(data),
            isStandardContent=data["isStandardContent"],
            documentVersion=data.get("documentVersion", None),
            **Instance.data_from_json(data)
        )

@dataclass(frozen=True)
class Assembly:
    type = "assembly"
    ref: QualifiedRef
    instances: dict[InstanceId, Instance]
    instanceByName: dict[str, InstanceId]
    features: dict[FeatureId, AssemblyFeature]

    @classmethod
    def from_json(cls, data: dict):
        ref = QualifiedRef.from_json(data)

        instances = {}
        for instanceData in data["instances"]:
            type = instanceData["type"]
            if type == "Assembly":
                instance = AssemblyInstance.from_json(instanceData)
            elif type == "Part":
                instance = PartInstance.from_json(instanceData)
            else:
                raise Exception("unknown instance type: " + type)
            instances[instance.id] = instance

        features = {}
        for feature in data["features"]:
            featureType = feature["featureType"]
            if featureType == "mate":
                mateFeature = AssemblyMateFeature.from_json(feature, ref)
            elif featureType == "mateGroup":
                mateFeature = AssemblyMateGroupFeature.from_json(feature, ref)
            elif featureType == "mateConnector":
                mateFeature = AssemblyMateConnectorFeature.from_json(feature, ref)
            else:
                raise Exception("unknown feature type: " + featureType)
            features[mateFeature.id] = mateFeature

        instanceByName = {}
        for instance in instances.values():
            assert instance.name not in instanceByName
            instanceByName[instance.name] = instance.id

        return cls(
            ref=ref,
            instances=instances,
            instanceByName=instanceByName,
            features=features
        )

@dataclass(frozen=True)
class Part:
    type = "part"
    isStandardContent: bool
    bodyType: str
    ref: PartRef
    documentVersion: str

    @classmethod
    def from_json(cls, data: dict):
        return cls(
            isStandardContent=data["isStandardContent"],
            bodyType=data["bodyType"],
            ref=PartRef.from_json(data),
            documentVersion=data.get("documentVersion", None)
        )
    
def readCSMatrix(data: dict) -> np.array:
    partMateT = np.eye(4)
    partMateT[:3, :3] = np.stack(
        [
            np.array(data["xAxis"]),
            np.array(data["yAxis"]),
            np.array(data["zAxis"])
        ]
    ).T
    partMateT[:3, 3] = data["origin"]
    return partMateT

@dataclass(frozen=True)
class Occurrence:
    path: InstancePath
    transform: np.array
    fixed: bool
    hidden: bool

    @classmethod
    def from_json(cls, data: dict, root: QualifiedRef):
        return cls(
            path=InstancePath(root, data["path"]),
            transform=np.array(np.reshape(data["transform"], (4, 4))),
            fixed=data["fixed"],
            hidden=data["hidden"]
        )

#@dataclass
#class FeatureDesc:
#    id: str
#    name: str
#    parameters: dict

@dataclass
class BTMType:
    @classmethod
    def deserialize(_cls, data: dict):
        raise NotImplementedError()

@dataclass
class BTMParameter:
    parameterId: str
    nodeId: str

    @classmethod
    def deserialize(_cls, data: dict):
        return dict(
            parameterId=data["parameterId"],
            nodeId=data["nodeId"]
        )

    def readValue(self, data):
        raise NotImplementedError()

@dataclass
class BTMParameterEnum(BTMParameter):
    type: str
    value: str

    @classmethod
    def deserialize(_cls, data: dict):
        return dict(
            type=data["enumName"],
            value=data["value"],
            **BTMParameter.deserialize(data)
        )

@dataclass
class BTMParameterBoolean(BTMParameter):
    value: bool

    @classmethod
    def deserialize(_cls, data: dict):
        return dict(
            value=data["value"],
            **BTMParameter.deserialize(data)
        )

@dataclass
class BTMParameterQuantity(BTMParameter):
    units: str
    value: float | int
    expression: str
    isInteger: bool

    @classmethod
    def deserialize(_cls, data: dict):
        return dict(
            units=data["units"],
            value=data["value"],
            expression=data["expression"],
            isInteger=data["isInteger"],
            **BTMParameter.deserialize(data)
        )

    def readValue(self, data):
        return data.readExpression(self.expression)

@dataclass
class BTMParameterNullableQuantity(BTMParameterQuantity):
    isNull: bool
    nullValue: str

    @classmethod
    def deserialize(_cls, data: dict):
        return dict(
            isNull=data["isNull"],
            nullValue=data["nullValue"],
            **BTMParameterQuantity.deserialize(data)
        )

    def readValue(self, data):
        if self.nullValue:
            return None

        return super(BTMParameterNullableQuantity, self).readValue(data)

btm_types = {
    "BTMParameterEnum": BTMParameterEnum,
    "BTMParameterQuantity": BTMParameterQuantity,
    "BTMParameterBoolean": BTMParameterBoolean,
    "BTMParameterNullableQuantity": BTMParameterNullableQuantity
}

@dataclass
class Mate:
    id: str
    name: str
    parameters: list[BTMParameter]

@dataclass(frozen=True)
class InstancePathLookup:
    isAbsolute: bool
    occurrence: Occurrence
    instance: Instance
    archetype: Assembly | Part

    def canonicalizePath(self, path: InstancePath):
        assert path.root == self.instance.ref
        return InstancePath(
            root=self.occurrence.path.root,
            elements=self.occurrence.path.elements + path.elements
        )

@dataclass(frozen=True)
class InstanceIdLookup:
    instance: Instance
    archetype: Assembly | Part

@dataclass(frozen=True)
class QualifiedRefLookup:
    archetype: Assembly | Part

@dataclass(frozen=True)
class FeatureIdLookup:
    pass

@dataclass
class OnshapeData:
    client: Client
    documentId: str
    assemblyId: str
    configuration: str

    assembly: dict
    jointFeatures: dict

    rootAssemblyId: QualifiedRef

    def __getitem__(self, index: InstancePath) -> InstancePathLookup:
        ...

    def __getitem__(self, index: InstanceId) -> InstanceIdLookup:
        ...

    def __getitem__(self, index: QualifiedRef) -> QualifiedRefLookup:
        ...

    def __getitem__(self, index):
        def getByQRef(qref):
            if isinstance(qref, QualifiedRef):
                return self.assemblies[qref]
            if isinstance(qref, PartRef):
                return self.parts[qref]
            raise Exception("invalid type" + str(qref))

        if isinstance(index, QualifiedRef) or isinstance(index, PartRef):
            return QualifiedRefLookup(
                archetype=getByQRef(index)
            )

        if isinstance(index, InstanceId):
            inst = self.instances[index]
            return InstanceIdLookup(
                instance=inst,
                archetype=getByQRef(inst.ref)
            )

        if isinstance(index, InstancePath):
            # Occurrence only present if we have an absolute path.
            isAbsolute = self.rootAssemblyId == index.root

            occ = None
            if isAbsolute:
                occ = self.occurrences[index]

            inst = self.instances[index.instance]
            return InstancePathLookup(
                isAbsolute=isAbsolute,
                occurrence=occ,
                instance=inst,
                archetype=getByQRef(inst.ref)
            )

        raise ValueError("invalid index: " + str(index))

    def pathPrettyName(self, path: InstancePath):
        segments = []
        for idx in range(len(path)):
            segments.append(self[path[idx]].instance.name)
        return " / ".join(segments)

    def getPartMassProperties(self, ref: PartRef):
        lookup = self[ref]

        if lookup.archetype.isStandardContent:
            # Workaround for issues with OnShape API.
            # For some reason you need to retrieve the mass properties
            # of standard content in this very specific and stupid way.
            # I can understand some of this somewhat, but WHY WHY do you
            # specifically need to use `documentVersion`?? Why does
            # `documentMicroversion` fail here and only here? So stupid.
            assert lookup.archetype.documentVersion is not None
            return self.client.get_direct_part_mass_properties_override_version(
                lookup.archetype.ref,
                lookup.archetype.documentVersion,
                linkDocumentId=self.rootAssemblyId.documentId,
                useMassPropertyOverrides=True,
                inferMetadataOwner=True
            )
        else:
            return self.client.get_part_mass_properties(
                lookup.archetype.ref, 
                linkDocumentId=self.rootAssemblyId.documentId, 
                useMassPropertyOverrides=True
            )

    @cached_property
    def occurranceNamePaths(self) -> dict[tuple[str], InstancePath]:
        result = {}

        def traverse(inst: InstanceId, path: tuple[InstanceId], namePath: tuple[str]):
            lookup = self[inst]

            subPath = tuple(itertools.chain(path, [inst]))
            subNamePath = tuple(itertools.chain(namePath, [lookup.instance.name]))

            assert subNamePath not in result
            result[subNamePath] = InstancePath(self.rootAssemblyId, subPath)

            if lookup.archetype.type == "assembly":
                for sub in lookup.archetype.instances.values():
                    traverse(sub.id, subPath, subNamePath)

        for instanceId in self[self.rootAssemblyId].archetype.instances.keys():
            traverse(instanceId, tuple(), tuple())

        return result

    @cached_property
    def instances(self) -> dict[InstanceId, Instance]:
        instances = {}
        for assembly in self.assemblies.values():
            for instance in assembly.instances.values():
                instances[instance.id] = instance
        return instances

    @cached_property
    def parts(self) -> dict[PartRef, Part]:
        parts = {}
        for partsData in self.assembly["parts"]:
            part = Part.from_json(partsData)
            parts[part.ref] = part
        return parts

    @cached_property
    def assemblies(self) -> dict[QualifiedRef, Assembly]:
        assemblies = {}

        assembly = Assembly.from_json(self.assembly["rootAssembly"])
        assemblies[assembly.ref] = assembly

        for assemblyData in self.assembly["subAssemblies"]:
            assembly = Assembly.from_json(assemblyData)
            assemblies[assembly.ref] = assembly

        return assemblies

    @cached_property
    def occurrenceToRef(self) -> dict[InstancePath, QualifiedRef]:
        occurrenceToRef = {}

        for path in self.occurrences.keys():
            current = self.assemblies[self.rootAssemblyId]
            for element in path.elements:
                current = current.instances[element]
            occurrenceToRef[path] = current.ref

        return occurrenceToRef

    @cached_property
    def occurrences(self) -> dict[InstancePath, Occurrence]:
        occurrences = {}

        for occurrenceData in self.assembly["rootAssembly"]["occurrences"]:
            occurrence = Occurrence.from_json(occurrenceData, self.rootAssemblyId)
            occurrences[occurrence.path] = occurrence

        return occurrences

    @cached_property
    def mateParameters(self) -> dict[FeatureId, Mate]:
        mates = {}
        for feature in self.jointFeatures["features"]:
            if feature["typeName"] != "BTMMate":
                continue

            message = feature["message"]
            assert(message["featureType"] == "mate")

            id = FeatureId(message["featureId"])
            name = message["name"]

            parameters = {}
            for param in message["parameters"]:
                typeName = param["typeName"]
                paramMessage = param["message"]
                paramId = paramMessage["parameterId"]
                typeClass = btm_types.get(typeName)

                value = None
                if typeClass is not None:
                    data = typeClass.deserialize(paramMessage)
                    value = typeClass(**data)

                if paramId in parameters:
                    raise Exception("duplicate parameter")

                parameters[paramId] = value

            mates[id] = Mate(
                id=id,
                name=name,
                parameters=parameters
            )

        return mates

    @cached_property
    def configuration_parameters(self):
        # Decode root config params
        config_params = {}
        parts = self.assembly["root"]["fullConfiguration"].split(";")
        for part in parts:
            kv = part.split("=")
            if len(kv) == 2:
                config_params[kv[0]] = kv[1].replace('+', ' ')
        return config_params

    def readExpression(self, expression: str):
        # TODO improve
        if expression[0] == "#":
            expression = self.configuration_parameters[expression[1:]]
        if expression[0:2] == "#":
            expression = "-" + self.configuration_parameters[expression[2:]]

        parts = expression.split(' ')
        if parts[1] == 'deg':
            return float(parts[0]) / 360.0 * (2*math.pi)
        if parts[1] in ['radian', 'rad']:
            if parts[0] == '(PI)':
                value = math.pi
            else:
                value = float(parts[0])
            return value
        if parts[1] == 'mm':
            return float(parts[0]) * ureg.millimeter
        if parts[1] == 'cm':
            return float(parts[0]) * ureg.centimeter
        if parts[1] == 'm':
            return float(parts[0]) * ureg.meter
        if parts[1] == 'in':
            return float(parts[0]) * ureg.inch
        raise NotImplementedError()

    @classmethod
    def from_onshape(_cls, client: Client, documentId: str, assemblyName: str, versionId=None, workspaceId=None, configuration="default"):
        # If none of version/workspace is specified, we fetch the default workspace.
        if versionId is None and workspaceId is None:
            document = client.get_document(documentId=documentId)
            workspaceId = document["defaultWorkspace"]["id"]

        # Fetch the current microversion for workspace/version.
        microversionId = client.get_microversion(documentId=documentId, versionId=versionId, workspaceId=workspaceId)["microversion"]

        # Fetch base elements from API
        elements = client.list_elements_by_microversion(documentId, microversionId=microversionId)

        # Find assembly which matches specified name.
        assemblyId = None
        for element in elements:
            if element["type"] == "Assembly" and element["name"] == assemblyName:
                assemblyId = element["id"]
        if assemblyId is None:
            raise Exception("could not find assembly by name in document")

        # Fetch base data for assembly.
        assemblyRef = QualifiedRef(documentId=documentId, elementId=assemblyId, microversionId=microversionId, configuration=configuration)
        assembly = client.get_assembly(assemblyRef)
        with open("assembly.json", "w") as f:
            f.write(json.dumps(assembly))
        joint_features = client.get_features(assemblyRef)

        return OnshapeData(
            client=client,
            documentId=documentId,
            assemblyId=assemblyId,
            configuration=configuration,

            assembly=assembly,
            jointFeatures=joint_features,

            rootAssemblyId=assemblyRef
        )