from functools import cached_property
import math
from typing import Self
import numpy as np
from scipy.spatial.transform import Rotation
import lxml.etree
import lxml.builder

E = lxml.builder.ElementMaker()

float_format = "{:.8f}".format

def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)

class SetterProperty(object):
    def __init__(self, func, doc=None):
        self.func = func
        self.__doc__ = doc if doc is not None else func.__doc__
    def __set__(self, obj, value):
        return self.func(obj, value)

class BaseBuilder:
    def __init__(self, element):
        self.elem = element

    def _addElem(self, typ):
        elem = E(typ)
        self.elem.append(elem)
        return elem

class PositionMixin:
    @SetterProperty
    def pos(self, pos: np.array):
        assert pos.shape == (3,)
        self.elem.set("pos", " ".join(map(float_format, pos)))

class RotationMixin:
    @SetterProperty
    def quat(self, quat: np.array):
        assert quat.shape == (4,)
        self.elem.set("quat", " ".join(map(float_format, quat)))
    @SetterProperty
    def euler(self, euler: np.array):
        assert euler.shape == (3,)
        self.elem.set("euler", " ".join(map(float_format, euler)))

# interesting: 9

#from itertools import permutations
#quat_order = list(list(permutations([0, 1, 2, 3]))[18])
#print(quat_order)
#assert False
quat_order = [3, 0, 1, 2]

class TransformMixin(PositionMixin, RotationMixin):
    @SetterProperty
    def transform(self, transform: np.array):
        transform = np.asarray(transform).copy()
        assert transform.shape == (4, 4)
        self.pos = transform[:3, 3].reshape((3,))
        #self.quat = Rotation.from_matrix(transform[:3, :3]).as_quat()[quat_order] 

        # WORKS (??)
        self.euler = Rotation.from_matrix(transform[:3, :3]).as_euler("XYZ")
        #self.quat = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]] 

class NameMixin:
    @SetterProperty
    def name(self, value):
        self.elem.set("name", value)

class JointBuilder(NameMixin, BaseBuilder):
    @SetterProperty
    def range(self, value: tuple[float, float]):
        minL, maxL = value
        self.elem.set("range", float_format(minL) + " " + float_format(maxL))
    @SetterProperty
    def axis(self, value: np.ndarray):
        self.elem.set("axis", " ".join(map(float_format, value)))
    @SetterProperty
    def pos(self, value: np.ndarray):
        self.elem.set("pos", " ".join(map(float_format, value)))
    @SetterProperty
    def frictionloss(self, value: float):
        self.elem.set("frictionloss", float_format(value))
    @SetterProperty
    def type(self, value: str):
        self.elem.set("type", value)

class InertialBuilder(BaseBuilder):
    @SetterProperty
    def inertia(self, inertia: np.array):
        assert check_symmetric(inertia)
        elems = [
            inertia[0, 0],
            inertia[1, 1],
            inertia[2, 2],
            inertia[0, 1],
            inertia[0, 2],
            inertia[1, 2]
        ]
        self.elem.set("fullinertia", " ".join(map(float_format, elems)))
    @SetterProperty
    def diaginertia(self, axes: np.array):
        assert axes.shape == (3,)
        self.elem.set("diaginertia", " ".join(map(float_format, axes)))
    @SetterProperty
    def xyaxes(self, axes: np.array):
        assert axes.shape == (2, 3)
        self.elem.set("xyaxes", " ".join(map(float_format, axes.reshape(6))))
    @SetterProperty
    def mass(self, mass: float):
        self.elem.set("mass", float_format(mass))
    @SetterProperty
    def pos(self, pos: np.ndarray):
        self.elem.set("pos", " ".join(map(float_format, pos)))

class GeomBuilder(NameMixin, TransformMixin, BaseBuilder):
    @SetterProperty
    def type(self, value: str):
        assert value in ["plane", "hfield", "sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh", "sdf"]
        self.elem.set("type", value)
    @SetterProperty
    def mesh(self, value: str):
        self.elem.set("mesh", value)
    @SetterProperty
    def contype(self, value: int):
        self.elem.set("contype", f"{value}")
    @SetterProperty
    def conaffinity(self, value: int):
        self.elem.set("conaffinity", f"{value}")
    @SetterProperty
    def size(self, value):
        npValue = np.asarray(value)
        if len(npValue.shape) == 0:
            self.elem.set("size", float_format(value))
        else:
            self.elem.set("size", " ".join(map(float_format, npValue)))
    @SetterProperty
    def fromto(self, value: np.array):
        assert value.shape == (2, 3)
        self.elem.set("fromto", " ".join(map(float_format, value.reshape(6))))

class SiteBuilder(NameMixin, PositionMixin, BaseBuilder):
    pass

class BodyBuilder(NameMixin, TransformMixin, BaseBuilder):
    def body(self, name) -> Self:
        Ebody = E.body()
        Ebody.set("name", name)
        self.elem.append(Ebody)
        return BodyBuilder(Ebody)

    def joint(self) -> JointBuilder:
        Ejoint = E.joint()
        self.elem.append(Ejoint)
        return JointBuilder(Ejoint)

    def freeJoint(self) -> JointBuilder:
        Efreejoint = E.freejoint()
        self.elem.append(Efreejoint)
        return JointBuilder(Efreejoint)

    def geom(self) -> GeomBuilder:
        Egeom = E.geom()
        self.elem.append(Egeom)
        return GeomBuilder(Egeom)

    def site(self) -> SiteBuilder:
        Esite = E.site()
        self.elem.append(Esite)
        return SiteBuilder(Esite)

    @cached_property
    def inertial(self) -> InertialBuilder:
        Einertial = E.inertial()
        self.elem.append(Einertial)
        return InertialBuilder(Einertial)

class MeshBuilder(NameMixin, BaseBuilder):
    @SetterProperty
    def file(self, value: str):
        self.elem.set("file", value)

class AssetBuilder(BaseBuilder):
    def mesh(self):
        return MeshBuilder(self._addElem("mesh"))

class ConnectBuilder(NameMixin, BaseBuilder):
    @SetterProperty
    def body1(self, name: str):
        self.elem.set("body1", name)
    @SetterProperty
    def body2(self, name: str):
        self.elem.set("body2", name)
    @SetterProperty
    def anchor(self, pos: np.array):
        assert pos.shape == (3,)
        self.elem.set("anchor", " ".join(map(float_format, pos)))

class EqualityBuilder(BaseBuilder):
    def connect(self) -> ConnectBuilder:
        return ConnectBuilder(self._addElem("connect"))

class MJCFBuilder:
    def __init__(self):
        self.Emjcf = E.mujoco()

    @property
    def element(self):
        return self.Emjcf

    @cached_property
    def worldBody(self) -> BodyBuilder:
        self.Eworldbody = E.worldbody()
        self.Emjcf.append(self.Eworldbody)
        return BodyBuilder(self.Eworldbody)

    @cached_property
    def asset(self) -> AssetBuilder:
        self.Easset = E.asset()
        self.Emjcf.append(self.Easset)
        return AssetBuilder(self.Easset)

    @cached_property
    def equality(self) -> EqualityBuilder:
        self.Eequality = E.equality()
        self.Emjcf.append(self.Eequality)
        return EqualityBuilder(self.Eequality)

    def toString(self):
        return lxml.etree.tostring(self.Emjcf, pretty_print=True, encoding=str)