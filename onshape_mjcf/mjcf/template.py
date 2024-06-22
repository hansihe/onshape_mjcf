from typing import Self
from lxml import etree
import lxml.builder
from io import StringIO

E = lxml.builder.ElementMaker()

class MJCFTemplate:
    def __init__(self, f):
        self.root = etree.parse(f).getroot()
        assert self.root.tag == "mujocotemplate"

    @classmethod
    def from_file(cls, path):
        with open(path, "r") as f:
            return cls(f)

    @classmethod
    def from_str(cls, str: str) -> Self:
        return cls(StringIO(str))

    def merge(self, instRoot):
        assert instRoot.tag == "mujoco"
        newRoot = E.mjcf()

        worldbodyT = None
        worldbodyI = None
        worldbodyO = None

        for child in self.root.iterchildren():
            if child.tag == "worldbody":
                assert worldbodyT is None
                worldbodyT = child
                worldbodyI = instRoot.find("worldbody")
                worldbodyO = E.worldbody()
                newRoot.append(worldbodyO)
            else:
                newRoot.append(child)

        assert worldbodyT is not None
        assert worldbodyI is not None
        assert worldbodyO is not None

        mergeBody(worldbodyT, worldbodyI, worldbodyO)

        return newRoot
    
def mergeBody(bT, bI, bO):
    for elem in bT:
        pass
    for elem in bI:
        pass

singletons = ["worldbody", "assets"]
namedTags = ["body"]

def makeOrder(bT, bI):
    bT_items = list(bT.iterchildren())
    bI_items = list(bI.iterchildren())
