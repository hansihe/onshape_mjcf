import lxml.etree

from .builder import *
from .template import *

def elementToString(elem):
    return lxml.etree.tostring(elem, pretty_print=True, encoding=str)