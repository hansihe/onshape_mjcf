import numpy as np

from onshape_mjcf.onshape_data import OnshapeData

def get_T_part_mate(matedEntity: dict):
    T_part_mate = np.eye(4)
    T_part_mate[:3, :3] = np.stack(
        (
            np.array(matedEntity["matedCS"]["xAxis"]),
            np.array(matedEntity["matedCS"]["yAxis"]),
            np.array(matedEntity["matedCS"]["zAxis"]),
        )
    ).T
    T_part_mate[:3, 3] = matedEntity["matedCS"]["origin"]

    return T_part_mate

def getLimits(data: OnshapeData, jointType, id):
    mateData = data.mateParameters[id]
    enabled = mateData.parameters["limitsEnabled"].value == True

    if enabled:
        if jointType == 'revolute':
            minimum = mateData.parameters["limitAxialZMin"].readValue(data)
            maximum = mateData.parameters["limitAxialZMax"].readValue(data)
        elif jointType == "prismatic":
            minimum = mateData.parameters["limitZMin"].readValue(data)
            maximum = mateData.parameters["limitZMax"].readValue(data)
        return (minimum, maximum)