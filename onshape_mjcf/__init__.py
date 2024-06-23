import pint
ureg = pint.UnitRegistry()

from onshape_mjcf.onshape_data import OnshapeData
from onshape_mjcf.onshape_api.client import Client, OnshapeCredentials
from onshape_mjcf.robot_description import RobotDescription
from onshape_mjcf.to_mjcf_basic import MJCFBuildOptions, toMJCFBasic

def toMJCFBasicCommand():
    import argparse

    parser = argparse.ArgumentParser(
        "onshape-to-mjcf-basic"
    )
    parser.add_argument("documentId")
    parser.add_argument("assemblyName")
    parser.add_argument("rootName")
    args = parser.parse_args()

    creds = OnshapeCredentials.from_env()
    client = Client(creds)

    data = OnshapeData.from_onshape(client, args.documentId, args.assemblyName)
    description = RobotDescription.from_onshape(data, args.rootName)

    mjcf = toMJCFBasic(data, description)

    with open("robot.xml", "w") as f:
        f.write(mjcf.toString())