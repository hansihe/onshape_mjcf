'''
client
======

Convenience functions for working with the Onshape API
'''

#from onshape_mjcf.onshape_data import QualifiedRef
from collections import OrderedDict
from dataclasses import dataclass
from .onshape import Onshape

import mimetypes
import random
import string
import os
import json
import hashlib
import base64
from pathlib import Path

from keyvalue_sqlite import KeyValueSqlite

DB_PATH = "./cache.sqlite"
cache = KeyValueSqlite(DB_PATH, "request_cache")

def escape_url(s):
    return s.replace('/', '%2f').replace('+', '%2b')

@dataclass
class OnshapeCredentials:
    url: str
    access_key: str
    secret_key: str

    @classmethod
    def from_env(_cls):
        url = os.getenv("ONSHAPE_API")
        access_key = os.getenv("ONSHAPE_ACCESS_KEY")
        secret_key = os.getenv("ONSHAPE_SECRET_KEY")

        if url is None or access_key is None or secret_key is None:
            raise Exception("""
            ERROR: No OnShape API access key set.

            TIP: Connect to https://dev-portal.onshape.com/keys, and edit your .bashrc file:
            export ONSHAPE_API=https://cad.onshape.com
            export ONSHAPE_ACCESS_KEY=Your_Access_Key
            export ONSHAPE_SECRET_KEY=Your_Secret_Key
            """)

        return OnshapeCredentials(
            url=url,
            access_key=access_key,
            secret_key=secret_key
        )

class Client():
    '''
    Defines methods for testing the Onshape API. Comes with several methods:

    - Create a document
    - Delete a document
    - Get a list of documents

    Attributes:
        - logging (bool, default=True): Turn logging on or off
    '''

    def __init__(self, credentials, logging=False):
        '''
        Instantiates a new Onshape client.

        Args:
            - logging (bool, default=True): Turn logging on or off
        '''

        self._metadata_cache = {}
        self._massproperties_cache = {}
        self._api = Onshape(credentials=credentials, logging=logging)
        self.useCollisionsConfigurations = True

    def request_cached(self, method, url, query={}, body={}, headers={}):
        key = {
            "method": method,
            "url": url,
            "query": OrderedDict(query.items()),
            "body": OrderedDict(body.items()),
            "headers": OrderedDict(headers.items())
        }
        key_json = json.dumps(key)

        cache_data = cache.get(key_json)
        if cache_data is None:
            response = self._api.request(method, url, query=query, body=body)

            if response.status_code != 200:
                raise Exception(f"bad response: {response}")

            content_type = response.headers["content-type"]
            match content_type:
                case "application/json;charset=utf-8":
                    cache_data = {"dt": "json", "d": response.json()}
                case "application/sla;charset=utf-8":
                    cache_data = {"dt": "bin", "d": base64.b64encode(response.content).decode()}
                case _:
                    raise Exception(f"unknown content type {content_type}")

            cache.set(key_json, cache_data)

        match cache_data["dt"]:
            case "json":
                return cache_data["d"]
            case "bin":
                return base64.b64decode(cache_data["d"])
            case _:
                raise Exception(f"unknown datatype signifier: {cache_data["dt"]}")


    # NOT cached
    def get_document(self, documentId):
        return self._api.request('get', f'/api/documents/{documentId}').json()

    # NOT cached
    def get_microversion(self, documentId, workspaceId=None, versionId=None):
        if workspaceId is None and versionId is None:
            raise Exception("either workspace or version id required")
        if workspaceId is not None and versionId is not None:
            raise Exception("only one of workspace and version id can be passed")

        if workspaceId is not None:
            url = f"/api/documents/d/{documentId}/w/{workspaceId}/currentmicroversion"
        if versionId is not None:
            url = f"/api/documents/d/{documentId}/v/{versionId}/currentmicroversion"
        
        return self._api.request("get", url).json()

    def list_elements_by_microversion(self, documentId, microversionId):
        url = f'/api/documents/d/{documentId}/m/{microversionId}/elements'
        return self.request_cached('get', url)

    def get_assembly(self, ref):
        query = {'includeMateFeatures': 'true', 'includeMateConnectors': 'true', 'includeNonSolids': 'true', 'configuration': ref.configuration}
        url = f'/api/assemblies/d/{ref.documentId}/m/{ref.microversionId}/e/{ref.elementId}'
        return self.request_cached('get', url, query=query)

    def get_features(self, ref):
        query = {'configuration': ref.configuration}
        url = f'/api/assemblies/d/{ref.documentId}/m/{ref.microversionId}/e/{ref.elementId}/features'
        return self.request_cached("get", url, query=query)

    def get_sketches(self, ref, includeGeometry=True):
        query = {"configuration": ref.configuration, "includeGeometry": includeGeometry}
        url = f'/api/partstudios/d/{ref.documentId}/m/{ref.microversionId}/e/{ref.elementId}/sketches'
        return self.request_cached("get", url, query=query)

    def get_parts(self, ref):
        query = {"configuration": ref.configuration}
        url = f'/api/parts/d/{ref.documentId}/m/{ref.microversionId}/e/{ref.elementId}'
        return self.request_cached("get", url, query=query)

    def get_part_metadata(self, ref, partId: str):
        query = {"configuration": ref.configuration}
        url = f'/api/metadata/d/{ref.documentId}/m/{ref.microversionId}/e/{ref.elementId}/p/{partId}'
        return self.request_cached("get", url, query=query)

    def get_part_studio_mass_properties(self, ref, linkDocumentId=None, useMassPropertyOverrides=False, massAsGroup=False):
        query = {
            "configuration": ref.configuration,
            "useMassPropertyOverrides": useMassPropertyOverrides,
            "massAsGroup": massAsGroup
        }
        if linkDocumentId:
            query["linkDocumentId"] = linkDocumentId

        url = f'/api/partstudios/d/{ref.documentId}/m/{ref.microversionId}/e/{ref.elementId}/massproperties'
        return self.request_cached("get", url, query=query)

    def get_part_mass_properties(self, ref, linkDocumentId=None, useMassPropertyOverrides=False):
        part_studio_props = self.get_part_studio_mass_properties(ref.ref, linkDocumentId=linkDocumentId, useMassPropertyOverrides=useMassPropertyOverrides)
        return part_studio_props["bodies"][ref.partId]

    """
    Workaround due to buggy Onshape API.
    This needs to be used for standard content. Normally `get_part_mass_properties` can be used.
    """
    def get_direct_part_mass_properties(self, ref, linkDocumentId=None, useMassPropertyOverrides=False, inferMetadataOwner=False):
        query = {
            "configuration": ref.ref.configuration,
            "useMassPropertyOverrides": useMassPropertyOverrides,
            "inferMetadataOwner": inferMetadataOwner
        }
        if linkDocumentId:
            query["linkDocumentId"] = linkDocumentId

        url = f'/api/parts/d/{ref.ref.documentId}/m/{ref.ref.microversionId}/e/{ref.ref.elementId}/partid/{ref.partId}/massproperties'
        return self.request_cached("get", url, query=query)["bodies"][ref.partId]

    """
    Workaround due to buggy Onshape API.
    This needs to be used for standard content. Normally `get_part_mass_properties` can be used.
    """
    def get_direct_part_mass_properties_override_version(self, ref, overrideVersion, linkDocumentId=None, useMassPropertyOverrides=False, inferMetadataOwner=False):
        query = {
            "configuration": ref.ref.configuration,
            "useMassPropertyOverrides": useMassPropertyOverrides,
            "inferMetadataOwner": inferMetadataOwner
        }
        if linkDocumentId:
            query["linkDocumentId"] = linkDocumentId

        url = f'/api/parts/d/{ref.ref.documentId}/v/{overrideVersion}/e/{ref.ref.elementId}/partid/{ref.partId}/massproperties'
        return self.request_cached("get", url, query=query)["bodies"][ref.partId]

    #def get_part_mass_properties(self, ref, linkDocumentId=None, useMassPropertyOverrides=False):
    #    query = {
    #        "configuration": ref.ref.configuration,
    #        "useMassPropertyOverrides": useMassPropertyOverrides
    #    }
    #    if linkDocumentId:
    #        query["linkDocumentId"] = linkDocumentId
    #    
    #    url = f'/api/parts/d/{ref.ref.documentId}/m/{ref.ref.microversionId}/e/{ref.ref.elementId}/partid/{ref.partId}/massproperties'

    #    return self.request_cached("get", url, query=query)

    def get_assembly_mass_properties(self, ref, linkDocumentId=None):
        query = {
            "configuration": ref.configuration
        }
        if linkDocumentId:
            query["linkDocumentId"] = linkDocumentId

        url = f'/api/assemblies/d/{ref.documentId}/m/{ref.microversionId}/e/{ref.elementId}/massproperties'

        return self.request_cached("get", url, query=query)

    def get_part_body_details(self, ref):
        query = {
            "configuration": ref.ref.configuration
        }
        url = f'/api/parts/d/{ref.ref.documentId}/m/{ref.ref.microversionId}/e/{ref.ref.elementId}/partid/{ref.partId}/bodydetails'
        return self.request_cached("get", url, query=query)

    def get_part_stl(self, ref, units="meter", mode="binary"):
        req_headers = {
            'Accept': '*/*'
        }
        query = {
            "units": units,
            "mode": mode
        }
        url = f'/api/parts/d/{ref.ref.documentId}/m/{ref.ref.microversionId}/e/{ref.ref.elementId}/partid/{ref.partId}/stl'
        return self.request_cached("get", url, query=query, headers=req_headers)

    def part_studio_stl(self, did, wid, eid):
        '''
        Exports STL export from a part studio

        Args:
            - did (str): Document ID
            - wid (str): Workspace ID
            - eid (str): Element ID

        Returns:
            - requests.Response: Onshape response data
        '''

        req_headers = {
            'Accept': '*/*'
        }
        return self._api.request('get', '/api/partstudios/d/' + did + '/w/' + wid + '/e/' + eid + '/stl', headers=req_headers)

    def find_new_partid(self, did, mid, eid, partid, configuration_before, configuration):
        before = self.get_parts(did, mid, eid, configuration_before)
        name = None
        for entry in before:
            if entry['partId'] == partid:
                name = entry['name']
        
        if name is not None:
            after = self.get_parts(did, mid, eid, configuration)
            for entry in after:
                if entry['name'] == name:
                    return entry['partId']
        else:
            print("OnShape ERROR: Can't find new partid for "+str(partid))

        return partid

    def part_studio_stl_m(self, did, mid, eid, partid, configuration = 'default'):
        if self.useCollisionsConfigurations:
            configuration_before = configuration
            parts = configuration.split(';')
            partIdChanged = False
            result = ''
            for k, part in enumerate(parts):
                kv = part.split('=')
                if len(kv) == 2:
                    if kv[0] == 'collisions':
                        kv[1] = 'true'
                        partIdChanged = True
                parts[k] = '='.join(kv)
            configuration = ';'.join(parts)

            if partIdChanged:
                partid = self.find_new_partid(did, mid, eid, partid, configuration_before, configuration)
    
        def invoke():
            req_headers = {
                'Accept': '*/*'
            }
            return self._api.request('get', '/api/parts/d/' + did + '/m/' + mid + '/e/' + eid + '/partid/'+escape_url(partid)+'/stl', query={'mode': 'binary', 'units': 'meter', 'configuration': configuration}, headers=req_headers)

        return self.cache_get('part_stl', (did, mid, eid, self.hash_partid(partid), configuration), invoke)