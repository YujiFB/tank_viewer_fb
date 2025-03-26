"""SkepticalFox 2015-2024"""

# imports
import logging
import os
from dataclasses import dataclass
from typing import IO

# blender imports
import bpy
import numpy as np
from mathutils import Vector  # type: ignore

# local imports
from ..ResourceManager import ResourceManager
from .tag_tools import TagFileType, TagObject, TagReader

logger = logging.getLogger(__name__)


def unpack_havok_packed_vert(packed: int) -> tuple[float, float, float]:
    """
    From https://kr.cm/f/t/43667/c/444660/
    10 + 11 + 11
    """
    tx = float((packed & 0xFFC00000) >> 22) / 0x3FF
    ty = float((packed & 0x3FF800) >> 11) / 0x7FF
    tz = float(packed & 0x7FF) / 0x7FF
    return (tz, tx, ty)


def unpack_havok_shared_vert(packed: int) -> tuple[float, float, float]:
    """
    From https://kr.cm/f/t/43667/c/444660/
    22 + 21 + 21
    """
    tx = float((packed >> 42) & 0x3FFFFF) / 0x3FFFFF
    ty = float((packed >> 21) & 0x1FFFFF) / 0x1FFFFF
    tz = float(packed & 0x1FFFFF) / 0x1FFFFF
    return (tz, tx, ty)


def unique_without_sort(arr):
    return arr[np.sort(np.unique(arr, return_index=True)[1])]


@dataclass(init=False)
class HavokSection:
    domain_min: np.array
    domain_max: np.array
    firstPackedVertexIndex: int
    firstSharedVertexIndex: int
    firstPrimitiveIndex: int
    numPackedVertices: int
    numPrimitives: int
    firstDataRunIndex: int
    numDataRuns: int

    def __init__(self, section: dict[bytes, TagObject]):
        self.domain_min = np.fromiter(
            map(
                lambda x: x.value,
                section[b'domain'].value[b'min'].value,
            ),
            dtype=np.float32,
        )
        self.domain_max = np.fromiter(
            map(
                lambda x: x.value,
                section[b'domain'].value[b'max'].value,
            ),
            dtype=np.float32,
        )
        self.firstPackedVertexIndex = section[b'firstPackedVertexIndex'].value
        self.firstSharedVertexIndex = section[b'firstSharedVertexIndex'].value
        self.firstPrimitiveIndex = section[b'firstPrimitiveIndex'].value
        self.numPackedVertices = section[b'numPackedVertices'].value
        self.numPrimitives = section[b'numPrimitives'].value
        self.firstDataRunIndex = section[b'firstDataRunIndex'].value
        self.numDataRuns = section[b'numDataRuns'].value


@dataclass(init=False)
class HavokGeometry:
    primitives: list[np.array]
    sharedVertices: np.array
    sharedVerticesIndex: np.array
    packedVertices: np.array
    sections: list[HavokSection]
    primitiveDataRuns: np.array

    def __init__(self, meshTree: dict[bytes, TagObject]):
        self.primitives = list(
            map(
                lambda x: unique_without_sort(
                    np.fromiter(
                        map(
                            lambda y: y.value,
                            x.value[b'indices'].value,
                        ),
                        dtype=np.uint32,
                    ),
                ),
                meshTree[b'primitives'].value,
            ),
        )
        self.sharedVertices = np.fromiter(
            map(
                lambda x: unpack_havok_shared_vert(x.value),
                meshTree[b'sharedVertices'].value,
            ),
            dtype=[
                ('', np.float32),
                ('', np.float32),
                ('', np.float32),
            ],
        )
        self.sharedVerticesIndex = np.fromiter(
            map(
                lambda x: x.value,
                meshTree[b'sharedVerticesIndex'].value,
            ),
            dtype=np.uint32,
        )
        self.packedVertices = np.fromiter(
            map(
                lambda x: unpack_havok_packed_vert(x.value),
                meshTree[b'packedVertices'].value,
            ),
            dtype=[
                ('', np.float32),
                ('', np.float32),
                ('', np.float32),
            ],
        )
        self.primitiveDataRuns = np.fromiter(
            map(
                lambda x: x.value[b'count'].value,
                meshTree[b'primitiveDataRuns'].value,
            ),
            dtype=np.uint32,
        )
        self.sections = []
        for sec in meshTree[b'sections'].value:
            self.sections.append(HavokSection(sec.value))


def read_geoms_from_havok(f: IO[bytes]) -> list[HavokGeometry]:
    assert TagReader.checkIO(f) == TagFileType.Object
    root_tag = TagReader.fromIO(f)

    assert len(root_tag.value[b'namedVariants'].value) == 1
    vals: list[TagObject] = root_tag.value[b'namedVariants'].value[0].value[b'variant'].value.value[b'resourceHandles'].value

    out = []
    for val in vals:
        subVal = val.value.value[b'variant'].value.value
        if b'bodyCinfos' in subVal:
            for subSubVal in subVal[b'bodyCinfos'].value:
                if b'data' in subSubVal.value[b'shape'].value.value:
                    out.append(HavokGeometry(subSubVal.value[b'shape'].value.value[b'data'].value.value[b'meshTree'].value))

    return out


def load_havok_file(res_mgr: ResourceManager, col: bpy.types.Collection, model: dict):
    filepath = model['File'].replace('.model', '.havok')

    logger.info(f'Start loading havok: {filepath}')
    havok_f = res_mgr.open_file(filepath)

    if not havok_f:
        logger.error('File not found!')
        return

    geoms = read_geoms_from_havok(havok_f)

    for i, geom in enumerate(geoms):
        for j, sec in enumerate(geom.sections):
            ob_name = os.path.basename(filepath).replace('.', '_') + f'_{i}_{j}'

            verts = np.concatenate(
                (
                    geom.packedVertices[sec.firstPackedVertexIndex : sec.firstPackedVertexIndex + sec.numPackedVertices],
                    geom.sharedVertices,
                ),
                axis=0,
            )
            faces = geom.primitives[sec.firstPrimitiveIndex : sec.firstPrimitiveIndex + sec.numPrimitives]

            # FIXME: double check:
            if sec.firstSharedVertexIndex > 0:
                apply_fixup = np.vectorize(
                    lambda x: sec.numPackedVertices + geom.sharedVerticesIndex[sec.firstSharedVertexIndex + (x - sec.numPackedVertices)]
                    if x >= sec.numPackedVertices
                    else x
                )
                faces = list(map(apply_fixup, faces))

            bmesh = bpy.data.meshes.new(f'Mesh_{ob_name}')
            bmesh.from_pydata(verts, [], faces)

            bmesh.validate()
            bmesh.update()

            ob = bpy.data.objects.new(ob_name, bmesh)

            ob.scale = Vector(sec.domain_max).xzy - Vector(sec.domain_min).xzy
            ob.location = Vector(sec.domain_min).xzy + model['Position']
            col.objects.link(ob)
