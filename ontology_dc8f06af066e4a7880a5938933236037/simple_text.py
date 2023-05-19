from decimal import *
from datetime import *
from typing import *
from marshmallow import Schema, fields, post_load
from openfabric_pysdk.concept import OpenfabricConcept

class SimpleText(OpenfabricConcept):
    text: List[str] = []

class SimpleTextSchema(Schema):
    text = fields.List(fields.String())

    @post_load
    def create(self, data, many, **kwargs):
        return SimpleText(data, many=many, **kwargs)
