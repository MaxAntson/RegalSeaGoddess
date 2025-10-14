import json
import os
import pathlib

from box import Box


config = {}
with open(os.path.join(pathlib.Path(__file__).parent, "config.json"), "rb") as f:
    config = Box(json.load(f))
