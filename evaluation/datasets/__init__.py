from . import chexpert, indiana, padchest

LOADERS = {"indiana": indiana.load, "padchest": padchest.load, "chexpert": chexpert.load}
