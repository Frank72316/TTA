"""Fixed two-view designs for the brightness × translation experiment."""
from itertools import product
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
GEOM = HERE.parent / "geometry_specificity"
sys.path.insert(0,str(GEOM))
from history import read, write

BRIGHT = ("B0.9500","B1.0500")
SHIFT = tuple("T_"+d+"_3" for d in ("left","right","up","down"))
COMBOS = tuple(product(BRIGHT,SHIFT))
VIEWS = ("O",*BRIGHT,*SHIFT,*(b+"+"+t for b,t in COMBOS))
UUID = "GPU-d329c8d0-f276-5f23-03a9-df221a2021ed"


def pixels(original,name):
    from geometry import view
    if "+" in name:
        b,t=name.split("+")
        return view(view(original,b),t)
    return view(original,name)
