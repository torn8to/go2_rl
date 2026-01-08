from gstaichi import _ti_core
import genesis as gs
from torch import Tensor



def terrain_geom_to_height_map()

@ti.data_oriented
class HeightMap:
    def __init__(self, size=(30, 30), resolution=0.1):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.height_map: Tensor = ti.field(dtype=ti.float32, shape=(int(width/resolution+1), int(height/resolution+1)))

    @ti.kernel
    def query_position(pos: ti.types.vector(3, ti.float32), size: ti.types.vector(2, ti.float32)) -> ti.field:
        pass

