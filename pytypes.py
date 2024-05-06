class BBox:
    def __init__(self, min, max) -> None:
        self.minx = min[0]
        self.miny = min[1]
        self.maxx = max[0]
        self.maxy = max[1]
        assert(self.minx < self.maxx)
        assert(self.miny < self.maxy)
