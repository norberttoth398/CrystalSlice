from CrystalSlice import ZonedCuboid

def test_cuboid_sample_slice():
    c = ZonedCuboid([[0.5, 0.5], [1,1]], (1,0.5))
    n = c.sample_slice(100)
