from CrystalSlice import Ellipsoid

def test_cuboid_sample_slice():
    c = Ellipsoid(0.5,0.5)
    n = c.sample_slice(100)

    assert len(n) == 100