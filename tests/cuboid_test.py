from CrystalSlice import Cuboid

def test_cuboid_sample_slice():
    c = Cuboid(0.5,0.5)
    n = c.sample_slice(100)

    assert len(n) == 100

    img = c.create_10x10_slices()
    c.plot()
    c.rotated_plot()
    c.plot_intersect()