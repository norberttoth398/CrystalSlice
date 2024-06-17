from CrystalSlice import create_WulffCryst_fromSmorf


def test_cuboid_sample_slice():
    c = create_WulffCryst_fromSmorf("tests/smorf_test_crystallographic.json")
    n = c.sample_slice(100)

    assert len(n) == 100
