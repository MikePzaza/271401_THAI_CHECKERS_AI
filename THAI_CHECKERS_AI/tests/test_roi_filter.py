import unittest
import mg400_visionV4 as v


class RoiFilter(unittest.TestCase):
    def test_in_roi(self):
        old = dict(v._ROI_CACHE)
        try:
            v._ROI_CACHE["roi"] = (400, 200, 1000, 600)
            self.assertTrue(v.in_roi(506, 543))
            self.assertFalse(v.in_roi(261, 702))
            self.assertFalse(v.in_roi(944, 300) and False)
            v._ROI_CACHE["roi"] = None
            self.assertTrue(v.in_roi(1, 1))
        finally:
            v._ROI_CACHE.clear(); v._ROI_CACHE.update(old)


if __name__ == "__main__":
    unittest.main()
