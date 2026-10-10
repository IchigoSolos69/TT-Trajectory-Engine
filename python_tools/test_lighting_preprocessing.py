import unittest
import numpy as np
from lighting_preprocessing import green_channel_rgb


class GreenChannelTests(unittest.TestCase):
    def test_exact_channels_geometry_and_original_unchanged(self):
        source=np.array([[[10,110,240],[80,25,190]],[[0,250,9],[255,255,255]]],dtype=np.uint8)
        before=source.copy();result=green_channel_rgb(source)
        self.assertEqual(result.shape,source.shape)
        self.assertEqual(result.dtype,np.uint8)
        for channel in range(3):np.testing.assert_array_equal(result[:,:,channel],source[:,:,1])
        np.testing.assert_array_equal(source,before)
        self.assertTrue(result.flags.c_contiguous)
        self.assertFalse(np.shares_memory(result,source))

    def test_clipped_information_is_not_invented(self):
        source=np.full((2,3,3),255,dtype=np.uint8)
        np.testing.assert_array_equal(green_channel_rgb(source),source)

    def test_invalid_input_fails_without_normalizing_or_resizing(self):
        for value in (None,np.zeros((2,2),np.uint8),np.zeros((2,2,4),np.uint8),np.zeros((2,2,3),np.float32),np.zeros((0,2,3),np.uint8)):
            with self.assertRaises(ValueError):green_channel_rgb(value)


if __name__=='__main__':unittest.main()
