import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processing import ImagePreprocessor

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.preprocessor = ImagePreprocessor(target_size=(640, 640))
        self.test_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    def test_resize_with_padding(self):
        resized, info = self.preprocessor.resize_with_padding(self.test_image)
        
        self.assertEqual(resized.shape[:2], (640, 640))
        self.assertIn('scale', info)
        self.assertIn('pad_x', info)
        self.assertIn('pad_y', info)
    
    def test_color_correction(self):
        corrected = self.preprocessor.color_correction(self.test_image)
        
        self.assertEqual(corrected.shape, self.test_image.shape)
        self.assertEqual(corrected.dtype, np.uint8)
    
    def test_preprocess_pipeline(self):
        processed, info = self.preprocessor.preprocess_pipeline(self.test_image)
        
        self.assertEqual(processed.shape[:2], (640, 640))
        self.assertIsNotNone(info)

if __name__ == '__main__':
    unittest.main()
