import unittest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.yolov10_crack import YOLOv10Crack
from models.attention import CBAM

class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = YOLOv10Crack(num_classes=3)
        cls.batch_size = 2
        cls.input_size = (3, 640, 640)
    
    def test_model_forward(self):
        dummy_input = torch.randn(self.batch_size, *self.input_size)
        output = self.model(dummy_input)
        
        self.assertIsNotNone(output)
        self.assertIsInstance(output, torch.Tensor)
    
    def test_model_output_shape(self):
        dummy_input = torch.randn(self.batch_size, *self.input_size)
        output = self.model(dummy_input)
        
        self.assertEqual(output.dim(), 3)
    
    def test_model_parameters(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.assertGreater(total_params, 0)
        self.assertEqual(total_params, trainable_params)

class TestAttention(unittest.TestCase):
    def setUp(self):
        self.in_channels = 256
        self.cbam = CBAM(self.in_channels)
    
    def test_cbam_forward(self):
        dummy_input = torch.randn(2, self.in_channels, 32, 32)
        output = self.cbam(dummy_input)
        
        self.assertEqual(output.shape, dummy_input.shape)

if __name__ == '__main__':
    unittest.main()
