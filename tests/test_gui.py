import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestGUI(unittest.TestCase):
    def test_import_pyqt5(self):
        try:
            from PyQt5.QtWidgets import QApplication
            self.assertTrue(True)
        except ImportError:
            self.skipTest("PyQt5 not installed")
    
    def test_main_window_creation(self):
        try:
            from PyQt5.QtWidgets import QApplication
            from gui.main_window import MainWindow
            
            app = QApplication([])
            window = MainWindow()
            
            self.assertIsNotNone(window)
            self.assertEqual(window.windowTitle(), "农田干裂程度识别系统")
        except ImportError:
            self.skipTest("PyQt5 not installed")

if __name__ == '__main__':
    unittest.main()
