import unittest
import xlabs_test

class Tests (unittest.TestCase):
    def test_default(self):
        xlabs_test.go('/home/bolster/Downloads/289-3678-2304-2504_xlabs.csv', calicut=800, interactive=False )