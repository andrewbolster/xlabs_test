import unittest
import xlabs_test

class Tests (unittest.TestCase):
    def test_default(self):
        xlabs_test.go('/home/bolster/Downloads/289-3678-2304-2504_xlabs.csv')

    def test_single_cut(self):
        xlabs_test.go('/home/bolster/Downloads/289-3678-2304-2504_xlabs.csv', calicut='800' )

    def test_multi_file(self):
        xlabs_test.go(['/home/bolster/Downloads/289-3678-2304-2504_xlabs.csv',
                       '/home/bolster/Downloads/289-3678-2304-2504_xlabs.csv'], calicut='800' )
