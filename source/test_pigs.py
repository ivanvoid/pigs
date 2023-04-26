import unittest
import logging
import _pigs
import numpy.testing as npt
# other libs
import sys
from PIL import Image
import numpy as np
from hilbert import decode, encode   

def run_kernel(rgb1, rgb2, pos1, pos2, N, beta):
    R1, G1, B1 = rgb1[0],rgb1[1],rgb1[2]
    R2, G2, B2 = rgb2[0],rgb2[1],rgb2[2]
    x1, y1 = pos1[0]/(2*N), pos1[1]/(2*N)
    x2, y2 = pos2[0]/(2*N), pos2[1]/(2*N)
    
    return pigs.rgb_pos_kernel(R1,G1,B1,R2,G2,B2,x1,y1,x2,y2,beta)
    
class PigsTest(unittest.TestCase):      
    def test_rgb_pos_kernel(self):
        self.assertEqual(run_kernel((1,1,1), (1,1,1), (5,5), (5,5), 10, 1), 1.0)
        self.assertEqual(run_kernel((0,0,0), (0,0,0), (0,0), (0,0), 10, 1), 1.0)
        self.assertEqual(run_kernel((0,0,0), (5,5,5), (0,0), (5,5), 10, 1), 0.00041210654649108955)

    def test_rgb_pos_kernel_vectorized(self):
        N = 10
        R1 = [1,2,0]; R1 = np.array(R1)
        G1 = [1,2,0]; G1 = np.array(G1)
        B1 = [1,2,0]; B1 = np.array(B1)
        R2 = [1,2,2]; R2 = np.array(R2)
        G2 = [1,2,2]; G2 = np.array(G2)
        B2 = [1,2,2]; B2 = np.array(B2)
        x1 = [0,1,3]; x1 = np.array(x1) / (2*N)
        y1 = [0,1,3]; y1 = np.array(y1) / (2*N)
        x2 = [0,1,4]; x2 = np.array(x2) / (2*N)
        y2 = [0,1,4]; y2 = np.array(y2) / (2*N)
        beta = 1
        
        true_results = np.array(
            [1, 1, 0.037220465006775054])
        
        out = _pigs.rgb_pos_kernel(R1,G1,B1,R2,G2,B2,x1,y1,x2,y2,beta)
        
        npt.assert_almost_equal(out, true_results)
#         self.assertListEqual(list(out), true_results, msg='DEBUG:{}'.format(type(out)))

    def test_flower_image_gram(self):
        m = 3
        # load image
        image = Image.open('../data/flower.jpg')
        image = image.resize((2**m,2**m))
        ndarray_image = np.array(image).astype(int)
        
        # flatten image
        locs = decode(np.arange(2**m*2**m), 2, m).astype(np.int)
        flatted_image = ndarray_image[locs[:,0], locs[:,1]]
        
        N = flatted_image.shape[0]
        
        R = flatted_image[:,0]
        G = flatted_image[:,1]
        B = flatted_image[:,2]
        x = locs[:,0] / (2*N)
        y = locs[:,1] / (2*N)
        
        gram = np.zeros((N**2 - N)//2) # Upper triangle without diagonal
        beta = 1
        k = 0
        for i in range(1, N-1):
            Ni = N-i

            out = pigs.rgb_pos_kernel(R[i].repeat(Ni),G[i].repeat(Ni),B[i].repeat(Ni),
                                       R[i:],G[i:],B[i:],
                                       x[i].repeat(Ni),y[i].repeat(Ni),
                                       x[i:],y[i:],
                                       beta)
            gram[k:N-i+k] = out
            k += N-i
            
#             self.assertEqual(0, 1, msg='DEBUG: {}{}'.format(k,N-i+k))
            
