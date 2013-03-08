import unittest
import numpy
import cachedConvolution
import scipy.signal as signal
 
class TestOpenCLConvolution(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(5)
        self.tolerance = 0.0001
        self.data_2d = numpy.random.rand(1,1,10,10)
        self.simple_kernel = numpy.array(([[[0,0,0,0,0],
                           [0,0,1,0,0],
                           [0,1,0,1,0],
                           [0,0,1,0,0],                           [0,0,0,0,0]]]))
        self.simple_kernel = numpy.random.rand(1,5,5)
        self.batch_data = numpy.random.rand(20,1,10,10)
        
        self.batch_3d = numpy.random.rand(20,4,6,7)
        self.kernel_3d = numpy.random.rand(2,3,3)
        self.large_data = numpy.random.rand(50,20,20,20)
        self.large_kernel = numpy.random.rand(5,10,10)
        self.uneven_data = numpy.random.rand(20,4,6,8)
        self.uneven_kernel = numpy.random.rand(2,3,4)
        self.convolver = cachedConvolution.CachedQueueConvolver()
        self.results = []
        
    def test_convolution_full(self):
        result = self.convolver.convolution(self.data_2d,self.simple_kernel,'full')
        truth = signal.fftconvolve(self.data_2d[0], self.simple_kernel, 'full') 
        
        result = result[0]
        
        difference = numpy.abs(result - truth)
        
        self.assertFalse(numpy.any(difference > self.tolerance))
        
    def test_convolution_valid(self):
        result = self.convolver.convolution(self.data_2d,self.simple_kernel,'valid')
        truth = signal.fftconvolve(self.data_2d[0], self.simple_kernel, 'valid') 
        
        result = result[0]
        
        difference = numpy.abs(result - truth)
        
        self.assertFalse(numpy.any(difference > self.tolerance))
        
    def test_convolution_same(self):
        result = self.convolver.convolution(self.data_2d,self.simple_kernel,'same')
        truth = signal.fftconvolve(self.data_2d[0], self.simple_kernel, 'same') 
        
        result = result[0]
        
        difference = numpy.abs(result - truth)
        
        self.assertFalse(numpy.any(difference > self.tolerance))
        
    def test_batch_convolution_full(self):
        result = self.convolver.convolution(self.batch_data,self.simple_kernel,'full')
        
        for i in xrange(self.data_2d.shape[0]):
            truth = signal.fftconvolve(self.batch_data[i], self.simple_kernel, 'full') 
            difference = numpy.abs(result[i] - truth)
            self.assertFalse(numpy.any(difference > self.tolerance))
            
    def test_batch_convolution_valid(self):
        result = self.convolver.convolution(self.batch_data,self.simple_kernel,'valid')
        
        for i in xrange(self.data_2d.shape[0]):
            truth = signal.fftconvolve(self.batch_data[i], self.simple_kernel, 'valid') 
            difference = numpy.abs(result[i] - truth)
            self.assertFalse(numpy.any(difference > self.tolerance))
            
    def test_batch_convolution_same(self):
        result = self.convolver.convolution(self.batch_data,self.simple_kernel,'same')
        
        for i in xrange(self.data_2d.shape[0]):
            truth = signal.fftconvolve(self.batch_data[i], self.simple_kernel, 'same') 
            difference = numpy.abs(result[i] - truth)
            self.assertFalse(numpy.any(difference > self.tolerance))
            
    def test_batch_convolution_3d(self):
        result = self.convolver.convolution(self.batch_3d,self.kernel_3d,'full')
        
        for i in xrange(self.data_2d.shape[0]):
            truth = signal.fftconvolve(self.batch_3d[i], self.kernel_3d, 'full') 
            difference = numpy.abs(result[i] - truth)
            self.assertFalse(numpy.any(difference > self.tolerance))
            
    def test_large_convolution(self):
        result = self.convolver.convolution(self.large_data,self.large_kernel,'full')
        
        for i in xrange(self.data_2d.shape[0]):
            truth = signal.fftconvolve(self.large_data[i], self.large_kernel, 'full') 
            difference = numpy.abs(result[i] - truth)
            self.assertFalse(numpy.any(difference > self.tolerance))
            
    def test_batch_convolution_uneven_data(self):
        result = self.convolver.convolution(self.uneven_data,self.kernel_3d,'full')
        
        for i in xrange(self.data_2d.shape[0]):
            truth = signal.fftconvolve(self.uneven_data[i], self.kernel_3d, 'full') 
            difference = numpy.abs(result[i] - truth)
            self.assertFalse(numpy.any(difference > self.tolerance))
            
    def test_batch_convolution_uneven_kernel(self):
        result = self.convolver.convolution(self.batch_3d,self.uneven_kernel,'full')
        
        for i in xrange(self.data_2d.shape[0]):
            truth = signal.fftconvolve(self.batch_3d[i], self.uneven_kernel, 'full') 
            difference = numpy.abs(result[i] - truth)
            self.assertFalse(numpy.any(difference > self.tolerance))
            
    def test_batch_convolution_uneven_both(self):
        result = self.convolver.convolution(self.uneven_data,self.uneven_kernel,'full')
        
        for i in xrange(self.data_2d.shape[0]):
            truth = signal.fftconvolve(self.uneven_data[i], self.uneven_kernel, 'full') 
            difference = numpy.abs(result[i] - truth)
            self.assertFalse(numpy.any(difference > self.tolerance))
        
            
if __name__ == '__main__':
    unittest.main()