'''
Provides a GPU enhanced convolution function equivalent to 
scipy.signal.convolve for 2d and 3d arrays, and can apply one kernel
to many arrays in parallel.

@author: Scott Hellman
'''

from pyfft.cl import Plan
import numpy
import pyopencl as cl
import pyopencl.array as cl_array
import scipy.signal as signal

def getPowerOfTwo(value):
    return int(numpy.power(2,numpy.ceil(numpy.log2(value))))

class CachedQueueConvolver:
    """Stores device, context, and queue information. Discards Convolvers 
       after use to avoid hogging GPU memory and crashing drivers."""
    def __init__(self):
        device = cl.get_platforms()[0].get_devices(cl.device_type.GPU)[0]
        self.ctx = cl.Context(devices=[device])
        self.queue = cl.CommandQueue(self.ctx)
        
    def convolution(self,A,kernel,mode='valid'):
        convolver = Convolver(A[0].shape,kernel.shape,A.shape[0],self.ctx,self.queue)
        return convolver.convolution(A,kernel,mode)
    
class Convolver:
    """ Class that computes the necessary information to perform a
    convolution and provides the actual convolution function. Can handle
    2d or 3d convolutions. """
    
    def __init__(self,insize,kernelsize,batchsize,context,queue): 
        self.sizes = []
        for i in xrange(len(insize)):
            self.sizes.append(getPowerOfTwo(insize[i]+kernelsize[i]+1))
        self.sizes = tuple(self.sizes)
        
        self.ctx = context
        self.queue = queue
        self.plan = Plan(self.sizes,queue=self.queue)
        self.inarray = cl.array.zeros(self.queue,(batchsize,self.sizes[0],self.sizes[1],self.sizes[2]),numpy.complex64)
        self.kernel = cl.array.zeros(self.queue,(batchsize,self.sizes[0],self.sizes[1],self.sizes[2]),numpy.complex64)
        self.result_buffer = numpy.zeros(self.inarray.shape,numpy.complex64)
        
        self.kernel_center = []
        for i in xrange(len(kernelsize)):
            self.kernel_center.append(kernelsize[i]/2)
        self.kernel_center = tuple(self.kernel_center)
        
        self.halves = []
        for i in xrange(len(kernelsize)):
            self.halves.append(numpy.ceil(kernelsize[i]/2.0)) 
        self.halves = tuple(self.halves)
        
        self.padding_locations = []
        for i in xrange(len(self.sizes)):
            #without this if even kernels result in an incorrect edge in the result 
            if kernelsize[i]%2 == 0:
                self.padding_locations.append(-1*((insize[i]-self.sizes[i])/2))
                self.padding_locations.append(-1*((self.sizes[i]-insize[i])/2))
            else:
                self.padding_locations.append((self.sizes[i]-insize[i])/2)
                self.padding_locations.append((insize[i]-self.sizes[i])/2)
        self.padding_locations = tuple(self.padding_locations)
        
        self.valid_locations = []
        for i in xrange(len(self.sizes)):
            self.valid_locations.append(self.padding_locations[(i*2)] + self.halves[i] - 1)
            self.valid_locations.append(self.padding_locations[(i*2)] + self.halves[i] + insize[i] - kernelsize[i])
        self.valid_locations = tuple(self.valid_locations)
        
        self.full_locations = []
        for i in xrange(len(self.sizes)):
            offset = self.sizes[i] - (insize[i]+kernelsize[i]-1)
            self.full_locations.append(offset/2)
            self.full_locations.append(-offset/2)
        
        self.batch_size = batchsize
        
        
    def convolution(self,A,kernel,type='valid'):
        inarray = numpy.zeros((self.batch_size,self.sizes[0],self.sizes[1],self.sizes[2]),numpy.complex64)
        inarray[:,self.padding_locations[0]:self.padding_locations[1],
                     self.padding_locations[2]:self.padding_locations[3],
                     self.padding_locations[4]:self.padding_locations[5]] = A
        self.inarray = cl.array.to_device(self.queue,inarray)
        kernel_buffer = numpy.zeros((self.batch_size,self.sizes[0],self.sizes[1],self.sizes[2]),numpy.complex64)
        kernel_buffer[:,:self.halves[0],:self.halves[1],:self.halves[2]] = kernel[self.kernel_center[0]:,self.kernel_center[1]:,self.kernel_center[2]:]
        kernel_buffer[:,:self.halves[0],:self.halves[1],-self.kernel_center[2]:] = kernel[self.kernel_center[0]:,self.kernel_center[1]:,:self.kernel_center[2]]
        kernel_buffer[:,:self.halves[0],-self.kernel_center[1]:,:self.halves[2]] = kernel[self.kernel_center[0]:,:self.kernel_center[1],self.kernel_center[2]:]
        kernel_buffer[:,:self.halves[0],-self.kernel_center[1]:,-self.kernel_center[2]:] = kernel[self.kernel_center[0]:,:self.kernel_center[1],:self.kernel_center[2]]
        if kernel.shape[0] > 1:
            kernel_buffer[:,-self.kernel_center[0]:,:self.halves[1],:self.halves[2]] = kernel[:self.kernel_center[0],self.kernel_center[1]:,self.kernel_center[2]:]
            kernel_buffer[:,-self.kernel_center[0]:,:self.halves[1],-self.kernel_center[2]:] = kernel[:self.kernel_center[0],self.kernel_center[1]:,:self.kernel_center[2]]
            kernel_buffer[:,-self.kernel_center[0]:,-self.kernel_center[1]:,:self.halves[2]] = kernel[:self.kernel_center[0],:self.kernel_center[1],self.kernel_center[2]:]
            kernel_buffer[:,-self.kernel_center[0]:,-self.kernel_center[1]:,-self.kernel_center[2]:] = kernel[:self.kernel_center[0],:self.kernel_center[1],:self.kernel_center[2]]
        self.kernel = cl.array.to_device(self.queue,kernel_buffer)
        
        
        #fourier transform, pointwise multiply, then invert => convolution
        self.plan.execute(self.inarray.data,batch=self.batch_size)
        
        self.plan.execute(self.kernel.data,batch=self.batch_size)
        
        self.result_buffer = self.inarray * self.kernel
        self.plan.execute(self.result_buffer.data,inverse=True,batch=self.batch_size)
        result = self.result_buffer.get().astype(float)
                                 
        if type == 'same':
            return result[:,self.padding_locations[0]:self.padding_locations[1],self.padding_locations[2]:self.padding_locations[3],self.padding_locations[4]:self.padding_locations[5]]
        elif type == 'full':
            return result[:,self.full_locations[0]:self.full_locations[1],self.full_locations[2]:self.full_locations[3],self.full_locations[4]:self.full_locations[5]]
        elif type == 'valid':
            return result[:,self.valid_locations[0]:self.valid_locations[1],self.valid_locations[2]:self.valid_locations[3],self.valid_locations[4]:self.valid_locations[5]]
    
def example():
    convolver = CachedQueueConvolver()
    batched_data = numpy.random.rand(20,3,10,10)
    kernel = numpy.random.rand(1,3,3)
    result = convolver.convolution(batched_data,kernel,'valid')
    
    for i in xrange(result.shape[0]):
        truth = signal.fftconvolve(batched_data[i], kernel, 'valid') 
        print(numpy.any(numpy.abs(result[i] - truth) > 0.0001))

if __name__ == "__main__":
    example()
