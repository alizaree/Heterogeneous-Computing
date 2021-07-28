import matplotlib.pyplot as plt
import numpy as np
import pycuda as PC
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

from pycuda.compiler import SourceModule

class deviceAdd:
    def __init__(self, a, b, length):
        """
        Attributes for instance of deviceAdd module
        Includes kernel code and input variables.
        """
        # If you are using any helper function to make 
        # blocksize or gridsize calculations, you may define them
        # here as lambda functions. 
        self.gridN= lambda a, b: int(a // b + 1)
        
        cuda.init()
        my_device = cuda.Device(0)
        my_dev_attributes_tuples = my_device.get_attributes().items()
        mydev_attributes = {}

        for key, value in my_dev_attributes_tuples:
            mydev_attributes[str(key)] = value
            
        self.mydev_attributes=mydev_attributes
        
        # host variables
        #
        
        self.aH= a.astype(np.float32)
        self.bH= b.astype(np.float32)
        self.lenH=np.uint32(length)
        
        # define block and grid dimensions
        #

        # kernel code wrapper
        #
        
        self.mod = SourceModule("""
        __global__ void sum(int n, float *a, float *b, float *c)
        {
        
            int i = blockIdx.x * blockDim.x + threadIdx.x; 
            if (i < n) {
            c[i] =  a[i] + b[i];
            }
        }
        """)  

        
        # Compile the kernel code when an instance
        # of this class is made. This way it only
        # needs to be done once for the 4 functions
        # you will call from this class.
        
    
    def explicitAdd(self):
        """
        Function to perform on-device parallel vector addition
        by explicitly allocating device memory for host variables.
        Returns
            c                               :   addition result
            e_start.time_till(e_end)*(1e-3) :   execution time
        """

        # Note: Use cuda.Event to time your executions

        # Device memory allocation for input and output arrays
        #
        a_gpu = cuda.mem_alloc(self.aH.size * self.aH.dtype.itemsize)
        cuda.memcpy_htod(a_gpu, self.aH)
        
        b_gpu = cuda.mem_alloc( self.bH.size * self.bH.dtype.itemsize)
        cuda.memcpy_htod(b_gpu, self.bH)
        
        c_gpu= cuda.mem_alloc( self.aH.size * self.aH.dtype.itemsize)
        
        
        func = self.mod.get_function("sum")
        start = cuda.Event()
        end = cuda.Event()
        blockDim=(int(self.mydev_attributes['MAX_BLOCK_DIM_X']),1,1)
        
        gridDim=(self.gridN(self.lenH,blockDim[0]), 1, 1)
        
        start.record()
        func(self.lenH, a_gpu, b_gpu,  c_gpu,block=blockDim, grid = gridDim)#, block = blockDim, grid = gridDim)
        end.record() # wait for event to finish
        #end.synchronize()
        # time event execution in milliseconds
        # Copy data from host to device
        #
        cH = np.empty_like(self.aH)
        cuda.memcpy_dtoh(cH, c_gpu)
        # Call the kernel function from the compiled module
        #
        # Record execution time and call the kernel loaded to the device
        
        # Wait for the event to complete
        #
        # Copy result from device to the host
        #
        t = start.time_till(end)*(1e-3)
        return cH, t

    
    def implicitAdd(self):
        """
        Function to perform on-device parallel vector addition
        without explicitly allocating device memory for 
        host variables.
        Returns
            c                               :   addition result
            i_start.time_till(i_end)*(1e-3) :   execution time
        """
        # Call the kernel function from the compiled module
        #
        
        # Record execution time and call the kernel loaded to the device
        #
        
        # Wait for the event to complete
        #
        
        func = self.mod.get_function("sum")
        start = cuda.Event()
        end = cuda.Event()
        blockDim=(int(self.mydev_attributes['MAX_BLOCK_DIM_X']),1,1)
        
        gridDim=(self.gridN(self.lenH,blockDim[0]), 1, 1)
        cH = np.empty_like(self.aH)
        start.record()
        func(self.lenH, cuda.In(self.aH), cuda.In(self.bH),  cuda.Out(cH),block=blockDim, grid = gridDim)#, block = blockDim, grid = gridDim)
        end.record() # wait for event to finish
        #end.synchronize()
        # time event execution in milliseconds
        # Copy data from host to device
        #
        # Call the kernel function from the compiled module
        #
        # Record execution time and call the kernel loaded to the device
        
        # Wait for the event to complete
        #
        # Copy result from device to the host
        #
        end.synchronize()
        t = start.time_till(end)*(1e-3)
        
        return cH, t


    def gpuarrayAdd_np(self):
        """
        Function to perform on-device parallel vector addition
        without explicitly allocating device memory for 
        host variables and WITHOUT calling the kernel. The operation
        is defined using numpy-like syntax. 
        Returns
            c                               :   addition result
            i_start.time_till(i_end)*(1e-3) :   execution time
        """
        # Allocate device memory using gpuarray class        
        #

        # Record execution time and execute operation with numpy syntax
        #

        # Wait for the event to complete
        #
        
        # Fetch result from device to host
        start = cuda.Event()
        end   = cuda.Event()
        cH = np.empty_like(self.aH)

        aD = gpuarray.to_gpu(self.aH)
        bD = gpuarray.to_gpu(self.bH)

        start.record()
        cD = (aD + bD)
        end.record() 
        end.synchronize()
        t = start.time_till(end) * 1e-3
        cH = cD.get()

        return cH, t
        
    
    def gpuarrayAdd(self):
        """
        Function to perform on-device parallel vector addition
        without explicitly allocating device memory for 
        host variables and WITHOUT calling the kernel. The operation
        is defined using numpy-like syntax. 
        Returns
            c                               :   addition result
            i_start.time_till(i_end)*(1e-3) :   execution time
        """
        # Allocate device memory using gpuarray class        
        #

        # Record execution time and execute operation with numpy syntax
        #
        # Wait for the event to complete
        #
    
        # Fetch result from device to host
        #
        start = cuda.Event()
        end   = cuda.Event()
        aD = gpuarray.to_gpu(self.aH)
        bD = gpuarray.to_gpu(self.bH)
        cD= gpuarray.zeros_like(aD)

        # --- Define a reference to the __global__ function and call it
        deviceAdd = self.mod.get_function("sum")
        blockDim=(int(self.mydev_attributes['MAX_BLOCK_DIM_X']),1,1)
        
        gridDim=(self.gridN(self.lenH,blockDim[0]), 1, 1)
        cH = np.empty_like(self.aH)
        start.record()
        deviceAdd(self.lenH, aD, bD,  cD, block = blockDim, grid = gridDim)
        end.record() 
        end.synchronize()
        t = start.time_till(end) * 1e-3
        #cuda.memcpy_dtoh(cH, cD)
        cH = cD.get()
    
        return cD, t

if __name__ == "__main__":
    # Main code to create arrays, call all functions, calculate average
    # execution and plot
    ini_length =100000
    a=np.ones((ini_length,1))*7
    b=np.ones((ini_length,1))*6
    myCU=deviceAdd(a, b, ini_length)
    c1,mytime1=myCU.explicitAdd()
    print('explicit',c1, mytime1)
    c2,mytime2=myCU.implicitAdd()
    print('implicit',c2, mytime2)
    c3,mytime3=myCU.gpuarrayAdd_np()
    print('gpuarray_np',c3, mytime3)
    c4,mytime4=myCU.gpuarrayAdd()
    print('gpuarray',c4, mytime4)
    time1=[]
    time2=[]
    time3=[]
    time4=[]
    itr=10
    for L in range(1,21):
        temp1=[]
        temp2=[]
        temp3=[]
        temp4=[]
        for i in range(itr):
            a=np.random.rand(ini_length*L)
            b=np.random.rand(ini_length*L)
            length=L*ini_length
            myCU=deviceAdd(a, b, ini_length)
            _,mytime1=myCU.explicitAdd()
            _,mytime2=myCU.implicitAdd()
            _,mytime3=myCU.gpuarrayAdd_np()
            _,mytime4=myCU.gpuarrayAdd()
            temp1.append(mytime1)
            temp2.append(mytime2)
            temp3.append(mytime3)
            temp4.append(mytime4)
        time1.append(np.mean(temp1))
        time2.append(np.mean(temp2))
        time3.append(np.mean(temp3))
        time4.append(np.mean(temp4))
    plt.figure()
    plt.plot(np.arange(1,21), time1, label='explicit')
    plt.plot(np.arange(1,21), time2, label='implicit')
    plt.plot(np.arange(1,21), time3, label='gpuarray_np')
    plt.plot(np.arange(1,21), time4, label='gpuarray')
    plt.xlabel('L')
    
    plt.ylabel('runningTime')
    plt.legend()
    plt.savefig('cudaAdd.png')
    plt.show()