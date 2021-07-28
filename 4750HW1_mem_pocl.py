import numpy as np
import pyopencl as cl
import pyopencl.array as pyA
import matplotlib.pyplot as plt
class clModule:
    def __init__(self, a, b, length):
        """
        Attributes for instance of clModule
        Includes OpenCL context, command queue, kernel code
        and input variables.
        """
        
        # Get platform and device property
        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
        	if platform.name == NAME:
        		devs = platform.get_devices()       
        
        # Set up a command queue:
        self.ctx = cl.Context(devs)
        
        # Enable profiling property to time event execution
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

        # host variables
        # 
        #
        self.aH=np.float32(a)
        self.bH=np.float32(b)
        self.nH=np.uint32(length)

        # kernel - will not be provided for future assignments!
        kernel_code = """__kernel void sum(__global float* c, __global float* a, __global float* b, const unsigned int n)
                        {
                            unsigned int i = get_global_id(0);
                            if (i < n) {
                                c[i] = a[i] + b[i];
                                //printf("c[i] %f" , c[i]);
                                //printf(" ");
                                //Use printf to help debug kernel code
                            }
                        }""" 
        
        # Build kernel code
        self.prg = cl.Program(self.ctx, kernel_code).build()


    def deviceAdd(self):
        """
        Function to perform vector addition using the cl.array class    
        Returns:
            c       :   vector sum of arguments a and b
            time_   :   execution time for pocl function 
        """
                
        # Device memory allocation
        #
        #
        aD = pyA.to_device(self.queue, self.aH)
        bD = pyA.to_device(self.queue, self.bH)
        cD = pyA.empty_like(aD)
        evt=self.prg.sum(self.queue, aD.shape, None, cD.data, aD.data,
                    bD.data, self.nH)  # Enqueue the program for execution and store the result in c
        evt.wait()
        time_my = evt.profile.end - evt.profile.start
        cH=np.empty_like(self.aH)
        #cl.enqueue_copy(self.queue, cH, cD)
        cH=cD.get()
        #print(cD)
        # Invoke kernel program and time execution using event.profile()
        # Wait for program execution to complete
        # Remember: OpenCL event profiling returns times in nanoseconds. 
        
        # Fetch result from device to host

        return cH, time_my

    
    def bufferAdd(self):
        """
        Function to perform vector addition using the cl.Buffer class
        Returns:
            c               :    vector sum of arguments a and b
            end - start     :    execution time for pocl function 
        """
        # Create three buffers (plans for areas of memory on the device)
        mf = cl.mem_flags
        aD = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.aH)
        bD = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.bH)
        cD = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.aH.nbytes)
        evt=self.prg.sum(self.queue, self.aH.shape, None, cD, aD,
                    bD, self.nH)  # Enqueue the program for execution and store the result in c
        evt.wait()
        time_my = evt.profile.end - evt.profile.start
        cH=np.empty_like(self.aH)
        cl.enqueue_copy(self.queue, cH, cD)
        # Invoke kernel program, time execution
        # Fetch result
        
        return cH, time_my


if __name__ == "__main__":
    # Main code to create arrays, call all functions, calculate average
    # execution and plot
    # 1
    ini_length =100000
    a=np.ones((ini_length,1))*4
    b=np.ones((ini_length,1))*5
    myCL=clModule(a, b, ini_length)
    c1,mytime1=myCL.deviceAdd()
    print(c1, mytime1)
    #2
    c2,mytime2=myCL.bufferAdd()
    print(c2, mytime2)
    #3
    trl=10
    time1=[]
    time2=[]
    for L in range(1,21):
        temp1=[]
        temp2=[]
        for i in range(trl):
            length=ini_length*L
            a=np.random.rand(ini_length*L).astype(np.float32)
            b=np.random.rand(ini_length*L).astype(np.float32)
            myCL=clModule(a, b, length)
            _,mytime1=myCL.deviceAdd()
            _,mytime2=myCL.bufferAdd()
            temp1.append(mytime1)
            temp2.append(mytime2)
        time1.append(np.mean(temp1))
        time2.append(np.mean(temp2))
    plt.figure()
    plt.plot(np.arange(1,21), time1, label='addDevice')
    plt.plot(np.arange(1,21), time2, label='addBuffer')
    plt.xlabel('L')
    plt.ylabel('runningTime')
    plt.legend()
    plt.savefig('poclAdd.png')
    plt.show()
    
            