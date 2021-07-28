import pyopencl as cl
import pyopencl.array
import matplotlib.pyplot as plt
import numpy as np
import time as tm
from PIL import Image
import math
import pickle

#Serial Kernel
def neuron_serial_kernel(I, vprev2, vprev, v, spikes, weights, m, h, n, NeuronsPerLayer, NoLayers):
    #Prevent python overflow of exponential
    def exp(imp):
        try:
            ans = math.exp(imp)
        except OverflowError:
            ans = float('inf')
        return ans
    
    #Constants
    Vref = -10
    Cm = 1 
    gNa = 120
    gK = 36
    gL = 0.3
    ENa = 115
    EK = -12
    EL = 10.613
    dt = 1e-2

    for Row in range(NeuronsPerLayer):
        for Col in range(NoLayers):

            am=bm=ah=bh=an=bn=dv_dt=dn_dt=dm_dt=dh_dt=IL=IK=INa=0
            idx=Row*NoLayers + Col # [Row][Col]
            Iidx=Row
            #define the states
            ml = m[Row][Col]
            nl = n[Row][Col]
            hl = h[Row][Col]
            vprevlocal = vprev[Row][Col]
            Il = 0;

            if (Col > 0):
                for i in range(NeuronsPerLayer):
                    widx=i*NeuronsPerLayer*NoLayers + (Col)*NeuronsPerLayer+Row
                    sidx=i*NoLayers + Col-1
                    Il += weights[i][Col*NeuronsPerLayer+Row] *spikes[i][Col-1]
            else:
                Il = I[Iidx]

            #Compute the states
            INa = gNa * ml*ml*ml * hl * (ENa - vprevlocal)
            IK = gK * nl*nl*nl*nl * (EK - vprevlocal)
            IL = gL * (EL - vprevlocal)

            am = (25-vprevlocal)/(10*(exp((25-vprevlocal)/10)-1))
            bm = 4*exp(-vprevlocal/18)

            ah = 0.07*exp(-vprevlocal/20)
            bh = 1/(exp((30-vprevlocal)/10)+1)

            an = (10-vprevlocal)/(100*(exp((10-vprevlocal)/10)-1))
            bn = 0.125*exp(-vprevlocal/80)
            
            #Compute the state derivatives
            dv_dt = ( INa + IK + IL + Il ) / Cm
            dm_dt = am * (1-ml) - ml * bm
            dh_dt = ah * (1-hl) - hl * bh
            dn_dt = an * (1-nl) - nl * bn
            
            #Compute Next State
            v[Row][Col] = vprevlocal + dv_dt * dt
            m[Row][Col] = ml + dm_dt * dt
            h[Row][Col] = hl + dh_dt * dt
            n[Row][Col] = nl + dn_dt * dt
            
            #Spike Detection
            if (vprev[Row][Col] >= 10 and vprev[Row][Col] > v[Row][Col] and vprev[Row][Col] > vprev2[Row][Col]):
                spikes[Row][Col] = 1
            else:
                spikes[Row][Col] = 0
            
            vprev2[Row][Col] = vprevlocal
            vprev[Row][Col] = v[Row][Col]

    return vprev2, vprev, v, spikes, weights, m, h, n


class SNN:
    def __init__(self, NeuronsPerLayer, NoLayers):
        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
            if platform.name == NAME:
                devs = platform.get_devices()
        self.ctx = cl.Context(devs)
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

        # host variables
        self.x = NeuronsPerLayer
        self.y = NoLayers
        
        # kernel code for SNN       
        kernel_code = """__kernel void neuron(__global float* I, __global float* vprev2, __global float* vprev, __global float* v, __global bool* spikes, __global float* weights, __global float* m, __global float* h, __global float* n, const unsigned int NeuronsPerLayer, const unsigned int NoLayers){
                                    unsigned int Row = get_global_id(0);
                                    unsigned int Col = get_global_id(1);
                                    unsigned int idx=Row*NoLayers + Col;
                                    unsigned int Iidx=Row;
                                    
                                    const int Vref = -10;
                                    const  int Cm = 1;   // [uF] membrane capacitance
                                    const  int gNa = 120;
                                    const  int gK = 36;
                                    const  float gL = 0.3;
                                    const  float ENa = 115;
                                    const  float EK = -12;
                                    const  float EL = 10.613;
                                    const  float dt = 1e-2;
                                    float am=0, bm=0, ah=0, bh=0, an=0, bn=0, dv_dt=0, dn_dt=0, dm_dt=0, dh_dt=0, IL=0, IK=0, INa=0;

                                if ((Row < NeuronsPerLayer) && (Col < NoLayers)) {
                                    float ml = m[idx];
                                    float nl = n[idx];
                                    float hl = h[idx];
                                    float vprevlocal = vprev[idx];
                                    float Il = 0;
                                    
                                    // Accumulate previous neurons
                                    if (Col > 0){
                                        for (int i=0; i < NeuronsPerLayer; i++){
                                            unsigned int widx=i*NeuronsPerLayer*NoLayers + (Col)*NeuronsPerLayer+Row;
                                            unsigned int sidx=i*NoLayers + Col-1;
                                            Il += weights[widx] *spikes[sidx];
                                        }
                                    }
                                    //Input Layer current
                                    else
                                        Il = I[Iidx];
                                    

                                    // ionic currents
                                    INa = gNa * ml*ml*ml * hl * (ENa - vprevlocal);
                                    IK = gK * nl*nl*nl*nl * (EK - vprevlocal);
                                    IL = gL * (EL - vprevlocal);

                                    am = (25-vprevlocal)/(10*(exp((25-vprevlocal)/10)-1));
                                    bm = 4*exp(-vprevlocal/18);

                                    ah = 0.07*exp(-vprevlocal/20);
                                    bh = 1/(exp((30-vprevlocal)/10)+1);

                                    an = (10-vprevlocal)/(100*(exp((10-vprevlocal)/10)-1));
                                    bn = 0.125*exp(-vprevlocal/80);

                                    // derivatives
                                    dv_dt = ( INa + IK + IL + Il ) / Cm;
                                    dm_dt = am * (1-ml) - ml * bm;
                                    dh_dt = ah * (1-hl) - hl * bh;
                                    dn_dt = an * (1-nl) - nl * bn;

                                    // calculate next step
                                    v[idx] = vprevlocal + dv_dt * dt;
                                    m[idx] = ml + dm_dt * dt;
                                    h[idx] = hl + dh_dt * dt;
                                    n[idx] = nl + dn_dt * dt;
                                    
                                    barrier(CLK_GLOBAL_MEM_FENCE);
                                    
                                    //Spike detection
                                    if (vprev[idx] >= 10 && vprev[idx] > v[idx] && vprev[idx] > vprev2[idx])
                                        spikes[idx] = 1;
                                    else
                                        spikes[idx] = 0;
                                    
                                    // updating previous voltages
                                    vprev2[idx] = vprevlocal;
                                    vprev[idx] = v[idx];
                                }
                            }""" 
        
        kernel_code_optimized = """__kernel void neuron_opt(__global float* I, __global float* vprev2, __global float* vprev, __global float* v, __global bool* spikes, __global float* weights, __global float* m, __global float* h, __global float* n, const unsigned int NeuronsPerLayer, const unsigned int NoLayers){
                                    #define TS 32
                                    
                                    unsigned int Row = get_global_id(0);
                                    unsigned int Col = get_global_id(1);
                                    const int globalRow = TS*get_group_id(0) + Row; 
                                    const int globalCol = TS*get_group_id(1) + Col;
                                    
                                    unsigned int idx=globalRow*NoLayers + globalCol;
                                    unsigned int Iidx=globalRow;
                                    
                                    // Local memory to fit a tile of TS*TS elements of A and B
                                    __local bool spike_sub[TS][TS];
                                    
                                    const int Vref = -10;
                                    const  int Cm = 1;   // [uF] membrane capacitance
                                    const  int gNa = 120;
                                    const  int gK = 36;
                                    const  float gL = 0.3;
                                    const  float ENa = 115;
                                    const  float EK = -12;
                                    const  float EL = 10.613;
                                    const  float dt = 1e-2;
                                    float am=0, bm=0, ah=0, bh=0, an=0, bn=0, dv_dt=0, dn_dt=0, dm_dt=0, dh_dt=0, IL=0, IK=0, INa=0;

                                if ((globalRow < NeuronsPerLayer) && (globalCol < NoLayers)) {
                                    float ml = m[idx];
                                    float nl = n[idx];
                                    float hl = h[idx];
                                    float vprevlocal = vprev[idx];
                                    float Il = 0;
                                    
                                    //Tiling
                                    for (int t=0; t<NeuronsPerLayer/TS; t++){
                                        // Accumulate previous neurons
                                        if (globalCol > 0){
                                                spike_sub[Row][Col] = spikes[idx-1];
                                                barrier(CLK_LOCAL_MEM_FENCE);
                                                for (int i=0; i < TS ; i++){
                                                    unsigned int widx=( (globalRow/TS) + i)*NeuronsPerLayer*NoLayers + (globalCol)*NeuronsPerLayer+globalRow;
                                                    Il += weights[widx] *spike_sub[i][Col];
                                            }
                                        }
                                        //Input Layer current
                                        else{
                                            Il = I[Iidx];
                                        }
                                        barrier(CLK_LOCAL_MEM_FENCE);
                                        }
                                    

                                    // ionic currents
                                    INa = gNa * ml*ml*ml * hl * (ENa - vprevlocal);
                                    IK = gK * nl*nl*nl*nl * (EK - vprevlocal);
                                    IL = gL * (EL - vprevlocal);

                                    am = (25-vprevlocal)/(10*(exp((25-vprevlocal)/10)-1));
                                    bm = 4*exp(-vprevlocal/18);

                                    ah = 0.07*exp(-vprevlocal/20);
                                    bh = 1/(exp((30-vprevlocal)/10)+1);

                                    an = (10-vprevlocal)/(100*(exp((10-vprevlocal)/10)-1));
                                    bn = 0.125*exp(-vprevlocal/80);

                                    // derivatives
                                    dv_dt = ( INa + IK + IL + Il ) / Cm;
                                    dm_dt = am * (1-ml) - ml * bm;
                                    dh_dt = ah * (1-hl) - hl * bh;
                                    dn_dt = an * (1-nl) - nl * bn;

                                    // calculate next step
                                    v[idx] = vprevlocal + dv_dt * dt;
                                    m[idx] = ml + dm_dt * dt;
                                    h[idx] = hl + dh_dt * dt;
                                    n[idx] = nl + dn_dt * dt;
                                    
                                    barrier(CLK_GLOBAL_MEM_FENCE);
                                    
                                    //Spike Detection
                                    if (vprev[idx] >= 10 && vprev[idx] > v[idx] && vprev[idx] > vprev2[idx])
                                        spikes[idx] = 1;
                                    else
                                        spikes[idx] = 0;
                                    
                                    // updating previous voltages
                                    vprev2[idx] = vprevlocal;
                                    vprev[idx] = v[idx];
                                }
                            }""" 
        kernel_code_unoptimized = """__kernel void neuron_unopt(__global float* I, __global float* vprev2, __global float* vprev, __global float* v, __global int* spikes, __global float* weights, __global float* m, __global float* h, __global float* n, const unsigned int NeuronsPerLayer, const unsigned int NoLayers){
                                    unsigned int Row = get_global_id(0);
                                    unsigned int Col = get_global_id(1);
                                    unsigned int idx=Row*NoLayers + Col;
                                    unsigned int Iidx=Row;
                                    
                                    const int Vref = -10;
                                    const  int Cm = 1;   // [uF] membrane capacitance
                                    const  int gNa = 120;
                                    const  int gK = 36;
                                    const  float gL = 0.3;
                                    const  float ENa = 115;
                                    const  float EK = -12;
                                    const  float EL = 10.613;
                                    const  float dt = 1e-2;
                                    float am=0, bm=0, ah=0, bh=0, an=0, bn=0, dv_dt=0, dn_dt=0, dm_dt=0, dh_dt=0, IL=0, IK=0, INa=0;

                                if ((Row < NeuronsPerLayer) && (Col < NoLayers)) {
                                    float vprevlocal = vprev[Row*NoLayers + Col];
                                    float Il = 0;
                                    
                                    // Accumulate previous neurons
                                    if (Col > 0){
                                        for (int i=0; i < NeuronsPerLayer; i++){
                                            unsigned int widx=i*NeuronsPerLayer*NoLayers + (Col)*NeuronsPerLayer+Row;
                                            unsigned int sidx=i*NoLayers + Col-1;
                                            Il += weights[widx] *spikes[sidx];
                                        }
                                    }
                                    //Input Layer current
                                    else
                                        Il = I[Iidx];
                                    
                                    
                                    // ionic currents
                                    INa = gNa * m[Row*NoLayers + Col]*m[Row*NoLayers + Col]*m[Row*NoLayers + Col] * h[Row*NoLayers + Col] * (ENa - vprevlocal);
                                    IK = gK * n[Row*NoLayers + Col]*n[Row*NoLayers + Col]*n[Row*NoLayers + Col]*n[Row*NoLayers + Col] * (EK - vprev[Row*NoLayers + Col]);
                                    IL = gL * (EL - vprev[Row*NoLayers + Col]);

                                    am = (25-vprev[Row*NoLayers + Col])/(10*(exp((25-vprev[Row*NoLayers + Col])/10)-1));
                                    bm = 4*exp(-vprev[Row*NoLayers + Col]/18);

                                    ah = 0.07*exp(-vprev[Row*NoLayers + Col]/20);
                                    bh = 1/(exp((30-vprev[Row*NoLayers + Col])/10)+1);

                                    an = (10-vprev[Row*NoLayers + Col])/(100*(exp((10-vprev[Row*NoLayers + Col])/10)-1));
                                    bn = 0.125*exp(-vprev[Row*NoLayers + Col]/80);

                                    // derivatives
                                    dv_dt = ( INa + IK + IL + Il ) / Cm;
                                    dm_dt = am * (1-m[Row*NoLayers + Col]) - m[Row*NoLayers + Col] * bm;
                                    dh_dt = ah * (1-h[Row*NoLayers + Col]) - h[Row*NoLayers + Col] * bh;
                                    dn_dt = an * (1-n[Row*NoLayers + Col]) - n[Row*NoLayers + Col] * bn;

                                    // calculate next step
                                    v[Row*NoLayers + Col] = vprev[Row*NoLayers + Col] + dv_dt * dt;
                                    m[Row*NoLayers + Col] = m[Row*NoLayers + Col] + dm_dt * dt;
                                    h[Row*NoLayers + Col] = h[Row*NoLayers + Col] + dh_dt * dt;
                                    n[Row*NoLayers + Col] = n[Row*NoLayers + Col] + dn_dt * dt;
                                    
                                    barrier(CLK_GLOBAL_MEM_FENCE);
                                    
                                    //Spike detection
                                    if (vprev[Row*NoLayers + Col] >= 10 && vprev[Row*NoLayers + Col] > v[Row*NoLayers + Col] && vprev[Row*NoLayers + Col] > vprev2[Row*NoLayers + Col])
                                        spikes[Row*NoLayers + Col] = 1;
                                    else
                                        spikes[Row*NoLayers + Col] = 0;
                                    
                                    // updating previous voltages
                                    vprev2[Row*NoLayers + Col] = vprev[Row*NoLayers + Col];
                                    vprev[Row*NoLayers + Col] = v[Row*NoLayers + Col];
                                }
                            }"""  

        # build kernel
        self.prg = cl.Program(self.ctx, kernel_code).build()
        self.prgopt = cl.Program(self.ctx, kernel_code_optimized).build()
        self.prgunopt = cl.Program(self.ctx, kernel_code_unoptimized).build()        


    def neuron_parallel(self, time):
        np.random.seed(55)

        # device memory allocation
        vprev = cl.array.to_device(self.queue, np.ones((self.x, self.y)).astype(np.float32)*-10)
        vprev2 = cl.array.to_device(self.queue,np.ones((self.x, self.y)).astype(np.float32)*-10)
        v = cl.array.empty(self.queue, shape=(self.x, self.y), dtype=np.float32)
        spikes = cl.array.to_device(self.queue,np.zeros((self.x, self.y)).astype(np.bool))
        wh = np.random.normal(1800, 2000, size=(self.x, self.x*self.y)).astype(np.float32)
        weights = cl.array.to_device(self.queue, wh)
        
        m_dev = cl.array.to_device(self.queue,np.zeros((self.x, self.y)))
        h_dev = cl.array.to_device(self.queue,np.ones((self.x, self.y)))
        n_dev = cl.array.to_device(self.queue,np.zeros((self.x, self.y)))
        
        timing = []
        voltage = []
        Spikes = []
        Current = []

        temp = np.random.uniform(size=self.x)
        Ih = (-10*(temp < 0.4) + 30*(temp >= 0.4))
        I = cl.array.to_device(self.queue, Ih)
        
        counter=0
        #Loop the time steps
        for t in range(time):
            #Pulse input
            counter+=1
            if counter>=1000 and  counter<=2000 :
                Ih[:int(self.x/2)] = 30
                Ih[int(self.x/2):] = -10
            if counter>2000:
                counter=0
                Ih[:int(self.x/2)] = -10
                Ih[int(self.x/2):] = 30
            #print(Ih)
            
            #Call the kernel
            evt = self.prg.neuron(self.queue, v.shape, None, I.data, vprev2.data, vprev.data, v.data, spikes.data, weights.data, m_dev.data, h_dev.data, n_dev.data, np.uint32(self.x), np.uint32(self.y))        
            evt.wait()
            
            timing.append(evt.profile.end - evt.profile.start)
            voltage.append(v.get())
            Spikes.append(spikes.get())
            Current.append(Ih)

        return voltage, Spikes, np.average(timing)*1e-6, Current  
    
    def unopt_parallel(self, time):
        np.random.seed(55)

        # device memory allocation
        vprev = cl.array.to_device(self.queue, np.ones((self.x, self.y)).astype(np.float32)*-10)
        vprev2 = cl.array.to_device(self.queue,np.ones((self.x, self.y)).astype(np.float32)*-10)
        v = cl.array.empty(self.queue, shape=(self.x, self.y), dtype=np.float32)
        spikes = cl.array.to_device(self.queue,np.zeros((self.x, self.y)).astype(np.int32))
        wh = np.random.normal(1800, 2000, size=(self.x, self.x*self.y)).astype(np.float32)
        weights = cl.array.to_device(self.queue, wh)

        m_dev = cl.array.to_device(self.queue,np.zeros((self.x, self.y)))
        h_dev = cl.array.to_device(self.queue,np.ones((self.x, self.y)))
        n_dev = cl.array.to_device(self.queue,np.zeros((self.x, self.y)))

        timing = []
        voltage = []
        Spikes = []
        Current = []
        
        temp = np.random.uniform(size=self.x)
        Ih = (-10*(temp < 0.4) + 30*(temp >= 0.4))
        I = cl.array.to_device(self.queue, Ih)

        counter=0
        for t in range(time):
            #Pulse Input
            counter+=1
            if counter>=1000 and  counter<=2000 :
                Ih[:int(self.x/2)] = 30
                Ih[int(self.x/2):] = -10
            if counter>2000:
                counter=0
                Ih[:int(self.x/2)] = -10
                Ih[int(self.x/2):] = 30
            
            #Call the Kernel
            evt = self.prgunopt.neuron_unopt(self.queue, v.shape, None, I.data, vprev2.data, vprev.data, v.data, spikes.data, weights.data, m_dev.data, h_dev.data, n_dev.data, np.uint32(self.x), np.uint32(self.y))        
            evt.wait()

            timing.append(evt.profile.end - evt.profile.start)

        return np.average(timing)*1e-6
    
    def opt_parallel(self, time, TS):
        np.random.seed(55)

        # device memory allocation
        vprev = cl.array.to_device(self.queue, np.ones((self.x, self.y)).astype(np.float32)*-10)
        vprev2 = cl.array.to_device(self.queue,np.ones((self.x, self.y)).astype(np.float32)*-10)
        v = cl.array.empty(self.queue, shape=(self.x, self.y), dtype=np.float32)
        spikes = cl.array.to_device(self.queue,np.zeros((self.x, self.y)).astype(np.bool))
        wh = np.random.normal(1800, 2000, size=(self.x, self.x*self.y)).astype(np.float32)
        weights = cl.array.to_device(self.queue, wh)
        
        m_dev = cl.array.to_device(self.queue,np.zeros((self.x, self.y)))
        h_dev = cl.array.to_device(self.queue,np.ones((self.x, self.y)))
        n_dev = cl.array.to_device(self.queue,np.zeros((self.x, self.y)))
        
        timing = []
        voltage = []
        Spikes = []
        Current = []

        temp = np.random.uniform(size=self.x)
        Ih = (-10*(temp < 0.4) + 30*(temp >= 0.4))
        I = cl.array.to_device(self.queue, Ih)
        
        
        counter=0
        for t in range(time):
            #Pulse Input
            counter+=1
            if counter>=100 and  counter<=200 :
                Ih[:int(self.x/2)] = 30
                Ih[int(self.x/2):] = -10
            if counter>200:
                counter=0
                Ih[:int(self.x/2)] = -10
                Ih[int(self.x/2):] = 30
                
            I = cl.array.to_device(self.queue, Ih)
            
            evt = self.prgopt.neuron_opt(self.queue, (int(v.shape[0]/TS),int(v.shape[1]/TS)), None , I.data, vprev2.data, vprev.data, v.data, spikes.data, weights.data, m_dev.data, h_dev.data, n_dev.data, np.uint32(self.x), np.uint32(self.y))        
            evt.wait()
            timing.append(evt.profile.end - evt.profile.start)

        return np.average(timing)*1e-6
    
    def neuron_serial(self, time):
        np.random.seed(3)
        vprev = np.ones((self.x, self.y)).astype(np.float32)*-10
        vprev2 = np.ones((self.x, self.y)).astype(np.float32)*-10
        v = np.ones((self.x, self.y)).astype(np.float32)*-10
        spikes = np.zeros((self.x, self.y)).astype(np.bool)
        weights = np.random.normal(1800, 2000, size=(self.x, self.x*self.y)).astype(np.float32)
        
        m = np.zeros((self.x, self.y))
        h = np.ones((self.x, self.y))
        n = np.zeros((self.x, self.y))
        temp = np.random.uniform(size=self.x)
        I = (-10*(temp < 0.4) + 30*(temp >= 0.4))
        
        timing = []
        
        for t in range(time):
            start = tm.time()
            vprev2, vprev, v, spikes, weights, m, h, n = neuron_serial_kernel(I, vprev2, vprev, v, spikes, weights, m, h, n, self.x, self.y)
            
            timing.append(tm.time()-start)

        return np.average(timing)*1e3

if __name__ == "__main__":
    Lrange = 2
    V=[]
    S=[]
    T=[]
    Ts=[]
    Current=[]
    Topt = []
    Tunopt=[]
    
    TS = 32
    N_N=TS*2
    N_L=TS*4
    
    for L in range(1, Lrange):
        snn_obj = SNN(N_N*L, N_L*L) #(No_neurons, Nlayers)
        voltages, spikes, t, Ih = snn_obj.neuron_parallel(5)
        to = snn_obj.opt_parallel(5, TS)
        ts = snn_obj.neuron_serial(1)
        tuo = snn_obj.opt_parallel(5, TS)
        V.append(voltages)
        S.append(spikes)
        T.append(t)
        Topt.append(to)
        Ts.append(ts)
        Tunopt.append(tuo)
        
        #Used to store the Voltages if Google Cloud gets disconnected
        '''
    with open("V.txt", "wb") as fp:   #Pickling
        pickle.dump(V, fp)    
    with open("S.txt", "wb") as fp:   #Pickling
        pickle.dump(S, fp)  
    with open("T.txt", "wb") as fp:   #Pickling
        pickle.dump(T, fp)  
    with open("Topt.txt", "wb") as fp:   #Pickling
        pickle.dump(Topt, fp)  
    with open("Ts.txt", "wb") as fp:   #Pickling
        pickle.dump(Ts, fp)  
    with open("Tunopt.txt", "wb") as fp:   #Pickling
        pickle.dump(Tunopt, fp)
        '''
        #Used to load the Voltages from text files
    '''
    #Tiling Size
    TS=32
    N_Ng=TS*2
    N_Lg=TS*4
    with open("S.txt", "rb") as fp:   #Pickling
        S=pickle.load(fp)  
    with open("T.txt", "rb") as fp:   #Pickling
        T=pickle.load(fp)  
    with open("Topt.txt", "rb") as fp:   #Pickling
        Topt=pickle.load(fp)  
    with open("Ts.txt", "rb") as fp:   #Pickling
        Ts=pickle.load(fp)  
    with open("Tunopt.txt", "rb") as fp:   #Pickling
        Tunopt=pickle.load(fp)
        '''

        fig = plt.figure(figsize=(35,35))
        pcaL= []
        LL= 0
        N_N= 5
        N_L= 2
        N_N *= LL+1
        N_L *= LL+1
        
        #Used to plot the neuronal voltages
        '''
        for neuro in np.arange(N_N):
            for layr in range(N_L):
                vv=[idd[neuro,layr] for idd in V[LL]]
                sp=[idd[neuro,layr]*50 for idd in S[LL]]
                #Red_act=pcaL[layr].transform(X)
                #axs[SNR, layr].axes(projection='3d')
                ax = fig.add_subplot(N_N,N_L ,neuro*(N_L)+layr+1 )
                ax.plot(np.arange(len(vv))/100, vv, label="Voltage")
                ax.plot(np.arange(len(sp))/100, sp,label="Spikes")
                plt.legend()
                #ax.text(Red_act[-1,0], Red_act[-1,1], Red_act[-1,2], "end", color='red')
                plt.title("Neuron: "+str(neuro)+" at Layer: "+str(layr))

                plt.ylabel("Voltage in mV")
                if neuro==N_N-1:
                    plt.xlabel("Time in milliseconds")

        plt.show()
        plt.savefig('plots/project_neurons_pulse.png')
        '''
        #PLot the execution time
        plt.figure()
        plt.plot(  ((np.arange(1,len(T)+1))**2) *N_Ng*N_Lg, T)
        plt.plot(  ((np.arange(1,len(T)+1))**2) *N_Ng*N_Lg, Topt)
        plt.plot(  ((np.arange(1,len(T)+1))**2) *N_Ng*N_Lg, Ts)
        plt.plot(  ((np.arange(1,len(T)+1))**2) *N_Ng*N_Lg, Tunopt)
        plt.legend(['Naive Parallel', 'Tiled and Optimized', 'Serial', 'Tiled Parallel'])
        plt.title("Execution time for the SNN for various implementations")
        plt.xlabel("Number of Neurons/Threads")
        plt.ylabel("Execution time in milliseconds in log scale")
        plt.yscale("log")
        plt.show()
        plt.savefig('plots/project_optimization_speed.png')
