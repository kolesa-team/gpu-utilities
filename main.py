from pynvml import nvml
from pynvml.smi import nvidia_smi

if __name__ == '__main__':
    try:
        nvsmi = nvidia_smi.getInstance()
    except nvml.NVMLError:
        print('Exception during nvidia_smi initialization')
    print('after exception')

    # gpu_info = nvsmi.DeviceQuery('name,uuid,index')
    # print(gpu_info)
