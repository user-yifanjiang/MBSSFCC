'''
@File    :   multi_processing.py

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/9/12 1:13   yifan Jiang      1.0         None
'''
from multiprocessing import Process
from model import main
import utils as util

if __name__ == "__main__":
    multiple = 1
    process = []
    path = "E:\yanyiwenjian\SSF-master\KUL_single_single3"
    names = ['S' + str(i+1) for i in range(1, 15)]
    for name in names:
        p = Process(target=main, args=(name, path,))
        p.start()
        process.append(p)
        util.monitor(process, multiple, 60)
