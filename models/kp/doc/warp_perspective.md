
## ocr业务上两种预处理的性能压测
1. opencv解码 + 预处理(getRotationMatrix2D、warpAffine)
   ![1.png](images/cpu_1.png)

![2.png](images/cpu_2.png)

服务QPS达到44.5左右，4个客户端同时请求的时延为88ms，cpu资源占用高达60%

2. torchnvjpeg解码 + kprocess预处理
   ![3.png](images/gpu_1.png)

![4.png](images/gpu_2.png)

服务QPS达到93.3左右，4个客户端同时请求的时延为41ms，cpu资源占用仅为27.6%
