1. linux下查看pid进程的具体信息,如pid=33445  
```ps aux|grep 33445```  
2. keras增加log的内容
``` python
    ppl = Lambda(K.exp)(loss)
	model.metrics_names.append('ppl')
	model.metrics_tensors.append(ppl)
```