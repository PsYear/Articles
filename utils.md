1. keras增加log的内容
``` python
    ppl = Lambda(K.exp)(loss)
	model.metrics_names.append('ppl')
	model.metrics_tensors.append(ppl)
```