### estimator的简单使用方式
[estimator](https://www.tensorflow.org/programmers_guide/estimators?hl=zh-cn)的官方使用方式介绍了使用自定义的estimator的model，没有涉及到从keras的model来使用estimator。
主要的使用方式来自这篇[notebook](https://github.com/kashif/tf-keras-tutorial/blob/master/7-estimators-multi-gpus.ipynb)在使用的时候没有遇上太多障碍。
但有一些细节花了一点时间去调试。
比如estimator能按照dataset重复次数```dataset.repeat(n)```来作为epoch，因此如果直接使用```dataset.repeat()```会在训练时陷入死循环。