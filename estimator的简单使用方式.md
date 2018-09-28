### estimator的简单使用方式
[estimator](https://www.tensorflow.org/programmers_guide/estimators?hl=zh-cn)的官方使用方式介绍了使用自定义的estimator的model，没有涉及到从keras的model来使用estimator。
主要的使用方式来自这篇[notebook](https://github.com/kashif/tf-keras-tutorial/blob/master/7-estimators-multi-gpus.ipynb)在使用的时候没有遇上太多障碍。
但有一些细节花了一点时间去调试。
比如estimator能按照dataset重复次数```dataset.repeat(n)```来作为epoch，因此如果直接使用```dataset.repeat()```会在训练时陷入死循环。

#### model_fn的处理
``` python
def model_fn(features, labels, mode):
    keras_estimator_obj = tf.keras.estimator.model_to_estimator(
        keras_model=base_model,
        model_dir=<model_dir>,
        config=<run_config>,
    ) 

    # pull model_fn that we need (hack)
    return keras_estimator_obj._model_fn
```

通过传递参数是无法打印更多的训练结果，但是可以通过创建一个logging hook来让estimator运行。
In the body of model_fn function for your estimator:
``` python
logging_hook = tf.train.LoggingTensorHook({"loss" : loss, 
    "accuracy" : accuracy}, every_n_iter=10)

# Rest of the function

return tf.estimator.EstimatorSpec(
    ...params...
    training_hooks = [logging_hook])
```

除了``` self.estimator.train()```以外,可以使用```tf.estimator.train_and_evaluate()```对```train```和```evaluate```进行更精细地操作。  

此外```add_metrics(estimator,my_auc)```只是把metrics加入到最终结果的输出里，而不是每一次step，对于每一次step需要在```EstimatorSpec(training_hook=[logging_hook])```里添加logging_hook

多gpu出现的
All hooks must be SessionRunHook instances问题在[#issues21444](https://github.com/tensorflow/tensorflow/issues/21444) 里解决，等待tf-1.11版本。

