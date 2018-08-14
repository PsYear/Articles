# 在keras中使用tf.dataset作为输入

tf.dataset的API支持keras，是Tensorflow-1.9.0的一个新特性。使用tf.dataset作为输入的pipeline可以减少系统内存和显存的占用率，使得在训练大规模数据时，避免内存容量不够的问题出现。

在tensorflow的官方文档中，有简单的[使用说明](https://www.tensorflow.org/guide/keras)，但仍有一些[bug](https://github.com/tensorflow/tensorflow/issues/20827)在tensorflow-1.10.0中才将被修复。针对这个版本尚存在的一些问题，以下介绍一些可以规避这些问题的使用办法。

将tf.dataset作为输入传入可以在fit()函数中，也可以在Input()层和compiler()函数中分别[传入](https://stackoverflow.com/questions/46135499/how-to-properly-combine-tensorflows-dataset-api-and-keras)。
 ``` python
#在fit()函数传入
model.fit(x=iter_x.get_next(),y=iter_y.get_next(),
          epochs=epochs,steps_per_epoch=steps_per_epoch)
# 在Input()层和compiler()函数传入
inputs = Input(tensor=iter_x.get_next())
model.compile(loss=loss,optimizer=optimizer,
              target_tensors=[iter_y.get_next()])
```
传入时需要将dataset类转换为tensor类。这一步涉及两个步骤，首先需要生成dataset的iterator,对于不同的dataset有不同生成方法，常用的有make_one_shot_iterator和make_initializable_iterator两种。之后通过iterator的get_next()函数迭代获取tensor类数据。
``` python
iterator = dataset.make_initializable_iterator()
iterator = dataset.make_one_shot_iterator()
iterator = iterator.get_next()
```
keras对于传入的tensor有tf.dtype的要求，x需要是tf.float32类型。如果类型不符，可以通过tf.cast()进行类型转换
``` python
inputs = Input(tensor=tf.cast(iter_x.get_next(),tf.float32))
```

面对一系列初始化的需要时，可以先获取kera的session，并在keras的session中对table和iterator进行初始化
``` python
from keras import backend as K
K.get_session().run(tf.tables_initializer())
K.get_session().run(iter_data.initializer)
```

训练时如果采用在Input()层和compiler()函数中传入x和y的方法，后续进行交叉验证、测试集测试以及结果生成时需要构建新的模型。因为这种方式相当于将模型的输入输出固定，将使模型不受输入输出的影响。新构建的模型需要符合原来模型的结构，但在Input()层和compiler()函数中的相应参数可以改为新的x和y。再传入训练时模型的权重并调用evaluate或者predict进行验证和预测。一个kears官方的[示例](https://github.com/keras-team/keras/blob/master/examples/mnist_dataset_api.py)
如果在notebook中使用这种方法需要注意在内存中清除原来的会话。
``` python
K.clear_session()
```