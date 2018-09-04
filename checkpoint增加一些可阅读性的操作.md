### checkpoint增加一些可阅读性的操作
checkpoint的基础使用在官方的[手册](https://www.tensorflow.org/guide/saved_model)里描述地比较清楚了。但在进行迁移学习时，需要对一些预训练的权重进行读取，因此如果能可阅读得打印一些变量，可以使得读取过程变得简捷便利。

#### checkpoint里变量名和权重值
``` python
    from tensorflow.python import pywrap_tensorflow
    import os
    model_dir = 'dir'
    file_name = 'ckptfile'

    checkpoint_path = os.path.join(model_dir,file_name)
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print('tensor_name: ',key) #变量名
        print(reader.get_tensor(key)) #变量值
```
在ckpt文件里的变量有两类，一类是进行前馈的权重，另一类是在后馈时的梯度。同时有一些变量并不是此前在构建模型时声明的，而是在实现各类模型api时自动产生的，通常这类变量会根据参数产生W和bias。对于包含多步线性计算的cell即各类RNN的cell而言，W会被整合成一个名为kernel的变量，其tensor大小将根据具体的计算方式生成，如lstm的kernel变量大小为(input_dim+lstm_dim,4*lstm_dim)。


#### sess里变量读取预训练权重
``` python
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    print_variable = [var.name for var in tf.global_variables()]
    print(print_variable)
```
因为ckpt读取变量需要新变量和ckpt里的变量名字完全一致，所以可以通过上述代码查看变量名是否满足条件。

``` python
    variables_to_restore = [var for var in tf.global_variables()
        if var.name=='bias:0' or var.name=='kernel:0']
    saver = tf.train.Saver(variables_to_restore)
    model_dir = 'dir'
    file_name = 'ckptfile'
    checkpoint_path = os.path.join(model_dir,file_name)
    saver.restore(self.sess,checkpoint_path)

```
通过在saver初始化时传入需要读取的参数，可以控制restore哪些变量。