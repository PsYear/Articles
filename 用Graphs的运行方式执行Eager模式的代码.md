
一、在执行代码时使用 ``` tf.enable_eager_execution()``` 开启eager模式

二、正向传播支持自定义class类型  
&ensp;&ensp;&ensp;&ensp; 1. [定义](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/rnn_ptb/rnn_ptb.py#L99)的model继承keras.model<br>
&ensp;&ensp;&ensp;&ensp; 2. 在[__init__()](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/rnn_ptb/rnn_ptb.py#L110)里定义所用到的layer类型<br>
&ensp;&ensp;&ensp;&ensp; 3. 在[call()](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/rnn_ptb/rnn_ptb.py#L133)里连接layer，返回[output](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/rnn_ptb/rnn_ptb.py#L146)<br>

二、反向传播的使用  
&ensp;&ensp;&ensp;&ensp;1. 先用tf的api定义[loss函数](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/rnn_ptb/rnn_ptb.py#L158)<br>
&ensp;&ensp;&ensp;&ensp;2. 用tfe的[api](https://www.tensorflow.org/api_docs/python/tf/contrib/eager/implicit_gradients)调用loss得到梯度grads
``` Python
     tfe.gradients_function(loss,x)
     tfe.implicit_gradients(loss)   
```  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;或者也可以采用[GradientTape(y,x)](https://www.tensorflow.org/api_docs/python/tf/GradientTape)来进行计算可以根据函数y计算变量x的梯度  
``` Python
      with tf.GradientTape() as grad_tape:
          grad_tape.gradient(y,x) 
```    
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;在tf的诸多eager execution样例中里用第二种方法较多  

&ensp;&ensp;&ensp;&ensp;3. 用tf定义的[optimizer](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/rnn_ptb/rnn_ptb.py#L317)优化梯度更新参数  
``` python
      optimizer.apply_gradients(grad)
```

三、Eager的输入使用```tf.data.Dataset```但不支持```placeholder```和``string_input_producer``这类在graph模式中使用的输入

四、使用```tf.train.Checkpoint()```[保存](https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint)模型的checkpoint

五、可以使用```tf.contrib.eager.defun```对python的函数进行封装转换成图的形式进行运算。
    用eager的写法可以实现graph的运算速度。  
&ensp;&ensp;&ensp;&ensp; 1. 使用了defun的forward propagation例子如下:
``` python
        model.call = tf.contrib.eager.defun(model.call)
        model(x, training=True)  # executes a graph, with dropout
```  
&ensp;&ensp;&ensp;&ensp; 2. 一个使用了defun的back propagation例子如下
``` python
        optimizer = tf.train.GradientDescentOptimizer()
        with tf.GradientTape() as tape:
          outputs = model(x)
        gradient = tape.gradient(outputs, model.trainable_variables)
        defun_gradients = tfe.defun(gradient)
        tfe.defun(optimizer.apply_gradients((grad, var) for grad, 
                     var in zip(gradient,model.trainable_variables)))
```
&ensp;&ensp;&ensp;&ensp; 然而此后[defun](https://www.tensorflow.org/api_docs/python/tf/contrib/eager/defun)可能会被[AutoGraph](https://medium.com/tensorflow/autograph-converts-python-into-tensorflow-graphs-b2a871f87ec7)替代




Refer:[Code with Eager Execution, Run with Graphs: Optimizing Your Code with RevNet as an Example](https://medium.com/tensorflow/code-with-eager-execution-run-with-graphs-optimizing-your-code-with-revnet-as-an-example-6162333f9b08?linkId=55410234)