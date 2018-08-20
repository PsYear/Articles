### Save keras model in protobuf format
tensorflow 有许多种不同的模型存储[格式](https://zhuanlan.zhihu.com/p/34471266),不同的存储方式调用的存储函数是不一样的，使用tensorflow作为后端的keras也同样支持这些模型保存的方法。这里介绍一种能够方便部署在服务器端和移动端的[protobuf](https://developers.google.com/protocol-buffers/)格式。
建议采用```from tensorflow import keras```以及``` from tensorflow.python.keras.models import...```来使用keras
在构建好keras的网络结构后就可以开始对模型进行保存。为了使之后能便捷找到模型的结点，可以构建模型时对每一层的name参数进行赋值，对相关层命名。
``` python
    context_input = Input(shape=(10,),name='context_input')
    out = Dense((1), activation = "sigmoid",name="out")
```
构建完模型后可以获得模型在tensorflow下的graph，查看网络结构是否如愿搭建以及此前命名是否成功。
``` python
    graph = K.get_session().graph
    K.set_learning_phase(0)
    for op in graph.get_operations():
        print(op.name)
```
模型在保存前，可以通过```K.set_learning_phase(0)```将模型的参数设为不可变化的非训练模式。
模型的保存可以采用SaveModel的API进行保存，保存的结果将得到一个目录，目录里包含模型结构的pb文件以及包含参数名称和值的另一个目录。
``` python
from tensorflow.python.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np
export_path = './keras_save'
signature = tf.saved_model.signature_def_utils.predict_signature_def(
    inputs={'sentence_name': dual_encoder.input}, #没有太明白signature的作用 #TODO
    outputs={'outputs_name': dual_encoder.output})
builder = tf.saved_model.builder.SavedModelBuilder(export_path)

with K.get_session() as sess:
    builder.add_meta_graph_and_variables(
                sess,['eval'],
                signature_def_map={'predict': signature})
    builder.save(True)
print('Finished export', export_path)
```
其中add_meta_graph_and_variables的第二个参数是[可选字符](https://www.tensorflow.org/api_docs/python/tf/saved_model/builder/SavedModelBuilder)虽然保存之后目录里有.pb文件，但这个.pb文件的格式并不能通过```graph_def.ParseFromString(f.read())```的方式进行读入，因为tensorflow的文件读写需要完全使用成对的api进行完成。对应于```f.saved_model.builder.SavedModelBuilder()```的是```tf.saved_model.loader.load()```具体的使用方式如下：
``` python

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ['eval'], pb_file_path+'keras_save')
    sess.run(tf.global_variables_initializer())
    input_x1 = sess.graph.get_tensor_by_name('context_input:0')
    output_y = sess.graph.get_tensor_by_name('out/Sigmoid:0') 
                                        #其实是有更好的办法读预测结构 #TODO
    ret = sess.run(output_y,feed_dict={input_x1:data1})
```
这样```ret```得到的值便是对应输入```data1```的模型前馈的结果。
这部分有两个比较耗时的地方，首先是在load部分，其次是在get_tensor_by_name部分。因此在实际部署时，可以保持sess处于常开状态来减小i/o开销。此外，每个sess第一次run的时间将为之后run时间的数十倍。

#### reference

- [Tensorflow-Guide Graphs and Sessions](https://www.tensorflow.org/guide/graphs)
- [Export Keras Model to ProtoBuf for Tensorflow Serving](https://medium.com/@johnsondsouza23/export-keras-model-to-protobuf-for-tensorflow-serving-101ad6c65142)
- [我们给你推荐一种TensorFlow模型格式](https://zhuanlan.zhihu.com/p/34471266)
- [Tensorflow 1.13 Serving搭建心得on Docker（三）把keras或tf model改写成serving格式](https://zhuanlan.zhihu.com/p/29374467)
- [TensorFlow 保存模型为 PB 文件](https://zhuanlan.zhihu.com/p/32887066)
- [metaflow-ai/blog](https://github.com/metaflow-ai/blog/tree/master/tf-freeze)
- [TensorFlow: How to freeze a model and serve it with a python API](https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc)