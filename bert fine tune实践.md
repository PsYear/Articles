# bert fine-tune 实践
从11月初开始，[google-research](https://github.com/google-research)就陆续开源了[bert](https://github.com/google-research/bert)的各个版本。google此次开源的bert是通过tensorflow高级ap—— ```tf.estimator```进行封装(wraper)的。因此对于不同数据集的适配，只需要修改代码中的processor部分，就能进行代码的训练、交叉验证和测试。

## 在自己的数据集上运行bert
bert的代码同论文里描述的一致，主要分为两个部分。一个是训练语言模型（language model）的预训练（pretrain）部分。另一个是训练具体任务(task)的fine-tune部分。在开源的代码中，预训练的入口是在```run_pretraining.py```而fine-tune的入口针对不同的任务分别在```run_classifier.py```和```run_squad.py```。其中```run_classifier.py```适用的任务为分类任务。如CoLA、MRPC、MultiNLI这些数据集。而```run_squad.py```适用的是阅读理解(MRC)任务，如squad2.0和squad1.1。预训练是bert很重要的一个部分，与此同时，预训练需要巨大的运算资源。按照论文里描述的参数，其Base的设定在消费级的显卡Titan x 或Titan 1080ti （12GB RAM）上，甚至需要近几个月的时间进行预训练，同时还会面临显存不足的问题。不过所幸的是谷歌针对大部分语言都公布了bert的[预训练模型](https://github.com/google-research/bert/blob/master/multilingual.md)。因此在我们可以比较方便得在自己的数据集上进行fine-tune。
### 下载预训练模型
对于中文而言，google公布了一个参数较小的bert预训练模型。具体参数数值如下所示：
>Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M parameters  

模型的[下载链接](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)可以在github上google的开源代码里找到。对下载的压缩文件进行解压，可以看到文件里有五个文件，其中bert_model.ckpt开头的文件是负责模型变量载入的，而vocab.txt是训练时中文文本采用的字典，最后bert_config.json是bert在训练时，可选调整的一些参数。

### 修改processor
任何模型的训练、预测都是需要有一个明确的输入，而bert代码中processor就是负责对模型的输入进行处理。我们以分类任务的为例，介绍如何修改processor来运行自己数据集上的fine-tune。在```run_classsifier.py```文件中我们可以看到，google对于一些公开数据集已经写了一些processor，如```XnliProcessor```,```MnliProcessor```,```MrpcProcessor```和```ColaProcessor```。这给我们提供了一个很好的示例，指导我们如何针对自己的数据集来写processor。  
对于一个需要执行训练、交叉验证和测试完整过程的模型而言，自定义的processor里需要继承DataProcessor，并重载获取label的```get_labels```和获取单个输入的```get_train_examples```,```get_dev_examples```和```get_test_examples```函数。其分别会在```main```函数的```FLAGS.do_train```、```FLAGS.do_eval```和```FLAGS.do_predict```阶段被调用。  
这三个函数的内容是相差无几的，区别只在于需要指定各自读入文件的地址。以```get_train_examples```为例，函数需要返回一个由```InputExample```类组成的```list```。```InputExample```类是一个很简单的类，只有初始化函数，需要传入的参数中guid是用来区分每个example的，可以按照```train-%d'%(i)```的方式进行定义。text_a是一串字符串，text_b则是另一串字符串。在进行后续输入处理后(bert代码中已包含，不需要自己完成) text_a和text_b将组合成```[CLS] text_a [SEP] text_b [SEP]```的形式传入模型。最后一个参数label也是字符串的形式，label的内容需要保证出现在```get_labels```函数返回的```list```里。  
举一个例子，假设我们想要处理一个能够判断句子相似度的模型，现在在```data_dir```的路径下有一个名为```train.csv```的输入文件，如果我们现在输入文件的格式如下csv形式：
```
1,你好,您好
0,你好,你家住哪 
```
那么我们可以写一个如下的```get_train_examples```的函数。当然对于csv的处理，可以使用诸如```csv.reader```的形式进行读入。
``` python
def get_train_examples(self, data_dir)：
    file_path = os.path.join(data_dir, 'train.csv')
    with open(file_path, 'r') as f:
        reader = t.readlines()
    examples = []
    for index, line in enumerate(reader):
        guid = 'train-%d'%index
        split_line = line.strip().split(',')
        text_a = tokenization.convert_to_unicode(split_line[1])
        text_b = tokenization.convert_to_unicode(split_line[2])
        label = split_line[0]
        examples.append(InputExample(guid=guid, text_a=text_a,
                                        text_b=text_b, label=label))
    return examples
```
同时对应判断句子相似度这个二分类任务，```get_labels```函数可以写成如下的形式：
``` python
def get_labels(self):
    reutrn ['0','1']
```
在对```get_dev_examples```和```get_test_examples```函数做类似```get_train_examples```的操作后，便完成了对processor的修改。其中```get_test_examples```可以传入一个随意的label数值，因为在模型的预测（prediction）中label将不会参与计算。  

### 修改processor字典
修改完成processor后，需要在在原本```main```函数的processor字典里，加入修改后的processor类，即可在运行参数里指定调用该processor。
``` python
 processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "xnli": XnliProcessor, 
      "selfsim": SelfProcessor #添加自己的processor
  }
```
### 运行fine-tune
之后就可以直接运行```run_classsifier.py```进行模型的训练。在运行时需要制定一些参数，一个较为完整的运行参数如下所示：
``` bash
export BERT_BASE_DIR=/path/to/bert/chinese_L-12_H-768_A-12 #全局变量 下载的预训练bert地址
export MY_DATASET=/path/to/xnli #全局变量 数据集所在地址

python run_classifier.py \
  --task_name=selfsim \ #自己添加processor在processors字典里的key名
  --do_train=true \
  --do_eval=true \
  --dopredict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \ #模型参数
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=2.0 \
  --output_dir=/tmp/selfsim_output/ #模型输出路径
```

## bert源代码里还有什么
在开始训练我们自己fine-tune的bert后，我们可以再来看看bert代码里除了processor之外的一些部分。
我们可以发现，process在得到字符串形式的输入后，在```file_based_convert_examples_to_features```里先是对字符串长度，加入[CLS]和[SEP]等一些处理后，将其写入成TFrecord的形式。这是为了能在estimator里有一个更为高效和简易的读入。  
我们还可以发现，在```create_model```的函数里，除了从```modeling.py```获取模型主干输出之外，还有进行fine-tune时候的loss计算。因此，如果对于fine-tune的结构有自定义的要求，可以在这部分对代码进行修改。如进行NER任务的时候，可以按照bert论文里的方式，不只读第一位的logits，而是将每一位logits进行读取。  
bert这次开源的代码，由于是考虑在google自己的TPU上高效地运行，因此




## issues里一些有趣的内容
从google对bert进行开源开始，issues里的讨论便异常活跃，bert论文第一作者javob也积极地在issues里进行回应


## 总结
总的来说，google此次开源的bert和其预训练模型是非常有价值的，可探索和改进的内容也很多。在感谢google这份付出的同时，我们也可以借此站在巨人的肩膀上，尝试将其运用在自然语言处理领域的方方面面，来让人工智能的梦想更近一步。