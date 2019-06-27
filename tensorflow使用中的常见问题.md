#### 使用```tf.saved_model.loader.load(self.sess, ['eval'], pb_file_path)```是遇到的KeyError错误
解决办法:在load的时候```import tensorflow.contrib.factorization```,原因可能是因为tensorflow在定位NearesNeighbors的时候遇到了错误。  
refer：[KeyError: u'NearestNeighbors' on loading saved model from tf.contrib.factorization.KMeansClustering](https://stackoverflow.com/questions/50276275/keyerror-unearestneighbors-on-loading-saved-model-from-tf-contrib-factorizati)

#### 使用```tf.estimator```在做eval和predict的时候会默认载入out_dir里最新的ckpt

解决办法:在```estimator.predict```里的参数```checkpoint_path```赋值


####  train.init_from_checkpoint does not support mirrorredStrategy and CollectiveAllReduceStrategy
1.确定optimizor