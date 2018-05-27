import tensorflow as tf
import os

class SL_network():
    def __init__(self, LOGDIR=None, learning_rate=1e-3, memory_size=3, testing=False, init_model_path=None):
        ''' testing - no summaries '''
        self.LOGDIR = LOGDIR
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.testing = testing
        self.init_model_path = init_model_path
        # init everything
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.sess.graph.as_default():
            self.init_graph()
            self.init_summary()
            # load model or init new weights
            if init_model_path is None:
                self.sess.run(tf.global_variables_initializer())
            elif init_model_path == 'same':
                self.saver.restore(self.sess, os.path.join(self.LOGDIR,'saved_model','model.ckpt'))
            else:
                self.saver.restore(self.sess, init_model_path)
    
    def init_graph(self):
        self.global_step_tensor = tf.Variable(0, trainable=False)
        # placeholders
        self.s = tf.placeholder(tf.float32, [None,self.memory_size,261],name='input_s')                         
        self.a = tf.placeholder(tf.float32, [None,10],name='input_a')                
        self.v = tf.placeholder(tf.float32, [None,1],name='input_v') 
        # conv layers
        conv1 = tf.layers.conv1d(self.s,filters=1,kernel_size=1,padding='same',data_format='channels_first') 
        conv1 = tf.reshape(conv1,[-1,261])                                                                   
        # fully connected layers
        fc1 = tf.layers.dense(conv1,512,activation=tf.nn.relu,name='fc1')                                   
        fc1 = tf.layers.dropout(fc1,rate=0.2)
        fc2 = tf.layers.dense(fc1,256,activation=tf.nn.relu,name='fc2')                                      
        fc2 = tf.layers.dropout(fc2,rate=0.2)
        fc3 = tf.layers.dense(fc2,64,activation=tf.nn.relu,name='fc3')                                   
        fc3 = tf.layers.dropout(fc3,rate=0.2)
        # outputs
        self.v_prediction = tf.layers.dense(fc3,1,name='v_pred')                       
        self.a_logits = tf.layers.dense(fc3,10,name='a_pred')
        self.a_probs = tf.nn.softmax(self.a_logits)                                             
        # losses
        self.v_loss = tf.losses.mean_squared_error(labels=self.v, predictions=self.v_prediction)
        self.a_loss = tf.losses.softmax_cross_entropy(onehot_labels=self.a, logits=self.a_logits)
        self.loss = self.a_loss + self.v_loss
        self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step_tensor)
        
    def init_summary(self):
        # model saver
        self.saver = tf.train.Saver()
        if not self.testing:
            # summary writer
            self.writer = tf.summary.FileWriter(self.LOGDIR, graph=self.sess.graph)
            # merged summaries
            tf.summary.scalar('total_loss',self.loss)
            tf.summary.scalar('v_loss',self.v_loss)
            tf.summary.scalar('a_loss',self.a_loss)
            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)
            self.merged_summary = tf.summary.merge_all()
        
    def save_summary(self, s_batch, a_batch, v_batch):
        global_step = tf.train.global_step(self.sess, self.global_step_tensor)
        feed_dict = { self.s: s_batch, self.a: a_batch, self.v: v_batch }
        [ summary ] = self.sess.run([self.merged_summary], feed_dict=feed_dict)
        self.writer.add_summary(summary, global_step)
    
    def save_model(self):
        self.saver.save(self.sess, os.path.join(self.LOGDIR,'saved_model','model.ckpt'))
        
    def update(self, s_batch, a_batch, v_batch):
        feed_dict = { self.s: s_batch, self.a: a_batch, self.v: v_batch }
        [ _ ] = self.sess.run([self.update_op], feed_dict=feed_dict)
        
    def forward(self, s_batch):
        feed_dict = { self.s: s_batch }
        [ action_probs, v ] = self.sess.run([self.a_probs, self.v_prediction], feed_dict=feed_dict)
        return action_probs, v