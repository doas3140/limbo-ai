import tensorflow as tf
import os

class RL_network():
    def __init__(self, LOGDIR=None, learning_rate=1e-5, memory_size=3, testing=False, init_model_path=None):
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
        self.s = tf.placeholder(tf.float32, [None,self.memory_size,261],name='s_input')           # (n,m,261)
        self.a = tf.placeholder(tf.float32, [None,10],name='a_input')                
        self.v_correct = tf.placeholder(tf.float32, [None,1],name='v_input')
        self.advantage = tf.placeholder(tf.float32, [None,1],name='advantage_input')
        # conv encoder
        conv1 = tf.layers.conv1d(self.s,filters=1,kernel_size=1,padding='same',data_format='channels_first') # (n,1,261)
        conv1 = tf.reshape(conv1,[-1,261])                                               # (n,261)
        # fully connected
        fc1 = tf.layers.dense(conv1,512,activation=tf.nn.relu,name='fc1')             # (n,128)
        fc2 = tf.layers.dense(fc1,128,activation=tf.nn.relu,name='fc2')
        fc3 = tf.layers.dense(fc2,128,activation=tf.nn.relu,name='fc3')                      # (n,64)
        # outputs
        self.v_prediction = tf.layers.dense(fc3,1,name='v_pred')      # (n,1)
        self.a_logits = tf.layers.dense(fc3,10,name='a_pred')
        self.a_probs = tf.nn.softmax(self.a_logits)
        self.sample_action = tf.squeeze(tf.multinomial(logits=self.a_logits, num_samples=1))
        # losses
        self.v_loss = tf.losses.mean_squared_error(labels=self.v_correct, predictions=self.v_prediction)
        entropy = tf.losses.softmax_cross_entropy(onehot_labels=self.a, logits=self.a_logits)
        self.a_loss = tf.reduce_sum(self.advantage*entropy)
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

    def save_summary(self, s_batch, a_batch, v_batch, advantage_batch): # ()
        global_step = tf.train.global_step(self.sess, self.global_step_tensor)
        feed_dict = { self.s: s_batch, self.a: a_batch, self.v_correct: v_batch, self.advantage: advantage_batch }
        [ summary ] = self.sess.run([self.merged_summary], feed_dict=feed_dict)
        self.writer.add_summary(summary, global_step)
    
    def save_model(self):
        self.saver.save(self.sess, os.path.join(self.LOGDIR,'saved_model','model.ckpt'))

    def forward(self, s_batch):
        feed_dict = { self.s: s_batch }
        [ action_probs, v ] = self.sess.run([self.a_probs, self.v_prediction], feed_dict=feed_dict)
        return action_probs, v

    def update(self, s_batch, a_batch, v_batch, advantage_batch):
        feed_dict = { self.s: s_batch, self.a: a_batch, self.v_correct:v_batch, self.advantage: advantage_batch }
        [ _ ] = self.sess.run([self.update_op], feed_dict=feed_dict)