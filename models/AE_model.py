import tensorflow as tf
import os

class AutoEncoder():
    def __init__(self, LOGDIR=None, learning_rate=1e-4, testing=False, init_model_path=None):
        ''' testing - no summaries '''
        self.LOGDIR = LOGDIR
        self.learning_rate = learning_rate
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
        self.inputs = tf.placeholder(tf.float32, [None,5,8,1280],name='input')                  
        self.targets = tf.placeholder(tf.float32, [None,144,256,1],name='target')               
        self.labels = tf.reshape(self.targets, shape=[-1,144*256])                             
        # conv encoder
        conv1 = tf.layers.conv2d(self.inputs,512,(3,3),padding='same',activation=tf.nn.relu)   
        conv1 = tf.layers.conv2d(conv1,512,(3,3),padding='same',activation=tf.nn.relu)         
        conv2 = tf.layers.conv2d(conv1,128,(3,3),padding='same',activation=tf.nn.relu)          
        conv2 = tf.layers.conv2d(conv2,128,(3,3),padding='same',activation=tf.nn.relu)       
        r1 = tf.reshape(conv2, shape=[-1,5*8*128])                                            
        # fc encoder
        fc1 = tf.layers.dense(r1,512,activation=tf.nn.relu)                                  
        self.encoded_img = tf.layers.dense(fc1,128,activation=tf.nn.relu,name='encoder_output')             
        # fc decoder
        fc3 = tf.layers.dense(self.encoded_img,512,activation=tf.nn.relu)                             
        fc4 = tf.layers.dense(fc3,5*8*128,activation=tf.nn.relu)                                
        fc4 = tf.reshape(fc4,shape=[-1,5,8,128])                                          
        # conv decoder
        us1 = tf.image.resize_images(fc4,(9,16),tf.image.ResizeMethod.NEAREST_NEIGHBOR)          
        deconv1 = tf.layers.conv2d_transpose(us1,256,(3,3),padding='same',activation=tf.nn.relu)
        deconv1 = tf.layers.conv2d_transpose(deconv1,256,(3,3),padding='same',activation=tf.nn.relu)
        us2 = tf.image.resize_images(deconv1,(18,36),tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        deconv2 = tf.layers.conv2d_transpose(us2,128,(3,3),padding='same',activation=tf.nn.relu)
        deconv2 = tf.layers.conv2d_transpose(deconv2,128,(3,3),padding='same',activation=tf.nn.relu)
        us3 = tf.image.resize_images(deconv2,(36,64),tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        deconv3 = tf.layers.conv2d_transpose(us3,64,(3,3),padding='same',activation=tf.nn.relu) 
        deconv3 = tf.layers.conv2d_transpose(deconv3,64,(3,3),padding='same',activation=tf.nn.relu)
        us4 = tf.image.resize_images(deconv3,(72,128),tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        deconv4 = tf.layers.conv2d_transpose(us4,64,(3,3),padding='same',activation=tf.nn.relu) 
        deconv4 = tf.layers.conv2d_transpose(deconv4,64,(3,3),padding='same',activation=tf.nn.relu)
        us5 = tf.image.resize_images(deconv4,(144,256),tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        deconv4 = tf.layers.conv2d_transpose(us5,1,(3,3),padding='same',activation=tf.nn.relu)
        self.logits = tf.layers.conv2d_transpose(deconv4,1,(3,3),padding='same',activation=tf.nn.relu)
        self.predictions = tf.reshape(self.logits,[-1,144*256])
        self.decoder_image = self.logits*255
        # loss
        self.loss = tf.losses.mean_squared_error(labels=self.labels, predictions=self.predictions)
        self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step_tensor)
        
    def init_summary(self):
        # model saver
        self.saver = tf.train.Saver()
        if not self.testing:
            # summary writer
            self.writer = tf.summary.FileWriter(self.LOGDIR, graph=self.sess.graph)
            # merged summaries
            tf.summary.image('output',self.decoder_image,1)
            tf.summary.scalar('loss',self.loss)
            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)
            self.merged_summary = tf.summary.merge_all()
        
    def save_summary(self, features_batch, images_batch): # (n,5,8,1280) (n,144,256,1)
        global_step = tf.train.global_step(self.sess, self.global_step_tensor)
        feed_dict = { self.inputs: features_batch, self.targets: images_batch }
        [ summary ] = self.sess.run([self.merged_summary], feed_dict=feed_dict)
        self.writer.add_summary(summary, global_step)
    
    def save_model(self):
        self.saver.save(self.sess, os.path.join(self.LOGDIR,'saved_model','model.ckpt'))
        
    def update(self, features_batch, images_batch):
        feed_dict = { self.inputs: features_batch, self.targets: images_batch }
        [ _ ] = self.sess.run([self.update_op], feed_dict=feed_dict)

    def forward(self, features_batch):
        feed_dict = { self.inputs: features_batch }
        [ output ] = self.sess.run([self.encoded_img], feed_dict=feed_dict)
        return output