import os
import numpy as np
import tensorflow as tf
import cv2
import time
import sys
import pickle
sys.path.append("/home/henry/yoro/utils/")
import ROLO_utils as utils
#sys.path.append("/home/henry/yoro/")
#import yolo as y

class ROLO_TF:
    disp_console = True
    restore_weights = True


    # ROLO Network Parameters
    rolo_weights_file = 'weightR/model_demo.ckpt'
    # rolo_weights_file = '/u03/Guanghan/dev/ROLO-dev/model_dropout_30.ckpt'
    lstm_depth = 3
    num_steps = 3  # number of frames as an input sequence
    num_feat = 4096
    num_predict = 6 # final output of LSTM 6 loc parameters
    num_gt = 4
    num_input = num_feat + num_predict # data input: 4096+6= 5002


    # ROLO Parameters
    batch_size = 1
    display_step = 1


    # tf Graph input
    x = tf.placeholder("float32", [None, num_steps, num_input])
    istate = tf.placeholder("float32", [None, 2*num_input]) #state & cell => 2x num_input
    y = tf.placeholder("float32", [None, num_gt])


    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([num_input, num_predict]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_predict]))
    }


    #reading command
    def __init__(self,argvs = []):
        print("ROLO init")
        self.ROLO(argvs)


    #LSTM unit
    def LSTM_single(self, name,  _X, _istate, _weights, _biases):
        # input shape: (batch_size, n_steps, n_input)
        _X = tf.transpose(_X, [1, 0, 2])  # permute num_steps and batch_size
        # Reshape to prepare input to hidden activation
        _X = tf.reshape(_X, [self.num_steps * self.batch_size, self.num_input]) # (num_steps*batch_size, num_input)
        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        _X = tf.split(0, self.num_steps, _X) # n_steps * (batch_size, num_input)
        #print("_X: ", _X)
        cell = tf.nn.rnn_cell.LSTMCell(self.num_input, state_is_tuple = False) #self.num_input
        state = _istate
        for step in range(self.num_steps):
            outputs, state = tf.nn.rnn(cell, [_X[step]], state)
            tf.get_variable_scope().reuse_variables()
        #print("output: ", outputs)
        #print("state: ", state)
        return outputs


    # Experiment with dropout
    def dropout_features(self, feature, prob):
        num_drop = int(prob * 4096)
        drop_index = random.sample(xrange(4096), num_drop)
        for i in range(len(drop_index)):
            index = drop_index[i]
            feature[index] = 0
        return feature
    
    
    #building ROLO network (LSTM part) / loading parameter
    def build_networks(self):
        if self.disp_console : print "Building ROLO graph..."

        # Build rolo layers
        self.lstm_module = self.LSTM_single('lstm_test', self.x, self.istate, self.weights, self.biases)
        self.ious= tf.Variable(tf.zeros([self.batch_size]), name="ious")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        #self.saver.restore(self.sess, self.rolo_weights_file)
        if self.disp_console : print "Loading complete!" + '\n'


    #testing rolo 
    def testing(self, x_path, y_path):
        total_loss = 0

        print("TESTING ROLO...")
        # Use rolo_input for LSTM training
        pred = self.LSTM_single('lstm_train', self.x, self.istate, self.weights, self.biases)
        print("pred: ", pred)
        self.pred_location = pred[0][:, 4097:4101]
        print("pred_location: ", self.pred_location)
        print("self.y: ", self.y)

        self.correct_prediction = tf.square(self.pred_location - self.y)
        print("self.correct_prediction: ", self.correct_prediction)
        self.accuracy = tf.reduce_mean(self.correct_prediction) * 100
        print("self.accuracy: ", self.accuracy)
        #optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.accuracy) # Adam Optimizer

        # Initializing the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        with tf.Session() as sess:

            if (self.restore_weights == True):
                sess.run(init)
                self.saver.restore(sess, self.rolo_weights_file)
                print "Loading complete!" + '\n'
            else:
                sess.run(init)

            id = 0 #don't change this

            # Keep training until reach max iterations
            while id < self.testing_iters - self.num_steps:
                # Load training data & ground truth
                batch_xs = self.rolo_utils.load_yolo_output_test(x_path, self.batch_size, self.num_steps, id) # [num_of_examples, num_input] (depth == 1)

                # Apply dropout to batch_xs
                #for item in range(len(batch_xs)):
                #    batch_xs[item] = self.dropout_features(batch_xs[item], 0.4)

                batch_ys = self.rolo_utils.load_rolo_gt_test(y_path, self.batch_size, self.num_steps, id)
                print("Batch_ys_initial: ", batch_ys)
                batch_ys = utils.locations_from_0_to_1(self.w_img, self.h_img, batch_ys)


                # Reshape data to get 3 seq of 5002 elements
                batch_xs = np.reshape(batch_xs, [self.batch_size, self.num_steps, self.num_input])
                batch_ys = np.reshape(batch_ys, [self.batch_size, 4])
                print("Batch_ys: ", batch_ys)

                pred_location= sess.run(self.pred_location,feed_dict={self.x: batch_xs, self.y: batch_ys, self.istate: np.zeros((self.batch_size, 2*self.num_input))})
                print("ROLO Pred: ", pred_location)
                #print("len(pred) = ", len(pred_location))
                print("ROLO Pred in pixel: ", pred_location[0][0]*self.w_img, pred_location[0][1]*self.h_img, pred_location[0][2]*self.w_img, pred_location[0][3]*self.h_img)
                #print("correct_prediction int: ", (pred_location + 0.1).astype(int))

                # Save pred_location to file
                utils.save_rolo_output_test(self.output_path, pred_location, id, self.num_steps, self.batch_size)

                #sess.run(optimizer, feed_dict={self.x: batch_xs, self.y: batch_ys, self.istate: np.zeros((self.batch_size, 2*self.num_input))})

                if id % self.display_step == 0:
                    # Calculate batch loss
                    loss = sess.run(self.accuracy, feed_dict={self.x: batch_xs, self.y: batch_ys, self.istate: np.zeros((self.batch_size, 2*self.num_input))})
                    print "Iter " + str(id*self.batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) #+ "{:.5f}".format(self.accuracy)
                    total_loss += loss
                id += 1
                print(id)

            print "Testing Finished!"
            avg_loss = total_loss/id
            print "Avg loss: " + str(avg_loss)
            #save_path = self.saver.save(sess, self.rolo_weights_file)
            #print("Model saved in file: %s" % save_path)

        return None
        
        
    def ROLO(self, argvs):

            self.rolo_utils= utils.ROLO_utils()
            self.rolo_utils.loadCfg()
            self.params = self.rolo_utils.params

            arguments = self.rolo_utils.argv_parser(argvs)

            if self.rolo_utils.flag_train is True:
                self.training(utils.x_path, utils.y_path)
            elif self.rolo_utils.flag_track is True:
                self.build_networks()
                self.track_from_file(utils.file_in_path)
            elif self.rolo_utils.flag_detect is True:
                self.build_networks()
                self.detect_from_file(utils.file_in_path)
            else:
                print "Default: running ROLO test."
                self.build_networks()

                test= 13
                [self.w_img, self.h_img, sequence_name, dummy_1, self.testing_iters] = utils.choose_video_sequence(test)
                print sequence_name
                x_path = os.path.join('benchmark/DATA', sequence_name, 'yolo_out/')
                y_path = os.path.join('benchmark/DATA', sequence_name, 'groundtruth_rect.txt')
                self.output_path = os.path.join('benchmark/DATA', sequence_name, 'rolo_out_test/')
                utils.createFolder(self.output_path)

                self.rolo_weights_file = 'weightR/model_demo.ckpt'
                self.testing(x_path, y_path)        


#----------------------------------------------------------This is rolo function part--------------------------------------------------------



def main(argvs):
  '''
  #yolo = YOLO_TF()
  yolo = y.YOLO_TF()
  #putting which one you want to tracking (seeing ROLO_util)
  test = 13
  heatmap = False # True
  [yolo.w_img, yolo.h_img, sequence_name, dummy_1, dummy_2]= util.choose_video_sequence(test)
        
  if (test >= 0 and test <= 29) or (test >= 90):
      root_folder = 'benchmark/DATA'
      img_fold = os.path.join(root_folder, sequence_name, 'img/')
  elif test<= 36:
      root_folder = 'benchmark/MOT/MOT2016/train'
      img_fold = os.path.join(root_folder, sequence_name, 'img1/')
  elif test<= 43:
      root_folder = 'benchmark/MOT/MOT2016/test'
      img_fold = os.path.join(root_folder, sequence_name, 'img1/')

  gt_file = os.path.join(root_folder, sequence_name, 'groundtruth_rect.txt')
  out_fold = os.path.join(root_folder, sequence_name, 'yolo_out/')
  heat_fold = os.path.join(root_folder, sequence_name, 'yolo_heat/')
  yolo.createFolder(out_fold)
  yolo.createFolder(heat_fold)

  if heatmap is True:
      yolo.prepare_training_data_heatmap(img_fold, gt_file, heat_fold)
  else:
      if (test >= 0 and test <= 29) or (test >= 90):
          yolo.prepare_training_data(img_fold,gt_file,out_fold)
      else:
          yolo.prepare_training_data_multiTarget(img_fold,out_fold)
  '''
  ROLO_TF()
        
if __name__=='__main__':
  main(sys.argv)  
        