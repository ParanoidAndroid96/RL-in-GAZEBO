import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
#%matplotlib inline
from helper import *
#from vizdoom import *
import random
import math
from statistics import mean

from random import choice
from time import sleep
from time import time
import rospy
import roslib
from gazebo_msgs.srv import DeleteModel, SpawnModel, SetModelState
from gazebo_msgs.msg import ModelState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import *
from sensor_msgs.msg import *
import tf.transformations as tft 
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from tensorflow.python.tools import inspect_checkpoint as chkp
#import pcl

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
	from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
	to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

	op_holder = []
	for from_var,to_var in zip(from_vars,to_vars):
		op_holder.append(to_var.assign(from_var))
	return op_holder

# Processes Doom screen image to produce cropped and resized image. 
def process_frame(frame):
	s = frame[10:-10,30:-30]
	s = scipy.misc.imresize(s,[200,150])
	#s = np.reshape(s,[np.prod(s.shape)]) / 255.0
	s = np.reshape(s, [-1, 90000])
	return s

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
	return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
	def _initializer(shape, dtype=None, partition_info=None):
		out = np.random.randn(*shape).astype(np.float32)
		out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
		return tf.constant(out)
	return _initializer

#Get random value for velocity from gaussian distribution
def get_vels(mean, variance, vel_type):
	#print("Mean "+str(mean))
	#print("Variance "+str(variance))
	variance = math.exp(variance)
	vel = np.random.normal(mean, variance, 1)[0]
	if(vel > 1.0):
		vel = 1.0
	elif(vel < 0.0): # and vel_type == "lin"):
		vel = 0.0
	elif(vel < -1.0 and vel_type == "ang"):
		vel = -1.0
	return vel

class Environment():
	#Subscribe to the topics for camera and orientation
	def __init__(self, name, target_pose):
		rospy.Subscriber("/"+name+"/odom",Odometry,self.read_odom)
		rospy.Subscriber("/camera/color/image_raw",Image,self.read_cam)
		rospy.Subscriber("/camera/depth/image_raw",Image,self.read_depth_cam)
		self.command_pub = rospy.Publisher('/'+name+'/cmd_vel', Twist, queue_size=1)
		self.reward = 0.0	
		self.target = target_pose
		self.done = False
		self.penalty = 30.0
		self.dist = 0.0
		self.walls = [[10.9, -11.4, -10.8, -11.79], [11.56, 10.4, 10.9, -11.1], [10.8, -11.1, 11.6, 10.6], [-10.8, -11.89, 11.0, -11.0], [-3.45, -11.2, -5.26, -6.1], [-2.9, -3.9, -1.4, -9.0], [3.87, 2.9, -1.4, -9.0], [10.9, 3.5, -5.0, -5.95], [10.75, 3.6, 1.42, 0.5], [3.9, 2.95, 10.75, 3.8], [-3.2, -11.0, 1.1, 0.15], [-2.95, -3.9, 10.8, 3.4], [1.65, -6.0, 8.32, 7.35], [-8.1, -11, 8.24, 7.3]]
	def new_episode(self):
		self.done = False
		self.reward = 0.0	

	def read_odom(self, msg):
		self.x = msg.pose.pose.position.x
		self.y = msg.pose.pose.position.y
		self.z = msg.pose.pose.position.z
		self.ori = msg.pose.pose.orientation
		self.angles = tft.euler_from_quaternion([self.ori.x, self.ori.y, self.ori.z, self.ori.w])
		self.angle = self.angles[2] * 180.0 / 3.14
		if(self.angle < 0.0):
			self.angle += 360.0
		self.bot_lin_x = msg.twist.twist.linear.x
		self.bot_lin_y = msg.twist.twist.linear.y
		self.bot_lin_z = msg.twist.twist.linear.z
		self.bot_ang_x = msg.twist.twist.angular.x
		self.bot_ang_y = msg.twist.twist.angular.y
		self.bot_ang_z = msg.twist.twist.angular.z

	def read_cam(self, msg):
		self.frame = msg

	def read_depth_cam(self, msg):
		bridge = CvBridge()
		cv_image = bridge.imgmsg_to_cv2(msg, "32FC1")
		cv_image_array = np.array(cv_image, dtype = np.dtype('f8'))
		cv_image_norm = cv2.normalize(cv_image_array, cv_image_array, 0, 1, cv2.NORM_MINMAX)
		cv_image_resized = cv2.resize(cv_image_norm, (200,150), interpolation = cv2.INTER_CUBIC)
		cv_image_reshaped = np.reshape(cv_image_resized, [-1, 30000])
		arr=[]
		for i in cv_image_reshaped[0]:
			if(math.isnan(i)):
				arr.append(1000.0)
			else:
				arr.append(i)
		self.depth_frame = np.reshape(arr, [-1,30000])
	#def read_pcl(self, ros_cloud):
    		#self.points_list = np.reshape(list(pc2.read_points(ros_cloud, skip_nans=True, field_names=("x", "y", "z"))), (-1,3))

	#Return raw image and orientation
	def get_frame(self):
		bridge = CvBridge()
		cv_image = bridge.imgmsg_to_cv2(self.frame, "bgr8")
		#Return velocity as well?
		return cv_image, self.angle, self.depth_frame

	#Publish commands to robot to execute correct action, implement getting random value from gaussian distribution
	def step(self, lin_vel, ang_vel):
		direct = Twist()
		direct.angular.x = 0.0
		direct.angular.y = 0.0
		direct.angular.z = ang_vel
		direct.linear.x = lin_vel
		direct.linear.y = 0.0
		direct.linear.z = 0.0
		print("Publishing: "+str(ang_vel)+" "+str(lin_vel))
		self.command_pub.publish(direct)
  

	def near_wall(self):
		for wall in self.walls:
			if(self.x < wall[0] and self.x > wall[1] and self.y < wall[2] and self.y > wall[3]):
				return 1
		return 0

	#Has the robot reached the target
	def at_target(self):
		thresh = 0.5
		if(abs(self.x - self.target[0]) < thresh and abs(self.y - self.target[1]) < thresh): #and stop_action == 1):
			return 1
		else:
			return 0

	def dist_to_target(self):
		return math.sqrt(pow(self.x-self.target[0], 2) + pow(self.y-self.target[1], 2))

	#Reward for action
	def get_reward(self, v, w, stop_action):
		#self.reward = 1.0 * (self.dist - self.dist_to_target())#-0.1
		#self.dist = self.dist_to_target()
		c1 = 2.0
		c2 = 2.0
		c3 = 0.1
		self.reward = (c1*v*v*math.cos(c2*v*w))-c3
		if(self.near_wall()):
			self.reward += -20.0

		#self.reward = -0.01
		#if(min(self.depth_frame[0]) == 1000.0):
			#self.reward = -1.0
		'''if(stop_action == 1):
			#self.done=True
			if(self.at_target()):
				self.done=True
				self.reward=50.0
			else:
				self.reward= -1.0'''
		if(self.at_target()):
			self.done=True
			self.reward=50.0
		return self.reward

class AC_Network():
	def __init__(self,s_size,a_size,scope,trainer):
		with tf.variable_scope(scope):
			#Change input to accept image into a cnn layer, orientation and embedded command aare then combined into a single embedding layer which is fed to the lstm and change output to give two sets of means and variance and a stop command
			#Input and visual encoding layers
			#The None in shape is for batch size and allows for any variable batch size
            		self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
	    		#The -1 in the shape is also for variable batch size that it infers on its own
            		self.imageIn = tf.reshape(self.inputs,shape=[-1,200,150,3])
	    		#Put in Resnet or something else instead
            		self.conv1 = slim.conv2d(activation_fn=tf.nn.elu, inputs=self.imageIn, num_outputs=16, kernel_size=[8,8], stride=[4,4], padding='VALID')
            		self.conv2 = slim.conv2d(activation_fn=tf.nn.elu, inputs=self.conv1, num_outputs=32, kernel_size=[4,4], stride=[2,2], padding='VALID')
            		hidden = slim.fully_connected(slim.flatten(self.conv2),256,activation_fn=tf.nn.elu)
		            
			#Pointcloud input
			self.point_cloud = tf.placeholder(shape=[None, 30000], dtype=tf.float32)
			self.imageIn_2 = tf.reshape(self.point_cloud,shape=[-1,200,150,1])
	    		#Put in Resnet or something else instead
            		self.conv3 = slim.conv2d(activation_fn=tf.nn.elu, inputs=self.imageIn_2, num_outputs=16, kernel_size=[8,8], stride=[4,4], padding='VALID')
            		self.conv4 = slim.conv2d(activation_fn=tf.nn.elu, inputs=self.conv3, num_outputs=32, kernel_size=[4,4], stride=[2,2], padding='VALID')			
			point_cloud_embed = slim.fully_connected(slim.flatten(self.conv4),256,activation_fn=tf.nn.elu)
			
			#Inputs for command and ori
			self.command_inp = tf.placeholder(shape=[None,5], dtype=tf.float32)
			command_inp_embed = slim.fully_connected(slim.flatten(self.command_inp),256,activation_fn=tf.nn.elu)
			self.ori_inp = tf.placeholder(shape=[None,1], dtype=tf.float32)
			ori_inp_embed = slim.fully_connected(slim.flatten(self.ori_inp),256,activation_fn=tf.nn.elu)		
			#Combine inputs
			self.combine = tf.keras.layers.concatenate([hidden, point_cloud_embed, command_inp_embed, ori_inp_embed],axis=-1)
			#combine = tf.concat(0,[combine, ori_inp])
			hidden_2 = slim.fully_connected(self.combine, 256, activation_fn = tf.nn.elu)

            		#Recurrent network for temporal dependencies
	    		#Since state_is_tuple=True, we need to pass an initial state that is a tuple of the cell_state and the hidden_state. Since we only want to initialize the first hidden state, the cell_state is initialized with zeros.
            		lstm_cell = tf.contrib.rnn.BasicLSTMCell(256,state_is_tuple=True)
	    		#cell state
	    		#shape = [batch_size, num_units]
            		c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)#used later
	    		#hidden state
            		h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)#used later
            		self.state_init = [c_init, h_init]
            		c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            		h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            		self.state_in = (c_in, h_in)
            		rnn_in = tf.expand_dims(hidden_2, [0])
            		step_size = tf.shape(self.imageIn)[:1]
            		state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            		lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size, time_major=False)
            		lstm_c, lstm_h = lstm_state
            		self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            		rnn_out = tf.reshape(lstm_outputs, [-1, 256])
            		hidden_3 = slim.fully_connected(rnn_out, 128)
            		#Output layers for policy and value estimations

			#Add stop action
			#hidden_4 = slim.fully_connected(hidden_3, 64)			
            		self.policy = slim.fully_connected(hidden_3,5,
            		    		activation_fn=tf.nn.sigmoid,
                			weights_initializer=normalized_columns_initializer(0.01),
                			biases_initializer=None)

            		self.value = slim.fully_connected(hidden_3,1,
                			activation_fn=None,
                			weights_initializer=normalized_columns_initializer(1.0),
                			biases_initializer=None)

			

            		#Only the worker network need ops for loss functions and gradient updating.
            		if scope != 'global':
                		self.actions = tf.placeholder(shape=[None,2],dtype=tf.float32)
				#a_size is the number of columns in the one_hot vectors and the length of self.actions is the number of rows of one_hot vectors and the values of self.actions are the indices in that row that are ON 
				self.stopping = tf.placeholder(shape=[None,1], dtype=tf.float32)
				self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                		self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)
				#Edit this function to get the correct loss	

				self.distrib_1 = tf.distributions.Normal(self.policy[:,0], tf.exp(self.policy[:,1]))
				self.distrib_2 = tf.distributions.Normal(self.policy[:,2], tf.exp(self.policy[:,3]))
    				self.log_prob_1 = tf.reduce_sum(self.distrib_1.log_prob(self.actions[:,0]))
    				self.log_prob_2 = tf.reduce_sum(self.distrib_2.log_prob(self.actions[:,1]))
    				self.policy_loss_1 = -tf.reduce_mean(self.advantages * self.log_prob_1)
    				self.policy_loss_2 = -tf.reduce_mean(self.advantages * self.log_prob_2)
    				self.entropy = tf.reduce_mean(self.distrib_1.entropy()) + tf.reduce_mean(self.distrib_2.entropy())


				self.responsible_outputs_3 = 0.5 * tf.reduce_sum((tf.maximum(self.stopping[:,0]*(tf.log(self.policy[:,4])), 1e-4)) + (tf.maximum(([[1]]-self.stopping[:,0])*(tf.log([[1]]-self.policy[:,4])), 1e-4)) , [1])

               			 #Loss functions
				#Action must be chosen as mean +/- (random(variance)) and this value should be subtracted by the mean on top of exponent divided by vriance to give loss which is logged and multiplied by advantage.
				self.value_loss = 0.5 * tf.reduce_mean(tf.square(self.target_v - [self.value[0,0]]))
				
				self.other_loss = -tf.reduce_mean(self.responsible_outputs_3*self.advantages) - 1e-6
			
				self.loss = 0.5 * self.value_loss + self.policy_loss_1 + self.policy_loss_2 - self.entropy * 0.01 + self.other_loss

				#Get gradients from local network using local losses
				local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
				#Gets the derivate of the loss function (self.loss), wrt. some local variables(local_vars). Returns a list of tensor of size len(local_vars) where each tensor is the sum(dy/dx) for y in self.loss 
				self.gradients = tf.gradients(self.loss,local_vars)
				self.var_norms = tf.global_norm(local_vars)
				#Normalization of the gradients
				grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
				
				#Apply local gradients to global network
				global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "global")
				self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))




class Worker():
	def __init__(self,name,s_size,a_size,trainer,model_path,global_episodes, xml, spawn_model, delete_model, data_dict):
		self.name = "worker_" + str(name)
		self.number = name        
		self.robot = xml
		self.model_path = model_path
		self.trainer = trainer
		self.global_episodes = global_episodes
		self.increment = self.global_episodes.assign_add(1)
		self.episode_rewards = []
		self.episode_lengths = []
		self.episode_mean_values = []
		self.data = data_dict
		self.summary_writer = tf.summary.FileWriter("train_"+str(self.number))
		#Create the local copy of the network and the tensorflow op to copy global paramters to local network
		self.local_AC = AC_Network(s_size,a_size,self.name,trainer)
		self.update_local_ops = update_target_graph('global',self.name)        
		
		#The Below code is related to setting up the Gazebo environment
		self.env_init(0)
		#End Gazebo set-up
		
	def env_init(self, task):
		#choose random number which will pick corresponding start_pose, target_pose and input
		ind = 0

		if(task == 1):
			print("Re-Positioning Model")
		
			state_msg = ModelState()
			state_msg.model_name = self.name
			#Get next data point and reposition to new start point
			quaternion = tft.quaternion_from_euler(0, 0, random.uniform(0,360))
			point = self.data[random.choice(self.data.keys())]
			state_msg.pose.position.x = random.uniform(point["x"][0], point["x"][1])
			state_msg.pose.position.y = random.uniform(point["y"][0], point["y"][1])
			state_msg.pose.position.z = 0.0
			state_msg.pose.orientation.x = quaternion[0]
			state_msg.pose.orientation.y = quaternion[1]
			state_msg.pose.orientation.z = quaternion[2]
			state_msg.pose.orientation.w = quaternion[3]
			#episode_buffer[-1][2] = episode_reward
			rospy.wait_for_service('/gazebo/set_model_state')
			set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
			resp = set_state( state_msg )

		else:
			print("Spawning model")

			quaternion = tft.quaternion_from_euler(0, 0, random.uniform(0,360))
			object_pose = Pose()
			point = self.data[random.choice(self.data.keys())]
			object_pose.position.x = random.uniform(point["x"][0], point["x"][1])
			object_pose.position.y = random.uniform(point["y"][0], point["y"][1])
			object_pose.position.z = 0.0#self.start_poses[ind][2]
			object_pose.orientation.x = quaternion[0]
			object_pose.orientation.y = quaternion[1]
			object_pose.orientation.z = quaternion[2]
			object_pose.orientation.w = quaternion[3]
			spawn_model(self.name, self.robot, "", object_pose, "world")

		self.com = "right"#random.choice(point["commands"].keys())
		if(self.com == "left"):
			self.input_com = [1,0,0,0,1]
		elif(self.com == "right"):
			self.input_com = [0,1,0,0,1]
		else:
			self.input_com = [0,0,1,0,1]
		self.input_com = np.reshape(self.input_com, [-1, 5])

		print("Command: "+self.com)
		self.env = Environment(self.name, point["commands"][self.com])
		rospy.sleep(1)
		

	def train(self,rollout,sess,gamma,bootstrap_value):
		rollout = np.array(rollout)
		observations = rollout[:,0]
		actions = rollout[:,1]
		rewards = rollout[:,2]
		next_observations = rollout[:,3]
		values = rollout[:,5]
		orientations = rollout[:,6]
		commands = rollout[:,7]
		points = rollout[:,8]
		stop_action = rollout[:,9]
		# Here we take the rewards and values from the rollout, and use them to 
		# generate the advantage and discounted returns. 
		# The advantage function uses "Generalized Advantage Estimation"
		self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
		discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
		self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
		advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
		advantages = discount(advantages,gamma)
		advantages = (advantages - advantages.mean()) / (max(advantages.std(), 1e-4))
		# Update the global network using gradients from loss
		# Generate network statistics to periodically save
		feed_dict = {self.local_AC.target_v:discounted_rewards,
		    self.local_AC.inputs:np.vstack(observations),
		    self.local_AC.actions:actions,
		    self.local_AC.advantages:advantages,
		    self.local_AC.state_in[0]:self.batch_rnn_state[0],
		    self.local_AC.state_in[1]:self.batch_rnn_state[1]}

		#print("Actions= " + str(np.shape(actions)))
		arr=[]
		for i in range(len(actions)):
			arr.append([actions[i][0], actions[i][1]])
		actions = arr
		actions = np.reshape(actions, [-1, 2])
		orientations = np.reshape(orientations, [-1, 1])
		stop_action = np.reshape(stop_action, [-1, 1])
	

		arr=[]
		for i in range(len(commands)):
			arr.append([commands[i][0][0], commands[i][0][1], commands[i][0][2], commands[i][0][3], commands[i][0][4]])
		commands = arr
		commands = np.reshape(commands, [-1,5])

				
		#print("Advantages:")
		#print(advantages)
		v_l,p_l_1,p_l_2,o_l,e_l,l,g,g_n,v_n, self.batch_rnn_state,_ = sess.run([ self.local_AC.value_loss,
		    self.local_AC.policy_loss_1,
		    self.local_AC.policy_loss_2,
		    self.local_AC.other_loss,
		    self.local_AC.entropy,
		    self.local_AC.loss,
		    self.local_AC.gradients,
		    self.local_AC.grad_norms,
		    self.local_AC.var_norms,
		    self.local_AC.state_out,
		    self.local_AC.apply_grads],
		    feed_dict={self.local_AC.target_v:discounted_rewards,
		    self.local_AC.inputs:np.vstack(observations),
		    self.local_AC.advantages:advantages,
		    self.local_AC.state_in[0]:self.batch_rnn_state[0],
		    self.local_AC.state_in[1]:self.batch_rnn_state[1],
		    self.local_AC.actions:actions,
		    self.local_AC.stopping:stop_action,
		    self.local_AC.command_inp:commands,
		    self.local_AC.ori_inp:orientations,
		    self.local_AC.point_cloud:np.vstack(points)})

		print(" ")
		#print(advantages)
		#print(r_o_1)
		#print(r_o_2)
		#print(r_o_3)
		print("Value loss: "+str(v_l))
		print("Policy loss 1: "+str(p_l_1))
		print("Policy loss 2: "+str(p_l_2))
		print("Other loss: "+str(o_l))
		print("Entropy: "+str(e_l))
		print("Grad Norms: "+str(g_n))
		print("Var Norms: "+str(v_n))
		print("Loss: "+str(l))
		#print("Gradients:")
		#print(g)

		return v_l / len(rollout),p_l_1 / len(rollout),e_l / len(rollout), g_n,v_n
        
	def work(self,max_episode_length,gamma,sess,coord,saver):
		episode_count = sess.run(self.global_episodes)
		total_steps = 0
		temp = 1
		f = open(self.name+".txt", "w+")
		print ("Starting worker " + str(self.number))
		with sess.as_default(), sess.graph.as_default():                 
			while not coord.should_stop():
				print("New Count")
				sess.run(self.update_local_ops)
				episode_buffer = []
				episode_values = []
				episode_frames = []
				episode_reward = 0
				episode_step_count = 0
				d = False
				ros_rate = rospy.Rate(20)
				if(temp <2):				
					rospy.wait_for_message("/camera/color/image_raw", Image)
					rospy.wait_for_message("/camera/depth/image_raw", Image)
					rospy.Subscriber("/"+self.name+"/odom",Odometry)
					temp = 0				
				#self.env.new_episode()
				print("Getting Frame")
				img, ori, points = self.env.get_frame()
				print("Got Frame")
				episode_frames.append(img)
				s = process_frame(img)
				rnn_state = self.local_AC.state_init
				self.batch_rnn_state = rnn_state
				while self.env.done == False:
					#Take an action using probabilities from policy network output.
					#print("Input" + str(np.shape(s)))
					#print("RNN state 0" + str(np.shape(rnn_state[0])))
					#print("RNN state 1" + str(np.shape(rnn_state[1])))
					print("************EPISODE "+str(episode_count)+" TIMESTEP "+str(episode_step_count)+"***************")
					print("COMMAND: "+self.com)
					print("Average: "+str(mean(points[0])))
					#print("Input" + str(np.shape()))
				    	a_4,v,rnn_state = sess.run([self.local_AC.policy,self.local_AC.value,self.local_AC.state_out], 
				        	feed_dict={self.local_AC.inputs:s, self.local_AC.command_inp:self.input_com, self.local_AC.ori_inp:[[ori]], self.local_AC.point_cloud:points,
				        	self.local_AC.state_in[0]:rnn_state[0],
				        	self.local_AC.state_in[1]:rnn_state[1]})
					
					lin_vel = get_vels(a_4[0,0], a_4[0,1], "lin")
					ang_vel = get_vels(a_4[0,2], a_4[0,3], "ang")
					self.env.step(lin_vel, ang_vel)
					#get stop action from output, if stop action = True, change the action to get the future value	
					#print("Stop action shape: "+str(np.shape(stop_action)))
					print("Lin_vel: "+str(lin_vel))
					print("Ang_vel: "+str(ang_vel))
					stop_action = a_4[0,4]	
					print("STOP: "+str(stop_action))		    	
					if(stop_action < 0.5):
						stop_action = 0
					else:
						stop_action = 1

					print("STOP: "+str(stop_action))

					r = self.env.get_reward(lin_vel, ang_vel, stop_action)
					print("REWARD: "+str(r))
				    	d = self.env.done
				    	if d == False:
				        	s1,ori1, points1 = self.env.get_frame()
				        	episode_frames.append(s1)
				        	s1 = process_frame(s1)
				    	else:
				    	    	s1 = s
				        	ori1 = ori
						points1 = points

				    	episode_buffer.append([s,[lin_vel, ang_vel],r,s1,d,v[0,0],ori,self.input_com,points, self.env.at_target()])#stop_action])
				    	episode_values.append(v[0,0])

				    	episode_reward += r
				    	s = s1
					ori = ori1
					points = points1                    
				    	total_steps += 1
				    	episode_step_count += 1
				    
				    	# If the episode hasn't ended, but the experience buffer is full, then we
				    	# make an update step using that experience rollout.
				    	if len(episode_buffer) == 1000: #and d != True and episode_step_count < max_episode_length - 1:
				        	# Since we don't know what the true final return is, we "bootstrap" from our current
				        	# value estimation.
						print("UPDATING")
				        	v1 = sess.run(self.local_AC.value,
				        	    feed_dict={self.local_AC.inputs:s, self.local_AC.command_inp:self.input_com, self.local_AC.ori_inp:[[ori]], self.local_AC.point_cloud:points,
				        	    self.local_AC.state_in[0]:rnn_state[0],
				        	    self.local_AC.state_in[1]:rnn_state[1]})[0,0]
				        	v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,v1)
				        	episode_buffer = []
				        	sess.run(self.update_local_ops)
					if(episode_step_count >= max_episode_length):
						d=True
						#episode_reward = 0.0#-30.0
				    	if d == True:
						
						direct = Twist()
						direct.angular.x = 0.0
						direct.angular.y = 0.0
						direct.angular.z = 0.0
						direct.linear.x = 0.0
						direct.linear.y = 0.0
						direct.linear.z = 0.0
						self.env.command_pub.publish(direct)
						print("!!!!!!!!!!!!EPISODE "+str(episode_count)+" REWARD!!!!!!!!!!!!!!   " + str(episode_reward))
						post = "Episode "+str(episode_count)+" TIMESTEP "+str(episode_step_count)+" : "+str(episode_reward)+" Value-Loss: "+str(v_l)+"\n"
						f.write(post)
						sess.run(self.increment)
						episode_count=sess.run(self.global_episodes)
						rospy.sleep(1)
						self.env_init(1)
				        	break
					#rospy.spin()
					ros_rate.sleep
				if(episode_count % 25 == 0 and episode_count != 0):
					saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
					print ("Saved Model") 
				self.episode_rewards.append(episode_reward)
				self.episode_lengths.append(episode_step_count)
				self.episode_mean_values.append(np.mean(episode_values))
				
				# Update the network using the episode buffer at the end of the episode.
				if len(episode_buffer) != 0:
				    	v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,0.0)
				                
				    
				# Periodically save gifs of episodes, model parameters, and summary statistics.
				if episode_count % 5 == 0 and episode_count == -10:
					print("Saving Model")
				    	if self.name == 'worker_0' and episode_count % 25 == 0:
				        	time_per_step = 0.05
				        	images = np.array(episode_frames)
				        	make_gif(images,'./frames/image'+str(episode_count)+'.gif',
				            		duration=len(images)*time_per_step,true_image=True,salience=False)
				    	if episode_count % 250 == 0 and self.name == 'worker_0':
				        	saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
				        	print ("Saved Model")

				    	mean_reward = np.mean(self.episode_rewards[-5:])
				    	mean_length = np.mean(self.episode_lengths[-5:])
				    	mean_value = np.mean(self.episode_mean_values[-5:])
				    	summary = tf.Summary()
				    	summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
				    	summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
				    	summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
				    	summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
				    	summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
				    	summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
				    	summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
				    	summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
				    	self.summary_writer.add_summary(summary, episode_count)

				    	self.summary_writer.flush()
					if self.name == 'worker_0':
				    		sess.run(self.increment)
					episode_count += 1




if __name__ == '__main__':

	#initializing ROS node
	rospy.init_node('spawner',anonymous=True)
	rospy.wait_for_service("gazebo/delete_model")
	rospy.wait_for_service("gazebo/spawn_sdf_model")
	print("Got it.")
	delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
	spawn_model = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)

	#Reading model file to spawn
	with open("/home/vaibhav/src/Firmware/Tools/sitl_gazebo/models/create_camera/create_3.sdf", "r") as f:
		product_xml = f.read()

	#Setting parameters
	max_episode_length = 1000
	gamma = .99 # discount rate for advantage estimation and reward discounting
	s_size = 90000 # Observations are greyscale frames of 84 * 84 * 1
	a_size = 2 # Agent can move Left, Right, or Fire
	load_model = True
	model_path = './model5'
	data_dict = {}
	data_dict["P1"] = {"x": [3.77, 10.52], "y": [-5.05, 0.48], "commands": {"left": [1.98, -3,25], "right": [1.94, 2.07], "straight": [-4.47, -0.45]}}
	data_dict["P2"] = {"x": [3.71, 10.52], "y": [-10.8, -5.89], "commands": {"right": [2.0, -7.92], "straight": [-5.38, -10.18]}}
	#Pass data_dict, randomly pick a point, randomly pick x and y in given range, randomly pick one of the commnads, execute till target reached
	start_pose = [[0.0, 0.0]]
	target_pose =[[1.0, 0.0]]
	commands = ["left"]
	tf.reset_default_graph()

	if not os.path.exists(model_path):
		os.makedirs(model_path)
	    
	#Create a directory to save episode playback gifs to
	if not os.path.exists('./frames'):
		os.makedirs('./frames')

	with tf.device("/cpu:0"): 
	    	global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
	    	trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
	    	master_network = AC_Network(s_size,a_size,'global',None) # Generate global network
	    	num_workers = 1 #multiprocessing.cpu_count() # Set workers to number of available CPU threads
	    	workers = []
	    	# Create worker classes
	    	for i in range(num_workers):
			workers.append(Worker(i, s_size, a_size, trainer, model_path, global_episodes, product_xml, spawn_model, delete_model, data_dict))
	    	saver = tf.train.Saver(max_to_keep=5)

	with tf.Session() as sess:
	    	coord = tf.train.Coordinator()
	    	if load_model == True:
			print ('Loading Model...')
			ckpt = tf.train.get_checkpoint_state(model_path)
			print(ckpt.model_checkpoint_path)
			saver.restore(sess,ckpt.model_checkpoint_path)
	    	else:
			sess.run(tf.global_variables_initializer())
		
	    	# This is where the asynchronous magic happens.
	    	# Start the "work" process for each worker in a separate threat.
	    	worker_threads = []
	    	for worker in workers:
			worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
			t = threading.Thread(target=(worker_work))
			t.start()
			sleep(0.5)
			worker_threads.append(t)
	    	coord.join(worker_threads)
