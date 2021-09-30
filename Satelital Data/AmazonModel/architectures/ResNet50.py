#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import applications

class MyModel(tf.keras.Model):

    def __init__(self, n_outputs=7, pretrained=False, freeze=False, size = 256, depth = 3):
        
        super(MyModel, self).__init__()
        
        
        if pretrained:
            self.model_weights = 'imagenet'
        else:
            self.model_weights = None
        
        # Download the architecture of ResNet50 with ImageNet weights
        self.resnet = applications.resnet50.ResNet50(include_top=False, weights=self.model_weights, input_shape= (size,size, depth))
        
        # Taking the output of the last convolution block in ResNet50
        self.res_out = self.resnet.output
        self.res_in = self.resnet.input
        
        self.GlobPoll = GlobalAveragePooling2D()
        
        # Adding a fully connected layer having 1024 neurons
        #self.fc1 = Dense(1024, activation='relu')
        
        # Sigmoid Out
        self.out = Dense(n_outputs, activation='sigmoid')
        
        if freeze:
            # Training only top layers i.e. the layers which we have added in the end
            self.resnet.trainable = False

    def call(self, inputs):

        x = self.resnet(inputs)
        x = self.GlobPoll(x)
        #x = self.fc1(x)
        x = self.out(x)
        
        return x



# In[ ]:




