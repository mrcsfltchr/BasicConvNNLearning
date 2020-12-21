import numpy as np

##################################################################
# Defines various neural network layer objects for learning purposes
##################################################################

##################################################################
# Determining output shape  of 2D convolutional layer



# we can work out the shape of the output layer by using the input size, the layer settings, and the kernel size.

# output height = (input height - kernel height + 2*padding_no)/stride + 1

# output width = ^^^^^

# channel dimension = number of kernels

##################################################################

# Conv2D is a layer object, it has natural possessions i.e its weights. It can be initialised by a shape and a 
# stride as well as some flags, e.g padding etc.


class Conv2D(object):
    
    def __init__(self, shape, stride, filters , padding,num_channels = 1):
        
        self.shape = shape
        self.stride = stride
        self.filters = filters
        
        self.weights = np.random.rand(filters, self.shape[0]*self.shape[1]*num_channels)
        
        self.padding = padding
        
        
    def __call__(self,x):
        
        # first make patch tensor
        
        X , output_height, output_width = self.create_patch_tensor(x)
        
        # now the patch tensor is created we can matrix multiply it with the kernel (weights)

        out = X.T.dot(self.weights.T)
        
        out = out.reshape(output_height,output_width,self.filters)
        
        return out
    
        
    def create_patch_tensor(self,x):
        
        # x: image of shape (height,width,channels)
        
        
        # first pad the input with zeros
        
        num_channels = x.shape[2]
        
        
        if self.padding is not None:
            
            x_padded = np.empty((x.shape[0]+2*self.padding[0],x.shape[1]+2*self.padding[1],num_channels))
            
            for n in range(num_channels):
                x_padded[:,:,n] = np.pad(x[:,:,n],self.padding,'constant')
        

        # determine output dimensions
        
        output_height = (x_padded.shape[0] - self.shape[0])//self.stride +1
        output_width  = (x_padded.shape[1] - self.shape[1])//self.stride +1
        
        
        
        
        # then create patch tensor 
        
        X = np.empty((self.shape[0]*self.shape[1]*num_channels, output_height*output_width))


        for i in np.arange(0,x_padded.shape[0]-(self.shape[0]-1),self.stride):
            for j in range(0,x_padded.shape[1] -(self.shape[1]-1),self.stride):
                patch = x_padded[i:i+self.shape[0],j:j+self.shape[1],:].flatten()

                X[:,i*(x_padded.shape[1]-(self.shape[1]-1))+j] = patch
                
        
        return X, output_height, output_width
        
        
        
  