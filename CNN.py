from __future__ import print_function
from scipy.signal import fftconvolve
import numpy as np
from scipy.special import softmax
from functools import partial
from .BaseClass import BaseClass



def get_Batch(n, bsize, randomize=False):
    
    '''Generator to iterate through a range in batches
    Args:
        n (int): range from 0 to n to iterate
        bsize (int): batch size.
    Keyword Args:
        randomize (bool): if True, iterate through randomly permuated range.
    Returns:
        idxii (list): list of integers with a length <= <bsize>.
    '''

    id1 = 0
    if randomize:
        idx = np.random.permutation(n)
    while True:
        id2 = id1+bsize
        id2 = min(id2, n)
        if randomize:
            idxii = idx[id1:id2]
        else:
            idxii = np.arange(id1, id2)
        if id1 > n-1:
            break
        yield idxii
        id1 += bsize

def crossEntropy(yhat, y):
    '''Cross entropy cost function '''
    eps = 1e-10
    yhat = np.clip(yhat, eps, 1-eps)
    aa = y*np.log(yhat)
    return -np.nansum(aa)

def ReLU(x):
    return np.maximum(x, 0)

def dReLU(x):
    '''Gradient of ReLU'''
    return 1.*(x > 0)


def pad_Array(var, pad1, pad2=None):
    '''Pad array with 0s
    Args:
        var (ndarray): 2d or 3d ndarray. Padding is done on the first 2 dimensions.
        pad1 (int): number of columns/rows to pad at left/top edges.
    Keyword Args:
        pad2 (int): number of columns/rows to pad at right/bottom edges.
            If None, same as <pad1>.
    Returns:
        var_pad (ndarray): 2d or 3d ndarray with 0s padded along the first 2
            dimensions.
    '''
    if pad2 is None:
        pad2 = pad1
    if pad1+pad2 == 0:
        return var
    var_pad = np.zeros(tuple(pad1+pad2+np.array(var.shape[:2])) + var.shape[2:])
    var_pad[pad1:-pad2, pad1:-pad2] = var

    return var_pad

def as_Stride(arr, sub_shape, stride):
    '''Get a strided sub-matrices view of an ndarray.
    Args:
        arr (ndarray): input array of rank 2 or 3, with shape (m1, n1) or (m1, n1, c).
        sub_shape (tuple): window size: (m2, n2).
        stride (int): stride of windows in both y- and x- dimensions.
    Returns:
        subs (view): strided window view.
    See also skimage.util.shape.view_as_windows()
    '''
    s0, s1 = arr.strides[:2]
    m1, n1 = arr.shape[:2]
    m2, n2 = sub_shape[:2]

    view_shape = (1+(m1-m2)//stride, 1+(n1-n2)//stride, m2, n2)+arr.shape[2:]
    strides = (stride*s0, stride*s1, s0, s1)+arr.strides[2:]
    subs = np.lib.stride_tricks.as_strided(
        arr, view_shape, strides=strides, writeable=False)

    return subs

def conv_3D3(var, kernel, stride=1, pad=0):
    '''3D convolution by strided view.
    Args:
        var (ndarray): 2d or 3d array to convolve along the first 2 dimensions.
        kernel (ndarray): 2d or 3d kernel to convolve. If <var> is 3d and <kernel>
            is 2d, create a dummy dimension to be the 3rd dimension in kernel.
    Keyword Args:
        stride (int): stride along the 1st 2 dimensions. Default to 1.
        pad (int): number of columns/rows to pad at edges.
    Returns:
        conv (ndarray): convolution result.
    '''
    kernel = checkShape(var, kernel)
    if pad > 0:
        var_pad = pad_Array(var, pad, pad)
    else:
        var_pad = var

    view = as_Stride(var_pad, kernel.shape, stride)
    if np.ndim(kernel) == 2:
        conv = np.sum(view*kernel, axis=(2, 3))
    else:
        conv = np.sum(view*kernel, axis=(2, 3, 4))

    return conv

def checkShape(var, kernel):
    '''Check shapes for convolution
    Args:
        var (ndarray): 2d or 3d input array for convolution.
        kernel (ndarray): 2d or 3d convolution kernel.
    Returns:
        kernel (ndarray): 2d kernel reshape into 3d if needed.
    '''
    var_ndim = np.ndim(var)
    kernel_ndim = np.ndim(kernel)

    if var_ndim not in [2, 3]:
        raise Exception("<var> dimension should be in 2 or 3.")
    if kernel_ndim not in [2, 3]:
        raise Exception("<kernel> dimension should be in 2 or 3.")
    if var_ndim < kernel_ndim:
        raise Exception("<kernel> dimension > <var>.")
    if var_ndim == 3 and kernel_ndim == 2:
        kernel = np.repeat(kernel[:, :, None], var.shape[2], axis=2)

    return kernel

def pick_Strided(var, stride):
    '''Pick sub-array by stride
    Args:
        var (ndarray): 2d or 3d ndarray.
        stride (int): stride/step along the 1st 2 dimensions to pick
            elements from <var>.
    Returns:
        result (ndarray): 2d or 3d ndarray picked at <stride> from <var>.
    '''
    if stride < 0:
        raise Exception("<stride> should be >=1.")
    if stride == 1:
        result = var
    else:
        result = var[::stride, ::stride, ...]
    return result

def conv_3D2(var, kernel, stride=1, pad=0):
    '''3D convolution by sub-matrix summing.
    Args:
        var (ndarray): 2d or 3d array to convolve along the first 2 dimensions.
        kernel (ndarray): 2d or 3d kernel to convolve. If <var> is 3d and <kernel>
            is 2d, create a dummy dimension to be the 3rd dimension in kernel.
    Keyword Args:
        stride (int): stride along the 1st 2 dimensions. Default to 1.
        pad (int): number of columns/rows to pad at edges.
    Returns:
        result (ndarray): convolution result.
    '''
    var_ndim = np.ndim(var)
    ny, nx = var.shape[:2]
    ky, kx = kernel.shape[:2]
    result = 0
    if pad > 0:
        var_pad = pad_Array(var, pad, pad)
    else:
        var_pad = var

    for ii in range(ky*kx):
        yi, xi = divmod(ii, kx)
        slabii = var_pad[yi:2*pad+ny-ky+yi+1:1,
                         xi:2*pad+nx-kx+xi+1:1, ...]*kernel[yi, xi]
        if var_ndim == 3:
            slabii = slabii.sum(axis=-1)
        result += slabii

    if stride > 1:
        result = pick_Strided(result, stride)

    return result

def conv_3D(var, kernel, stride=1, pad=0):
    '''3D convolution using scipy.signal.fftconvolve.
    Args:
        var (ndarray): 2d or 3d array to convolve along the first 2 dimensions.
        kernel (ndarray): 2d or 3d kernel to convolve. If <var> is 3d and <kernel>
            is 2d, create a dummy dimension to be the 3rd dimension in kernel.
    Keyword Args:
        stride (int): stride along the 1st 2 dimensions. Default to 1.
        pad (int): number of columns/rows to pad at edges.
    Returns:
        conv (ndarray): convolution result.
    '''
    stride = int(stride)
    kernel = checkShape(var, kernel)
    if pad > 0:
        var_pad = pad_Array(var, pad, pad)
    else:
        var_pad = var

    conv = fftconvolve(var_pad, kernel, mode='valid')

    if stride > 1:
        conv = pick_Strided(conv, stride)

    return conv

def interLeave(arr, sy, sx):
    '''Interleave array with rows/columns of 0s.
    Args:
        arr (ndarray): input 2d or 3d array to interleave in the first 2 dimensions.
        sy (int): number of rows to interleave.
        sx (int): number of columns to interleave.
    Returns:
        result (ndarray): input <arr> array interleaved with 0s.
 
    '''

    ny, nx = arr.shape[:2]
    shape = (ny+sy*(ny-1), nx+sx*(nx-1))+arr.shape[2:]
    result = np.zeros(shape)
    result[0::(sy+1), 0::(sx+1), ...] = arr
    return result

def compute_Size(n, f, s):
    '''Compute the shape of a full convolution result
    Args:
        n (int): length of input array x.
        f (int): length of kernel.
        s (int): stride.
    Returns:
        nout (int): lenght of output array y.
        pad_left (int): number padded to the left in a full convolution.
        pad_right (int): number padded to the right in a full convolution.
    E.g. x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    f = 3, s = 2.
    A full convolution is done on [*, *, 0], [0, 1, 2], [2, 3, 4], ..., [6, 7, 8],
    [9, 10, *]. Where * is missing outside of the input domain.
    Therefore, the full convolution y has length 6. pad_left = 2, pad_right = 1.
    '''

    nout = 1
    pad_left = f-1
    pad_right = 0
    idx = 0   
    while True:
        idx_next = idx+s
        win_left = idx_next-f+1
        if win_left <= n-1:
            nout += 1
            idx = idx+s
        else:
            break
    pad_right = idx-n+1

    return nout, pad_left, pad_right

def full_Conv3D(var, kernel, stride):
    '''Full mode 3D convolution using stride view.
    Args:
        var (ndarray): 2d or 3d array to convolve along the first 2 dimensions.
        kernel (ndarray): 2d or 3d kernel to convolve. If <var> is 3d and <kernel>
            is 2d, create a dummy dimension to be the 3rd dimension in kernel.
    Keyword Args:
        stride (int): stride along the 1st 2 dimensions. Default to 1.
    Returns:
        conv (ndarray): convolution result.
    Note that the kernel is not filpped inside this function.
    '''
    stride = int(stride)
    ny, nx = var.shape[:2]
    ky, kx = kernel.shape[:2]
    # interleave 0s
    var2 = interLeave(var, stride-1, stride-1)
    # pad boundaries
    nout, pad_left, pad_right = compute_Size(ny, ky, stride)
    var2 = pad_Array(var2, pad_left, pad_right)
    # convolve
    conv = conv_3D3(var2, kernel, stride=1, pad=0)

    return conv



def pooling(mat, f, method='max', pad=False):
    '''Non-overlapping pooling on 2D or 3D data.
    Args:
        mat (ndarray): input array to do pooling on the first 2 dimensions.
        f (int): pooling kernel size.
        method (str): 'max for max-pooling,
                      'mean' for average-pooling.
        pad (bool): pad <mat> or not. If true, pad <mat> at the end in
               y-axis with (f-n%f) number of nans, if not evenly divisible,
               similar for the x-axis.
    Returns:
        result (ndarray): pooled array.
    '''
    m, n = mat.shape[:2]
    _ceil = lambda x, y: x//y + 1

    if pad:
        ny = _ceil(m, f)
        nx = _ceil(n, f)
        size = (ny*f, nx*f)+mat.shape[2:]
        mat_pad = np.full(size, np.nan)
        mat_pad[:m, :n, ...] = mat
    else:
        ny = m//f
        nx = n//f
        mat_pad = mat[:ny*f, :nx*f, ...]

    new_shape = (ny, f, nx, f)+mat.shape[2:]

    if method == 'max':
        result = np.nanmax(mat_pad.reshape(new_shape), axis=(1, 3))
    else:
        result = np.nanmean(mat_pad.reshape(new_shape), axis=(1, 3))

    return result

def pooling_Overlap(mat, f, stride=None, method='max', pad=False,
                   return_max_pos=False):
    '''Overlapping pooling on 2D or 3D data.
    Args:
        mat (ndarray): input array to do pooling on the first 2 dimensions.
        f (int): pooling kernel size.
    Keyword Args:
        stride (int or None): stride in row/column. If None, same as <f>,
            i.e. non-overlapping pooling.
        method (str): 'max for max-pooling,
                      'mean' for average-pooling.
        pad (bool): pad <mat> or not. If true, pad <mat> at the end in
               y-axis with (f-n%f) number of nans, if not evenly divisible,
               similar for the x-axis.
        return_max_pos (bool): whether to return an array recording the locations
            of the maxima if <method>=='max'. This could be used to back-propagate
            the errors in a network.
    Returns:
        result (ndarray): pooled array.
    See also unpooling().
    '''
    m, n = mat.shape[:2]
    if stride is None:
        stride = f
    _ceil = lambda x, y: x//y + 1

    if pad:
        ny = _ceil(m, stride)
        nx = _ceil(n, stride)
        size = ((ny-1)*stride+f, (nx-1)*stride+f) + mat.shape[2:]
        mat_pad = np.full(size, 0)
        mat_pad[:m, :n, ...] = mat
    else:
        mat_pad = mat[:(m-f)//stride*stride+f, :(n-f)//stride*stride+f, ...]

    view = as_Stride(mat_pad, (f, f), stride)
    if method == 'max':
        result = np.nanmax(view, axis=(2, 3), keepdims=return_max_pos)
    else:
        result = np.nanmean(view, axis=(2, 3), keepdims=return_max_pos)

    if return_max_pos:
        pos = np.where(result == view, 1, 0)
        result = np.squeeze(result)
        return result, pos
    else:
        return result

def unpooling(mat, pos, ori_shape, stride):
    '''Undo a max-pooling of a 2d or 3d array to a larger size
    Args:
        mat (ndarray): 2d or 3d array to unpool on the first 2 dimensions.
        pos (ndarray): array recording the locations of maxima in the original
            array. If <mat> is 2d, <pos> is 4d with shape (iy, ix, cy, cx).
            Where iy/ix are the numbers of rows/columns in <mat>,
            cy/cx are the sizes of the each pooling window.
            If <mat> is 3d, <pos> is 5d with shape (iy, ix, cy, cx, cc).
            Where cc is the number of channels in <mat>.
        ori_shape (tuple): original shape to unpool to.
        stride (int): stride used during the pooling process.
    Returns:
        result (ndarray): <mat> unpoolled to shape <ori_shape>.
    '''
    assert np.ndim(pos) in [4, 5], '<pos> should be rank 4 or 5.'
    result = np.zeros(ori_shape)

    if np.ndim(pos) == 5:
        iy, ix, cy, cx, cc = np.where(pos == 1)
        iy2 = iy*stride
        ix2 = ix*stride
        iy2 = iy2+cy
        ix2 = ix2+cx
        values = mat[iy, ix, cc].flatten()
        result[iy2, ix2, cc] = values
    else:
        iy, ix, cy, cx = np.where(pos == 1)
        iy2 = iy*stride
        ix2 = ix*stride
        iy2 = iy2+cy
        ix2 = ix2+cx
        values = mat[iy, ix].flatten()
        result[iy2, ix2] = values

    return result

def unpooling_Average(mat, f, ori_shape):
    '''Undo an average-pooling of a 2d or 3d array to a larger size
    Args:
        mat (ndarray): 2d or 3d array to unpool on the first 2 dimensions.
        f (int): pooling kernel size.
        ori_shape (tuple): original shape to unpool to.
    Returns:
        result (ndarray): <mat> unpoolled to shape <ori_shape>.
    '''
    m, n = ori_shape[:2]
    ny = m//f
    nx = n//f

    tmp = np.reshape(mat, (ny, 1, nx, 1)+mat.shape[2:])
    tmp = np.repeat(tmp, f, axis=1)
    tmp = np.repeat(tmp, f, axis=3)
    tmp = np.reshape(tmp, (ny*f, nx*f)+mat.shape[2:])
    result=np.zeros(ori_shape)

    result[:tmp.shape[0], :tmp.shape[1], ...]= tmp

    return result







def force2D(x):
    '''Force rank 1 array to 2d to get a column vector'''
    return np.reshape(x, (x.shape[0], -1))

def force3D(x):
    return np.atleast_3d(x)




class ConvLayer(object):
    def __init__(self, f, pad, stride, nc_in, nc, learning_rate, af=None, lam=0.01,
            clipvalue=0.5):
        '''Convolutional layer
        Args:
            f (int): kernel size for height and width.
            pad (int): padding on each edge.
            stride (int): convolution stride.
            nc_in (int): number of channels from input layer.
            nc (int): number of channels this layer.
            learning_rate (float): initial learning rate.
        Keyword Args:
            af (callable): activation function. Default to ReLU.
            lam (float): regularization parameter.
            clipvalue (float): clip gradients within [-clipvalue, clipvalue]
                during back-propagation.
        The layer has <nc> channels/filters. Each filter has shape (f, f, nc_in).
        The <nc> filters are saved in a list `self.filters`, therefore `self.filters[0]`
        corresponds to the 1st filter.
        Bias is saved in `self.biases`, which is a 1d array of length <nc>.
        '''
        self.type = 'conv'
        self.f = f
        self.pad = pad
        self.stride = stride
        self.lr = learning_rate
        self.nc_in = nc_in
        self.nc = nc
        if af is None:
            self.af = ReLU
        else:
            self.af = af
        self.lam = lam
        self.clipvalue = clipvalue
        self.init()
    def init(self):
        '''Initialize weights
        Default to use HE initialization:
            w ~ N(0, std)
            std = \sqrt{2 / n}
        where n is the number of inputs
        '''
        np.random.seed(100)
        std = np.sqrt(2/self.f**2/self.nc_in)
        self.filters = np.array([
            np.random.normal(0, scale=std, size=[self.f, self.f, self.nc_in])
            for i in range(self.nc)])
        self.biases = np.random.normal(0, std, size=self.nc)
    @property
    def n_params(self):
        '''Number of parameters in layer'''
        n_filters = self.filters.size
        n_biases = self.nc
        return n_filters + n_biases
    def forward(self, x):
        '''Forward pass of a single image input'''
        x = force3D(x)
        x = pad_Array(x, self.pad, self.pad)
        # input size:
        ny, nx, nc = x.shape
        # output size:
        oy = (ny+2*self.pad-self.f)//self.stride + 1
        ox = (nx+2*self.pad-self.f)//self.stride + 1
        oc = self.nc
        weight_sums = np.zeros([oy, ox, oc])
        # loop through filters
        for ii in range(oc):
            slabii = conv_3D3(x, self.filters[ii], stride=self.stride, pad=0)
            weight_sums[:, :, ii] = slabii[:, :]
        # add bias
        weight_sums = weight_sums+self.biases
        # activate func
        activations = self.af(weight_sums)
        return weight_sums, activations
    def back_Prop_Error(self, delta_in, z):
        '''Back-propagate errors
        Args:
            delta_in (ndarray): delta from the next layer in the network.
            z (ndarray): weighted sum of the current layer.
        Returns:
            result (ndarray): delta of the current layer.
        The theoretical equation for error back-propagation is:
            \delta^{(l)} = \delta^{(l+1)} \bigotimes_f Rot(W^{(l+1)}) \bigodot f'(z^{(l)})
        where:
            \delta^{(l)} : error of layer l, defined as \partial J / \partial z^{(l)}.
            \bigotimes_f : convolution in full mode.
            Rot() : is rotating the filter by 180 degrees, i.e. a kernel flip.
            W^{(l+1)} : weights of layer l+1.
            \bigodot : Hadamard (elementwise) product.
            f() : activation function of layer l.
            z^{(l)} : weighted sum in layer l.
        Computation in practice is more complicated than the above equation.
        '''
        # number of channels of input to layer l weights
        nc_pre = z.shape[-1]
        # number of channels of output from layer l weights
        nc_next = delta_in.shape[-1]
        result = np.zeros_like(z)
        # loop through channels in layer l
        for ii in range(nc_next):
            # flip the kernel
            kii = self.filters[ii, ::-1, ::-1, ...]
            deltaii = delta_in[:, :, ii]
            # loop through channels of input
            for jj in range(nc_pre):
                slabij = full_Conv3D(deltaii, kii[:, :, jj], self.stride)
                result[:, :, jj] += slabij
        result = result*dReLU(z)
        return result
    def compute_Gradients(self, delta, act):
        '''Compute gradients of cost wrt filter weights
        Args:
            delta (ndarray): errors in filter ouputs.
            act (ndarray): activations fed into filter.
        Returns:
            grads (ndarray): gradients of filter weights.
            grads_bias (ndarray): 1d array, gradients of biases.
        The theoretical equation of gradients of filter weights is:
            \partial J / \partial W^{(l)} = a^{(l-1)} \bigotimes \delta^{(l)}
        where:
            J : cost function of network.
            W^{(l)} : weights in filter.
            a^{(l-1)} : activations fed into filter.
            \bigotimes : convolution in valid mode.
            \delta^{(l)} : errors in the outputs from the filter.
        Computation in practice is more complicated than the above equation.
        '''
        nc_out = delta.shape[-1]   # number of channels in outputs
        nc_in = act.shape[-1]      # number of channels in inputs
        grads = np.zeros_like(self.filters)
        for ii in range(nc_out):
            deltaii = np.take(delta, ii, axis=-1)
            gii = grads[ii]
            for jj in range(nc_in):
                actjj = act[:, :, jj]
                gij = conv_3D3(actjj, deltaii, stride=1, pad=0)
                gii[:, :, jj] += gij
            grads[ii] = gii
        # gradient clip
        gii = np.clip(gii, -self.clipvalue, self.clipvalue)
        grads_bias = np.sum(delta, axis=(0, 1))  # 1d
        return grads, grads_bias
    def gradient_Descent(self, grads, grads_bias, m):
        '''Gradient descent weight and bias update'''
        self.filters = self.filters * (1 - self.lr * self.lam/m) - self.lr * grads/m
        self.biases = self.biases-self.lr*grads_bias/m
        return




class PoolLayer(object):
    def __init__(self, f, pad, stride, method='max'):
        '''Pooling layer
        Args:
            f (int): kernel size for height and width.
            pad (int): padding on each edge.
            stride (int): pooling stride. Required to be the same as <f>, i.e.
                non-overlapping pooling.
        Keyword Args:
            method (str): pooling method. 'max' for max-pooling.
                'mean' for average-pooling.
        '''
        self.type = 'pool'
        self.f = f
        self.pad = pad
        self.stride = stride
        self.method = method
        if method != 'max':
            raise Exception("Method %s not implemented" % method)
        if self.f != self.stride:
            raise Exception("Use equal <f> and <stride>.")
    @property
    def n_params(self):
        return 0
    def forward(self, x):
        '''Forward pass'''
        x = force3D(x)
        result, max_pos = pooling_Overlap(x, self.f, stride=self.stride,
            method=self.method, pad=False, return_max_pos=True)
        self.max_pos = max_pos  # record max locations
        return result, result
    def back_Prop_Error(self, delta_in, z):
        '''Back-propagate errors
        Args:
            delta_in (ndarray): delta from the next layer in the network.
            z (ndarray): weighted sum of the current layer.
        Returns:
            result (ndarray): delta of the current layer.
        For max-pooling, each error in <delta_in> is assigned to where it came
        from in the input layer, and other units get 0 error. This is achieved
        with the help of recorded maximum value locations.
        For average-pooling, the error in <delta_in> is divided by the kernel
        size and assigned to the whole pooling block, i.e. even distribution
        of the errors.
        '''
        result = unpooling(delta_in, self.max_pos, z.shape, self.stride)
        return result





class FlattenLayer(object):
    def __init__(self, input_shape):
        '''Flatten layer'''
        self.type = 'flatten'
        self.input_shape = input_shape
    @property
    def n_params(self):
        return 0
    def forward(self, x):
        '''Forward pass'''
        x = x.flatten()
        return x, x
    def back_Prop_Error(self, delta_in, z):
        '''Back-propagate errors
        '''
        result = np.reshape(delta_in, tuple(self.input_shape))
        return result





class FCLayer(object):
    def __init__(self, n_inputs, n_outputs, learning_rate, af=None, lam=0.01,
            clipvalue=0.5):
        '''Fully-connected layer
        Args:
            n_inputs (int): number of inputs.
            n_outputs (int): number of layer outputs.
            learning_rate (float): initial learning rate.
        Keyword Args:
            af (callable): activation function. Default to ReLU.
            lam (float): regularization parameter.
            clipvalue (float): clip gradients within [-clipvalue, clipvalue]
                during back-propagation.
        '''
        self.type = 'fc'
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.lr = learning_rate
        self.clipvalue = clipvalue
        if af is None:
            self.af = ReLU
        else:
            self.af = af
        self.lam = lam   # regularization parameter
        self.init()
    @property
    def n_params(self):
        return self.n_inputs*self.n_outputs + self.n_outputs
    def init(self):
        '''Initialize weights
        Default to use HE initialization:
            w ~ N(0, std)
            std = \sqrt{2 / n}
        where n is the number of inputs
        '''
        std = np.sqrt(2/self.n_inputs)
        np.random.seed(100)
        self.weights = np.random.normal(0, std, size=[self.n_outputs, self.n_inputs])
        self.biases = np.random.normal(0, std, size=self.n_outputs)
    def forward(self, x):
        '''Forward pass'''
        z = np.dot(self.weights, x)+self.biases
        a = self.af(z)
        return z, a
    def back_Prop_Error(self, delta_in, z):
        '''Back-propagate errors
        Args:
            delta_in (ndarray): delta from the next layer in the network.
            z (ndarray): weighted sum of the current layer.
        Returns:
            result (ndarray): delta of the current layer.
        The theoretical equation for error back-propagation is:
            \delta^{(l)} = W^{(l+1)}^{T} \cdot \delta^{(l+1)} \bigodot f'(z^{(l)})
        where:
            \delta^{(l)} : error of layer l, defined as \partial J / \partial z^{(l)}.
            W^{(l+1)} : weights of layer l+1.
            \bigodot : Hadamard (elementwise) product.
            f() : activation function of layer l.
            z^{(l)} : weighted sum in layer l.
        '''
        result = np.dot(self.weights.T, delta_in)*dReLU(z)
        return result
    def compute_Gradients(self, delta, act):
        '''Compute gradients of cost wrt weights
        Args:
            delta (ndarray): errors in ouputs.
            act (ndarray): activations fed into weights.
        Returns:
            grads (ndarray): gradients of weights.
            grads_bias (ndarray): 1d array, gradients of biases.
        The theoretical equation of gradients of filter weights is:
            \partial J / \partial W^{(l)} = \delta^{(l)} \cdot a^{(l-1)}^{T}
        where:
            J : cost function of network.
            W^{(l)} : weights in layer.
            a^{(l-1)} : activations fed into the weights.
            \delta^{(l)} : errors in the outputs from the weights.
        When implemented, had some trouble getting the shape correct. Therefore
        used einsum().
        '''
        #grads = np.einsum('ij,kj->ik', delta, act)
        grads = np.outer(delta, act)
        # gradient-clip
        grads = np.clip(grads, -self.clipvalue, self.clipvalue)
        grads_bias = np.sum(delta, axis=-1)
        return grads, grads_bias
    def gradient_Descent(self, grads, grads_bias, m):
        '''Gradient descent weight and bias update'''
        self.weights = self.weights * (1 - self.lr * self.lam/m) - self.lr * grads/m
        self.biases = self.biases-self.lr*grads_bias/m
        return




class CNNClassifier(object):

    def __init__(self, cost_func=None):
        '''CNN classifier
        Keyword Args:
            cost_func (callable or None): cost function. If None, use cross-
                entropy cost.
        '''

        self.layers = []

        if cost_func is None:
            self.cost_func = crossEntropy
        else:
            self.cost_func = cost_func

    @property
    def n_layers(self):
        '''Number of layers in network'''
        return len(self.layers)

    @property
    def n_params(self):
        '''Total number of trainable parameters of all layers in network'''
        result = 0
        for ll in self.layers:
            result += ll.n_params
        return result


    def add(self, layer):
        '''Add new layers to the network
        Args:
            layer (ConvLayer|PoolLayer|FCLayer): a ConvLayer, PoolLayer or FCLayer
                object.
        '''
        self.layers.append(layer)


    def feed_ForwardProp(self, x):
        '''Forward pass of a single record
        Args:
            x (ndarray): input with shape (h, w) or (h, w, c). h is the image
                height, w the image width and c the number of channels.
        Returns:
            weight_sums (dict): keys: layer indices starting from 0 for
                the input layer to N-1 for the last layer. Values:
                weighted sums of each layer:
                    z^{(l+1)} = a^{(l)} \cdot \theta^{(l+1)}^{T} + b^{(l+1)}
                where:
                    z^{(l+1)}: weighted sum in layer l+1.
                    a^{(l)}: activation in layer l.
                    \theta^{(l+1)}: weights that map from layer l to l+1.
                    b^{(l+1)}: biases added to layer l+1.
                The value for key=0 is the same as input <x>.
            activations (dict): keys: layer indices starting from 0 for
                the input layer to N-1 for the last layer. Values:
                    activations in each layer. See above.
        '''

        activations = {0: x}
        weight_sums = {0: x}
        a1 = x
        for ii in range(self.n_layers):
            lii = self.layers[ii]
            zii, aii = lii.forward(a1)
            activations[ii+1] = aii
            weight_sums[ii+1] = zii
            a1 = aii

        return weight_sums, activations

    def feed_BackwardProp(self, weight_sums, activations, y):
        '''Backward propogation for a single record
        Args:
            weight_sums (dict): keys: layer indices starting from 0 for
                the input layer to N-1 for the last layer. Values:
                weighted sums of each layer.
            activations (dict): keys: layer indices starting from 0 for
                the input layer to N-1 for the last layer. Values:
                    activations in each layer.
            y (ndarray): label in shape (m,). m is the number of
                final output units.
        Returns:
            grads (dict): keys: layer indices starting from 0 for
                the input layer to N-1 for the last layer. Values:
                summed gradients for the weight matrix in each layer.
            grads_bias (dict): keys: layer indices starting from 0 for
                the input layer to N-1 for the last layer. Values:
                summed gradients for the bias in each layer.
        '''

        delta = activations[self.n_layers] - y
        grads={}
        grads_bias={}

        for jj in range(self.n_layers, 0, -1):
            layerjj = self.layers[jj-1]
            if layerjj.type in ['fc', 'conv']:
                gradjj, biasjj = layerjj.compute_Gradients(delta, activations[jj-1])
                grads[jj-1]=gradjj
                grads_bias[jj-1]=biasjj

            delta = layerjj.back_Prop_Error(delta, weight_sums[jj-1])

        return grads, grads_bias


    def sample_Cost(self, yhat, y):
        '''Cost of a single training sample
        Args:
            yhat (ndarray): prediction in shape (m,). m is the number of
                final output units.
            y (ndarray): label in shape (m,).
        Returns:
            cost (float): summed cost.
        '''
        j = self.cost_func(yhat, y)
        return j

    def regularization_Cost(self):
        '''Cost from the regularization term
        Defined as the summed squared weights in all layers, not including
        biases.
        '''
        j = 0
        for lii in self.layers:
            if hasattr(lii, 'filters'):
                wii = lii.filters
                jii = np.sum([np.sum(ii**2) for ii in wii])
            elif hasattr(lii, 'weights'):
                wii = lii.weights
                jii = np.sum(wii**2)
            j = j+jii

        return j


    def gradient_Descent(self, grads, grads_bias, n):
        '''Perform gradient descent parameter update
        Args:
            grads (dict): keys: layer indices starting from 0 for
                the input layer to N-1 for the last layer. Values:
                summed gradients for the weight matrix in each layer.
            grads_bias (dict): keys: layer indices starting from 0 for
                the input layer to N-1 for the last layer. Values:
                summed gradients for the bias in each layer.
            n (int): number of records contributing to the gradient computation.
        Update rule:
            \theta_i = \theta_i - \alpha (g_i + \lambda \theta_i)
        where:
            \theta_i: weight i.
            \alpha: learning rate.
            \g_i: gradient of weight i.
            \lambda: regularization parameter.
        '''

        for ii, layerii in enumerate(self.layers):
            if layerii.type in ['fc', 'conv']:
                gradii = grads[ii]
                grad_biasii = grads_bias[ii]
                layerii.gradient_Descent(gradii, grad_biasii, n)
        return

    def stochastic_Train(self, x, y, epochs):
        '''Stochastic training
        Args:
            x (ndarray): input with shape (n, h, w) or (n, h, w, c).
                n is the number of records.
                h the image height, w the image width and c the number of channels.
            y (ndarray): input with shape (n, m). m is the number of output units,
                n is the number of records.
            epochs (int): number of epochs to train.
        Returns:
            self.costs (ndarray): overall cost at each epoch.
        '''
        costs = []
        m = len(x)
        for ee in range(epochs):
            idxs = np.random.permutation(m)
            for ii in idxs:
                xii = np.atleast_3d(x[ii])
                yii = y[ii]
                weight_sums, activations = self.feed_ForwardProp(xii)
                gradsii, grads_biasii = self.feed_BackwardProp(weight_sums, activations, yii)
                self.gradient_Descent(gradsii, grads_biasii, 1)

            je = self.evaluate_Cost(x, y)
            print('# <stochastic_Train>: cost at epoch %d, j = %f' % (ee, je))
            costs.append(je)

        return np.array(costs)

    def sum_Gradients(self, g1, g2):
        '''Add gradients in two dicts'''
        result=dict([(k, g1[k]+g2[k]) for k in g1.keys()])
        return result

    def batch_train(self, x, y, epochs, batch_size):
        '''Training using mini batches
        Args:
            x (ndarray): input with shape (n, h, w) or (n, h, w, c).
                n is the number of records.
                h the image height, w the image width and c the number of channels.
            y (ndarray): input with shape (n, m). m is the number of output units,
                n is the number of records.
            epochs (int): number of epochs to train.
            batch_size (int): mini-batch size.
        Returns:
            self.costs (ndarray): overall cost at each epoch.
        '''
        costs = []
        m = len(x)
        for ee in range(epochs):
            batches = get_Batch(m, batch_size, randomize=True)
            for idxjj in batches:
                for idxii in idxjj:
                    xii = np.atleast_3d(x[idxii])
                    yii = y[idxii]
                    weight_sums, activations = self.feed_ForwardProp(xii)
                    gradsii, grads_biasii = self.feed_BackwardProp(weight_sums, activations, yii)

                    if idxii == idxjj[0]:
                        gradsjj = gradsii
                        grads_biasjj = grads_biasii
                    else:
                        gradsjj = self.sum_Gradients(gradsjj, gradsii)
                        grads_biasjj = self.sum_Gradients(grads_biasjj, grads_biasii)

                self.gradient_Descent(gradsjj, grads_biasjj, batch_size)

            je = self.evaluate_Cost(x, y)
            print('# <batch_train>: cost at epoch %d, j = %f' % (ee, je))
            costs.append(je)

        return np.array(costs)


    def predict(self, x):
        '''Model prediction
        Args:
            x (ndarray): input with shape (h, w) or (h, w, c). h is the image
                height, w the image width and c the number of channels.
        Returns:
            yhat (ndarray): prediction with shape (m,). m is the number of output units.
        '''

        x = np.atleast_3d(x)
        weight_sums, activations = self.feed_ForwardProp(x)
        yhat = activations[self.n_layers]
        return yhat


    def evaluate_Cost(self, x, y):
        '''Compute mean cost on a dataset
        Args:
            x (ndarray): input with shape (n, h, w) or (n, h, w, c).
                n is the number of records.
                h the image height, w the image width and c the number of channels.
            y (ndarray): input with shape (n, m). m is the number of output units,
                n is the number of records.
        Returns:
            j (float): mean cost over dataset <x, y>.
        '''
        j = 0
        n = len(x)
        for ii in range(n):
            yhatii = self.predict(x[ii])
            yii = y[ii]
            jii = self.sample_Cost(yhatii, yii)
            j += jii
        j2 = self.regularization_Cost()
        j += j2
        return j/n









