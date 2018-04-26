import math
import unittest
import functools
from copy import deepcopy
import torch
import torch.optim as optim
import torch.legacy.optim as old_optim
import torch.nn.functional as F
from torch.optim import SGD
from torch.autograd import Variable
from torch import sparse
from test_common import TestCase, run_tests
import p03_layers
import torch.nn as nn


def rosenbrock(tensor):
    x, y = tensor
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def drosenbrock(tensor):
    x, y = tensor
    return torch.DoubleTensor((-400 * x * (y - x ** 2) - 2 * (1 - x), 200 * (y - x ** 2)))


def wrap_old_fn(old_fn, **config):
    def wrapper(closure, params, state):
        return old_fn(closure, params, config, state)
    return wrapper


class TestOptim(TestCase):

    def _test_rosenbrock(self, constructor, old_fn):
        params_t = torch.Tensor([1.5, 1.5])
        state = {}

        params = Variable(torch.Tensor([1.5, 1.5]), requires_grad=True)
        optimizer = constructor([params])

        solution = torch.Tensor([1, 1])
        initial_dist = params.data.dist(solution)

        def eval():
            optimizer.zero_grad()
            loss = rosenbrock(params)
            loss.backward()
            # loss.backward() will give **slightly** different
            # gradients, than drosenbtock, because of a different ordering
            # of floating point operations. In most cases it doesn't matter,
            # but some optimizers are so sensitive that they can temporarily
            # diverge up to 1e-4, just to converge again. This makes the
            # comparison more stable.
            params.grad.data.copy_(drosenbrock(params.data))
            return loss

        for i in range(2000):
            optimizer.step(eval)
            old_fn(lambda _: (rosenbrock(params_t), drosenbrock(params_t)),
                   params_t, state)
            self.assertEqual(params.data, params_t)

        self.assertLessEqual(params.data.dist(solution), initial_dist)

    def _test_rosenbrock_sparse(self, constructor, sparse_only=False):
        params_t = torch.Tensor([1.5, 1.5])

        params = Variable(params_t, requires_grad=True)
        optimizer = constructor([params])

        if not sparse_only:
            params_c = Variable(params_t.clone(), requires_grad=True)
            optimizer_c = constructor([params_c])

        solution = torch.Tensor([1, 1])
        initial_dist = params.data.dist(solution)

        def eval(params, sparse_grad, w):
            # Depending on w, provide only the x or y gradient
            optimizer.zero_grad()
            loss = rosenbrock(params)
            loss.backward()
            grad = drosenbrock(params.data)
            # NB: We torture test the optimizer by returning an
            # uncoalesced sparse tensor
            if w:
                i = torch.LongTensor([[0, 0]])
                x = grad[0]
                v = torch.DoubleTensor([x / 4., x - x / 4.])
            else:
                i = torch.LongTensor([[1, 1]])
                y = grad[1]
                v = torch.DoubleTensor([y - y / 4., y / 4.])
            x = sparse.DoubleTensor(i, v, torch.Size([2]))
            if sparse_grad:
                params.grad.data = x
            else:
                params.grad.data = x.to_dense()
            return loss

        for i in range(2000):
            # Do cyclic coordinate descent
            w = i % 2
            optimizer.step(functools.partial(eval, params, True, w))
            if not sparse_only:
                optimizer_c.step(functools.partial(eval, params_c, False, w))
                self.assertEqual(params.data, params_c.data)

        self.assertLessEqual(params.data.dist(solution), initial_dist)

    def _test_basic_cases_template(self, weight, bias, input, constructor):
        weight = Variable(weight, requires_grad=True)
        bias = Variable(bias, requires_grad=True)
        input = Variable(input)
        optimizer = constructor(weight, bias)

        # to check if the optimizer can be printed as a string
        optimizer.__repr__()

        def fn():
            optimizer.zero_grad()
            y = weight.mv(input)
            if y.is_cuda and bias.is_cuda and y.get_device() != bias.get_device():
                y = y.cuda(bias.get_device())
            loss = (y + bias).pow(2).sum()
            loss.backward()
            return loss

        initial_value = fn().data[0]
        for i in range(200):
            optimizer.step(fn)
        self.assertLess(fn().data[0], initial_value)

    def _test_state_dict(self, weight, bias, input, constructor):
        weight = Variable(weight, requires_grad=True)
        bias = Variable(bias, requires_grad=True)
        input = Variable(input)

        def fn_base(optimizer, weight, bias):
            optimizer.zero_grad()
            i = input_cuda if weight.is_cuda else input
            loss = (weight.mv(i) + bias).pow(2).sum()
            loss.backward()
            return loss

        optimizer = constructor(weight, bias)
        fn = functools.partial(fn_base, optimizer, weight, bias)

        # Prime the optimizer
        for i in range(20):
            optimizer.step(fn)
        # Clone the weights and construct new optimizer for them
        weight_c = Variable(weight.data.clone(), requires_grad=True)
        bias_c = Variable(bias.data.clone(), requires_grad=True)
        optimizer_c = constructor(weight_c, bias_c)
        fn_c = functools.partial(fn_base, optimizer_c, weight_c, bias_c)
        # Load state dict
        state_dict = deepcopy(optimizer.state_dict())
        state_dict_c = deepcopy(optimizer.state_dict())
        optimizer_c.load_state_dict(state_dict_c)
        # Run both optimizations in parallel
        for i in range(20):
            optimizer.step(fn)
            optimizer_c.step(fn_c)
            self.assertEqual(weight, weight_c)
            self.assertEqual(bias, bias_c)
        # Make sure state dict wasn't modified
        self.assertEqual(state_dict, state_dict_c)

        # Check that state dict can be loaded even when we cast parameters
        # to a different type and move to a different device.
        if not torch.cuda.is_available():
            return

        input_cuda = Variable(input.data.float().cuda())
        weight_cuda = Variable(weight.data.float().cuda(), requires_grad=True)
        bias_cuda = Variable(bias.data.float().cuda(), requires_grad=True)
        optimizer_cuda = constructor(weight_cuda, bias_cuda)
        fn_cuda = functools.partial(fn_base, optimizer_cuda, weight_cuda, bias_cuda)

        state_dict = deepcopy(optimizer.state_dict())
        state_dict_c = deepcopy(optimizer.state_dict())
        optimizer_cuda.load_state_dict(state_dict_c)

        # Make sure state dict wasn't modified
        self.assertEqual(state_dict, state_dict_c)

        for i in range(20):
            optimizer.step(fn)
            optimizer_cuda.step(fn_cuda)
            self.assertEqual(weight, weight_cuda)
            self.assertEqual(bias, bias_cuda)

    def _test_basic_cases(self, constructor, ignore_multidevice=False):
        self._test_state_dict(
            torch.randn(10, 5),
            torch.randn(10),
            torch.randn(5),
            constructor
        )
        self._test_basic_cases_template(
            torch.randn(10, 5),
            torch.randn(10),
            torch.randn(5),
            constructor
        )
        # non-contiguous parameters
        self._test_basic_cases_template(
            torch.randn(10, 5, 2)[..., 0],
            torch.randn(10, 2)[..., 0],
            torch.randn(5),
            constructor
        )
        # CUDA
        if not torch.cuda.is_available():
            return
        self._test_basic_cases_template(
            torch.randn(10, 5).cuda(),
            torch.randn(10).cuda(),
            torch.randn(5).cuda(),
            constructor
        )
        # Multi-GPU
        if not torch.cuda.device_count() > 1 or ignore_multidevice:
            return
        self._test_basic_cases_template(
            torch.randn(10, 5).cuda(0),
            torch.randn(10).cuda(1),
            torch.randn(5).cuda(0),
            constructor
        )

    def _build_params_dict(self, weight, bias, **kwargs):
        return [dict(params=[weight]), dict(params=[bias], **kwargs)]

    def _build_params_dict_single(self, weight, bias, **kwargs):
        return [dict(params=bias, **kwargs)]

    def test_sgd(self):
        try:
            self._test_rosenbrock(
                lambda params: p03_layers.P3SGD(params, lr=1e-3),
                wrap_old_fn(old_optim.sgd, learningRate=1e-3)
            )
            self._test_rosenbrock(
                lambda params: p03_layers.P3SGD(params, lr=1e-3, momentum=0.9,
                                                dampening=0, weight_decay=1e-4),
                wrap_old_fn(old_optim.sgd, learningRate=1e-3, momentum=0.9,
                            dampening=0, weightDecay=1e-4)
            )
            self._test_basic_cases(
                lambda weight, bias: p03_layers.P3SGD([weight, bias], lr=1e-3)
            )
            self._test_basic_cases(
                lambda weight, bias: p03_layers.P3SGD(
                    self._build_params_dict(weight, bias, lr=1e-2),
                    lr=1e-3)
            )
            self._test_basic_cases(
                lambda weight, bias: p03_layers.P3SGD(
                    self._build_params_dict_single(weight, bias, lr=1e-2),
                    lr=1e-3)
            )
        except NotImplementedError:
            pass


def check_net(model):
    model.train()
    # output from network
    data = torch.autograd.Variable(torch.rand(2, 1, 28, 28))
    # output from network
    target = torch.autograd.Variable((torch.rand(2) * 2).long())
    optimizer = SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    # Forward prediction step
    output = model(data)
    loss = F.nll_loss(output, target)
    # Backpropagation step
    loss.backward()
    optimizer.step()


def test_nets():
    model = p03_layers.Net()
    check_net(model)


def test_sgd():
    return TestOptim

class TestDropout(TestCase):
    def test_p3dropout(self):

        #testing dropout behaviour at training mode or when p = 0
        input = Variable(torch.randn(5))
        m = p03_layers.P3Dropout(0.0, inplace = False, training = True)
        output = m(input)
        self.assertEqual(output, input)
        m = p03_layers.P3Dropout(0.5, inplace = False, training = False)
        output = m(input)
        self.assertEqual(output, input)

        #testing wrong probability values
        self.assertRaises(ValueError, p03_layers.P3Dropout, 1.2)
        self.assertRaises(ValueError, p03_layers.P3Dropout, -0.4)

        #testing inplace property
        input = Variable(torch.randn(5))
        m = p03_layers.P3Dropout(0.5, inplace = False, training = True)
        output = m(input)
        self.assertNotEqual(output, input)
        m = p03_layers.P3Dropout(0.5, inplace = True, training = True)
        output = m(input)
        self.assertEqual(output, input)

        #testing for correctness of results
        #input = Variable(torch.randn(5))
        p = 0.5
        m = p03_layers.P3Dropout(p, training = True)
        count = 0
        for _ in range(1000):
            input = Variable(torch.randn(100))
            output = m(input)
            count += output.nonzero().numel()
        print (count)
        self.assertAlmostEqual(count/(1000*100), p, 1e-3)
        
        #testing for consistent type of input and output
        input = Variable(torch.randn(5))
        m = p03_layers.P3Dropout()
        output = m(input)
        self.assertIsInstance(input.data, type(output.data))


    def test_p3dropout2d(self):
        # TODO Implement me
        pass

class TestLinear(TestCase):

    #testing Linear modules forward for correctness
    def test_p3linear(self):

        input = Variable(torch.randn(128, 20))

        torch.manual_seed(7)
        nn_model = nn.Linear(20, 30)
        nn_output = nn_model(input)

        torch.manual_seed(7)
        my_model = p03_layers.P3Linear(20,30)
        my_output = my_model(input)

        self.assertEqual(nn_output, my_output)

    #testing for correct saving of parameters for backprop
    def test_p3linear_saving_for_backwards(self):
        input = Variable(torch.randn(128, 20))
        my_model = p03_layers.P3Linear(20,30)
        my_func = p03_layers.P3LinearFunction()
        output = my_func(input, my_model.weight, my_model.bias)
        saved_input, saved_weights, saved_bias = my_func.saved_variables

        self.assertEqual(input, saved_input)
        self.assertEqual(saved_weights, my_model.weight)
        self.assertEqual(saved_bias, my_model.bias)

class TestActivations(TestCase):
    def test_p3relu_function(self):

        #testing the correctness of relu
        input = Variable(torch.randn(5))
        output = p03_layers.p3relu(input)
        nn_output = F.relu(input)
        self.assertEqual(nn_output, output)
        self.assertIsInstance(input.data, type(output.data))

        #testing the correctness of relu with inplace
        p03_layers.p3relu(input, inplace = True)
        nn_output = F.relu(input)
        self.assertEqual(nn_output, input)



    def test_p3relu_class(self):
        input = Variable(torch.randn(5))
        my_relu = p03_layers.P3ReLU()
        output = my_relu(input)
        nn_relu = nn.ReLU()
        nn_output = nn_relu(input)
        self.assertEqual(nn_output, output)
        self.assertIsInstance(input.data, type(output.data))

        #testing the correctness of relu with inplace
        my_relu = p03_layers.P3ReLU(inplace = True)
        my_relu(input)
        nn_relu = nn.ReLU()
        nn_output = nn_relu(input)
        self.assertEqual(nn_output, input)

    def test_p3elu_function(self):
        # testing for 1D input tensor
        input = Variable(torch.randn(5))
        my_func = p03_layers.P3ELUFunction()
        output = my_func(input)
        nn_output = F.elu(input)
        self.assertEqual(output, nn_output)

        # testing for 2D input tensor
        input = Variable(torch.randn(5,10))
        my_func = p03_layers.P3ELUFunction()
        output = my_func(input)
        nn_output = F.elu(input)
        self.assertEqual(output, nn_output)


    def test_p3elu_class(self):
        input = Variable(torch.randn(5))
        my_elu = p03_layers.P3ELU()
        output = my_elu.forward(input)
        nn_elu = nn.ELU()
        nn_output = nn_elu(input)
        self.assertEqual(nn_output, output)
        self.assertIsInstance(input.data, type(output.data))

     # testing for different alpha
    def test_p3elu_class(self):
        input = Variable(torch.randn(5))
        my_elu = p03_layers.P3ELU(2.5, False)
        output = my_elu(input)
        nn_elu = nn.ELU(alpha = 2.5, inplace = False)
        nn_output = nn_elu(input)
        self.assertEqual(nn_output, output)
        self.assertIsInstance(input.data, type(output.data))

    # testing for inplace argument
    def test_p3elu_class(self):
        input = Variable(torch.randn(5))
        my_elu = p03_layers.P3ELU(2.5, True)
        nn_output = F.elu(input, alpha = 2.5)
        my_elu(input)
        self.assertEqual(nn_output, input)
        self.assertIsInstance(input.data, type(input.data))

'''#class TestActivations(TestCase):       
def test_p3bce_loss():
        
    pass'''


if __name__ == '__main__':
    # Automatically call every function
    # in this file which begins with test.
    # see unittest library docs for details.
    run_tests()
