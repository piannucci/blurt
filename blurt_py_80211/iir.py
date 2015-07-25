import numpy as np
from scipy import weave
from scipy.weave import converters
import subprocess

def _iir_cpp_impl(order, alpha, beta, gamma, float=False):
    impl = "for (int i=0; i<%d && i<N; i++)\n" % order
    impl += "\ty(i) = x(i)*%.10e;\n" % (1./gamma)
    impl += "for (int i=%d; i<N; i++)\n" % order
    impl += "\ty(i) = " + " + ".join("%.10e%s*x(i%s)" % (alpha[i], 'f' if float else '', str(i-order) if i!=order else "") for i in range(order+1))
    impl += " + " + " + ".join("%.10e%s*y(i%s)" % (beta[i], 'f' if float else '', str(i-order)) for i in range(order))
    impl += ";\nfor (int i=0; i<N; i++)\n\ty(i) *= %.10e;" % gamma
    return impl

def _iir_cpp_impl_continuous(order, alpha, beta, gamma, float=False):
    impl = "for (int i=0; i<%d; i++)\n" % order
    impl += "{\n"
    impl += "\tx(i) = x_hist(i);\n"
    impl += "\ty(i) = y_hist(i);\n"
    impl += "}\n"
    impl += "for (int i=%d; i<N+%d; i++)\n" % (order, order)
    impl += "\ty(i) = " + " + ".join("%.10e%s*x(i%s)" % (alpha[i], 'f' if float else '', str(i-order) if i!=order else "") for i in range(order+1))
    impl += " + " + " + ".join("%.10e%s*y(i%s)" % (beta[i], 'f' if float else '', str(i-order)) for i in range(order))
    impl += ";\n"
    impl += "for (int i=0; i<%d; i++)\n" % order
    impl += "{\n"
    impl += "\tx_hist(i) = x(N+i);\n"
    impl += "\ty_hist(i) = y(N+i);\n"
    impl += "}\n"
    impl += "for (int i=%d; i<N+%d; i++)\n" % (order, order)
    impl += "\ty(i) *= %.10e;" % gamma
    return impl

class IIRFilter:
    def __init__(self, order, alpha, beta, gamma):
        self.code = _iir_cpp_impl(order, alpha, beta, gamma)
        self.codef = _iir_cpp_impl(order, alpha, beta, gamma, True)
    def __call__(self, x):
        y = np.zeros_like(x)
        N = y.size
        weave.inline(self.code if x.real.dtype==np.float64 else self.codef,
                     ['N','x','y'], type_converters=converters.blitz, verbose=0)
        return y

class ContinuousIIRFilter:
    def __init__(self, order, alpha, beta, gamma, dtype):
        single = (np.dtype(dtype).name in ['float32', 'complex64'])
        self.code = _iir_cpp_impl_continuous(order, alpha, beta, gamma, single)
        self.x_hist = np.zeros(order, dtype)
        self.y_hist = np.zeros(order, dtype)
        self.order = order
    def __call__(self, x):
        N = x.size
        x = np.r_[np.empty(self.order, x.dtype), x]
        y = np.empty(N+self.order, x.dtype)
        x_hist = self.x_hist
        y_hist = self.y_hist
        weave.inline(self.code, ['N','x_hist','y_hist','x','y'],
                     type_converters=converters.blitz, verbose=0)
        return y[self.order:]

def mkfilter(method, type, order, alpha, continuous=False, dtype=None):
    if not isinstance(alpha, (tuple, list)):
        alpha = (alpha,)
    args = ['./mkfilter'] + ('-%s' % method).split(' ') + ['-%s' % type, '-o', str(order), '-l', '-a'] + [str(a) for a in alpha]
    result = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()[0]
    result = result.strip().split('\n')[1:]
    gamma = 1./float(result[0].strip().split(' ')[-1])
    alpha = [float(x.strip()) for x in result[2:3+order]]
    beta = [float(x.strip()) for x in result[4+order:5+2*order]]
    if not continuous:
        return IIRFilter(order, alpha, beta, gamma)
    else:
        return ContinuousIIRFilter(order, alpha, beta, gamma, dtype)

def lowpass(freq, order=6, method='Bu', continuous=False, dtype=None):
    return mkfilter(method, 'Lp', order, freq, continuous, dtype)

def highpass(freq, order=6, method='Bu', continuous=False, dtype=None):
    return mkfilter(method, 'Hp', order, freq, continuous, dtype)
