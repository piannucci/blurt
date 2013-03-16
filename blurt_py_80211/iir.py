from numpy import *
from scipy import weave
from scipy.weave import converters
import subprocess

def _iir_cpp_impl(order, alpha, beta, gamma, float=False):
    impl = "for (int i=0; i<%d && i<N; i++)\n" % order
    impl += "\ty(i) = x(i)*%.10e;\n" % (1./gamma)
    impl += "for (int i=%d; i<N; i++)\n" % order
    impl += "\ty(i) = " + " + ".join("%.10e%s*x(i%s)" % (alpha[i], 'f' if float else '', str(i-order) if i!=order else "") for i in xrange(order+1))
    impl += " + " + " + ".join("%.10e%s*y(i%s)" % (beta[i], 'f' if float else '', str(i-order)) for i in xrange(order))
    impl += ";\nfor (int i=0; i<N; i++)\n\ty(i) *= %.10e;" % gamma
    return impl

class IIRFilter:
    def __init__(self, order, alpha, beta, gamma):
        self.code = _iir_cpp_impl(order, alpha, beta, gamma)
        self.codef = _iir_cpp_impl(order, alpha, beta, gamma, True)
    def __call__(self, x):
        y = zeros_like(x)
        N = y.size
        weave.inline(self.code if x.real.dtype==float64 else self.codef,
                     ['N','x','y'], type_converters=converters.blitz, verbose=0)
        return y

def mkfilter(method, type, order, alpha):
    if not isinstance(alpha, (tuple, list)):
        alpha = (alpha,)
    args = ['./mkfilter'] + ('-%s' % method).split(' ') + ['-%s' % type, '-o', str(order), '-l', '-a'] + [str(a) for a in alpha]
    result = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()[0]
    result = result.strip().split('\n')[1:]
    gamma = 1./float(result[0].strip().split(' ')[-1])
    alpha = [float(x.strip()) for x in result[2:3+order]]
    beta = [float(x.strip()) for x in result[4+order:5+2*order]]
    return IIRFilter(order, alpha, beta, gamma)

def lowpass(freq, order=6, method='Bu'):
    return mkfilter(method, 'Lp', order, freq)

def highpass(freq, order=6, method='Bu'):
    return mkfilter(method, 'Hp', order, freq)
