#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION

#include "Python.h"
#include "blitz/array.h"
#include "numpy/arrayobject.h"

template<class T> static const int numeric_type = 0;
template<> int numeric_type<long> = NPY_LONG;
template<> int numeric_type<npy_ubyte> = NPY_UBYTE;
template<> int numeric_type<npy_uint> = NPY_UINT;
template<> int numeric_type<double> = NPY_DOUBLE;

template<class T, int N>
static blitz::Array<T,N> convert_to_blitz(PyArrayObject* arr_obj, const char* name)
{
    if (PyArray_NDIM(arr_obj) != N)
    {
        PyErr_Format(PyExc_TypeError, "Conversion Error: array argument '%s' has ndim %d; expected %d", name, PyArray_NDIM(arr_obj), N);
        throw 1;
    }
    if (PyTypeNum_ISEXTENDED(numeric_type<T>))
    {
        PyErr_Format(PyExc_TypeError, "Conversion Error: extended types not supported for array argument '%s'", name);
        throw 1;
    }
    PyArray_Descr *descr = PyArray_DESCR(arr_obj);
    if (!PyArray_EquivTypenums(descr->type_num, numeric_type<T>))
    {
        PyArray_Descr *expected = PyArray_DescrFromType(numeric_type<T>);
        PyErr_Format(PyExc_TypeError, "Conversion Error: array argument '%s' has type %s; expected %s", name,
                descr->typeobj->tp_name, expected->typeobj->tp_name);
        Py_DECREF(expected);
        throw 1;
    }
    blitz::TinyVector<int,N> shape(0);
    blitz::TinyVector<int,N> strides(0);
    for (int i=0; i<N; i++)
    {
        shape[i] = PyArray_DIMS(arr_obj)[i];
        strides[i] = PyArray_STRIDES(arr_obj)[i]/sizeof(T);
    }
    return blitz::Array<T,N>((T*)PyArray_DATA(arr_obj), shape, strides, blitz::neverDeleteData);
}

static PyObject* crc(PyObject*self, PyObject* args)
{
    PyArrayObject *py_a = NULL, *py_lut = NULL;
    if(!PyArg_ParseTuple(args,"O!O!:crc",&PyArray_Type,&py_a,&PyArray_Type,&py_lut))
        return NULL;
    try
    {
        auto a = convert_to_blitz<long,1>(py_a,"a");
        auto lut = convert_to_blitz<npy_uint,1>(py_lut,"lut");
        uint32_t r = 0, A=a.extent(blitz::firstDim);
        for (uint32_t i=0; i<A; i++)
            r = (r << 16) ^ a(i) ^ lut(r >> 16);
        return PyLong_FromUnsignedLong(r);
    }
    catch(...)
    {
        return nullptr;
    }
}

static PyObject* viterbi(PyObject*self, PyObject* args)
{
    PyArrayObject *py_x = NULL, *py_msg = NULL;
    long N;
    if(!PyArg_ParseTuple(args,"lO!O!:viterbi",&N,&PyArray_Type,&py_x,&PyArray_Type,&py_msg))
        return NULL;
    try
    {
        auto x = convert_to_blitz<double,2>(py_x,"x");
        auto msg = convert_to_blitz<npy_ubyte,1>(py_msg,"msg");
        const int M = 128;
        int64_t cost[M*2], scores[M] = {/* zero-initialized */};
        uint8_t bt[N][M];
        for (int k=0; k<N; k++)
        {
            for (int i=0; i<M; i++)
            {
                cost[2*i+0] = scores[((i<<1) & 127) | 0] + x(k, i);
                cost[2*i+1] = scores[((i<<1) & 127) | 1] + x(k, i);
            }
            for (int i=0; i<M; i++)
            {
                int a = cost[2*i+0];
                int b = cost[2*i+1];
                bt[k][i] = (a<b) ? 1 : 0;
                scores[i] = (a<b) ? b : a;
            }
        }
        int i = (scores[0] < scores[1]) ? 1 : 0;
        for (int k=N-1; k>=0; k--)
        {
            int j = bt[k][i];
            msg(k) = i >> 6;
            i = ((i<<1)&127) + j;
        }
        Py_INCREF(Py_None);
        return Py_None;
    }
    catch(...)
    {
        return nullptr;
    }
}

static PyObject* ccenc(PyObject*self, PyObject* args)
{
    PyArrayObject *py_y = NULL, *py_output = NULL, *py_output_map = NULL;
    if(!PyArg_ParseTuple(args,"O!O!O!:ccenc",&PyArray_Type,&py_y,&PyArray_Type,&py_output,&PyArray_Type,&py_output_map))
        return NULL;
    try
    {
        auto y = convert_to_blitz<long,1>(py_y,"y");
        auto output = convert_to_blitz<npy_ubyte,1>(py_output,"output");
        auto output_map = convert_to_blitz<long,2>(py_output_map,"output_map");
        int sh = 0, N = y.extent(blitz::firstDim);
        for (int i=0; i<N; i++)
        {
            sh = (sh>>1) ^ ((int)y(i) << 6);
            output(2*i+0) = output_map(0,sh);
            output(2*i+1) = output_map(1,sh);
        }
        Py_INCREF(Py_None);
        return Py_None;
    }
    catch(...)
    {
        return nullptr;
    }
}

static PyMethodDef compiled_methods[] =
{
    {"crc", (PyCFunction)crc, METH_VARARGS},
    {"viterbi", (PyCFunction)viterbi, METH_VARARGS},
    {"ccenc", (PyCFunction)ccenc, METH_VARARGS},
    {NULL, NULL}
};

static struct PyModuleDef moduledef = {PyModuleDef_HEAD_INIT, "kernels", NULL, 0, compiled_methods};

extern "C"
PyObject *PyInit_kernels(void)
{
    PyObject *module = PyModule_Create(&moduledef);
    Py_Initialize();
    import_array();
    PyImport_ImportModule("numpy");
    return module;
}
