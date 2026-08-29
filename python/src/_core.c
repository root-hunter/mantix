#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "mantix/mantix.h"
#include "mantix/limb.h"

typedef struct {
    PyObject_HEAD
    mtx_float val;
} MantixFloatObject;

static PyTypeObject MantixFloat_Type;

#define MantixFloat_Check(op) PyObject_TypeCheck((op), &MantixFloat_Type)

#define MANTIX_FREELIST_MAX 4096
static MantixFloatObject *mantix_freelist[MANTIX_FREELIST_MAX];
static int mantix_numfree = 0;

static inline MantixFloatObject *mantix_float_alloc(size_t precision)
{
    MantixFloatObject *obj;
    if (mantix_numfree > 0) {
        obj = mantix_freelist[--mantix_numfree];
        _Py_NewReference((PyObject *)obj);
        obj->val.precision = precision;
        obj->val.used = 0;
        obj->val.exponent = 0;
        obj->val.negative = false;
        obj->val.limbs = &obj->val.inline_limb;
        obj->val.capacity = 1;
        return obj;
    }
    obj = PyObject_Malloc(sizeof(MantixFloatObject));
    if (__builtin_expect(obj == NULL, 0)) {
        PyErr_NoMemory();
        return NULL;
    }
    PyObject_Init((PyObject *)obj, &MantixFloat_Type);
    obj->val.precision = precision;
    obj->val.used = 0;
    obj->val.exponent = 0;
    obj->val.negative = false;
    obj->val.limbs = &obj->val.inline_limb;
    obj->val.capacity = 1;
    return obj;
}

static void MantixFloat_dealloc(MantixFloatObject *self)
{
    if (self->val.limbs != &self->val.inline_limb) {
        free(self->val.limbs);
    }
    if (mantix_numfree < MANTIX_FREELIST_MAX && Py_IS_TYPE(self, &MantixFloat_Type)) {
        mantix_freelist[mantix_numfree++] = self;
    } else {
        Py_TYPE(self)->tp_free((PyObject *)self);
    }
}

static PyObject *MantixFloat_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    if (type == &MantixFloat_Type) {
        return (PyObject *)mantix_float_alloc(53U);
    }
    MantixFloatObject *self = (MantixFloatObject *)type->tp_alloc(type, 0);
    if (self != NULL) {
        if (mtx_init(&self->val, 53U) != MTX_OK) {
            Py_DECREF(self);
            PyErr_SetString(PyExc_RuntimeError, "Failed to initialize MantixFloat internal representation");
            return NULL;
        }
    }
    return (PyObject *)self;
}

static int MantixFloat_init(MantixFloatObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"value", "precision", NULL};
    PyObject *val_obj = NULL;
    unsigned long long precision = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OK", kwlist, &val_obj, &precision)) {
        return -1;
    }

    if (precision > 0) {
        self->val.precision = (size_t)precision;
    }

    if (val_obj == NULL || val_obj == Py_None) {
        mtx_set_zero(&self->val);
        return 0;
    }

    if (Py_IS_TYPE(val_obj, &MantixFloat_Type)) {
        MantixFloatObject *other = (MantixFloatObject *)val_obj;
        if (mtx_set(&self->val, &other->val) != MTX_OK) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy MantixFloat");
            return -1;
        }
        return 0;
    }

    if (PyFloat_CheckExact(val_obj)) {
        double d = PyFloat_AS_DOUBLE(val_obj);
        if (__builtin_expect(mtx_set_f64(&self->val, d) != MTX_OK, 0)) {
            PyErr_SetString(PyExc_ValueError, "Failed to set float value (e.g. NaN/Inf not supported)");
            return -1;
        }
        return 0;
    }

    if (PyLong_CheckExact(val_obj)) {
        int overflow;
        long long v = PyLong_AsLongLongAndOverflow(val_obj, &overflow);
        if (!overflow) {
            if (mtx_set_i64(&self->val, (int64_t)v) != MTX_OK) {
                PyErr_SetString(PyExc_RuntimeError, "Failed to set integer value");
                return -1;
            }
            return 0;
        }
        double d = PyLong_AsDouble(val_obj);
        if (PyErr_Occurred()) {
            return -1;
        }
        if (mtx_set_f64(&self->val, d) != MTX_OK) {
            PyErr_SetString(PyExc_ValueError, "Failed to set large integer value");
            return -1;
        }
        return 0;
    }

    if (PyUnicode_Check(val_obj)) {
        const char *s = PyUnicode_AsUTF8(val_obj);
        if (s == NULL) return -1;
        char *endptr = NULL;
        double d = strtod(s, &endptr);
        if (endptr == s) {
            PyErr_Format(PyExc_ValueError, "Cannot parse float from string '%s'", s);
            return -1;
        }
        if (mtx_set_f64(&self->val, d) != MTX_OK) {
            PyErr_SetString(PyExc_ValueError, "Failed to set parsed float value");
            return -1;
        }
        return 0;
    }

    PyErr_SetString(PyExc_TypeError, "Expected float, int, str, or MantixFloat");
    return -1;
}

static int to_mantix_float(PyObject *obj, size_t prec, mtx_float *out, bool *allocated)
{
    *allocated = false;
    if (Py_IS_TYPE(obj, &MantixFloat_Type)) {
        *out = ((MantixFloatObject *)obj)->val;
        return 0;
    }
    if (PyFloat_CheckExact(obj)) {
        if (mtx_init(out, prec) != MTX_OK) return -1;
        *allocated = true;
        double d = PyFloat_AS_DOUBLE(obj);
        if (mtx_set_f64(out, d) != MTX_OK) {
            mtx_clear(out);
            return -1;
        }
        return 0;
    }
    if (PyLong_Check(obj)) {
        if (mtx_init(out, prec) != MTX_OK) return -1;
        *allocated = true;
        int overflow;
        long long v = PyLong_AsLongLongAndOverflow(obj, &overflow);
        if (!overflow) {
            mtx_set_i64(out, (int64_t)v);
            return 0;
        }
        double d = PyLong_AsDouble(obj);
        if (PyErr_Occurred()) {
            mtx_clear(out);
            return -1;
        }
        mtx_set_f64(out, d);
        return 0;
    }
    return -1;
}

static PyObject *MantixFloat_add(PyObject *v, PyObject *w)
{
    if (__builtin_expect(Py_IS_TYPE(v, &MantixFloat_Type) && Py_IS_TYPE(w, &MantixFloat_Type), 1)) {
        MantixFloatObject *a = (MantixFloatObject *)v;
        MantixFloatObject *b = (MantixFloatObject *)w;
        size_t target_prec = a->val.precision > b->val.precision ? a->val.precision : b->val.precision;
        MantixFloatObject *res = mantix_float_alloc(target_prec);
        if (__builtin_expect(res == NULL, 0)) return NULL;
        mtx_add(&res->val, &a->val, &b->val, MTX_ROUND_TO_NEAREST_EVEN);
        return (PyObject *)res;
    }

    size_t prec = 53U;
    if (Py_IS_TYPE(v, &MantixFloat_Type)) prec = ((MantixFloatObject *)v)->val.precision;
    else if (Py_IS_TYPE(w, &MantixFloat_Type)) prec = ((MantixFloatObject *)w)->val.precision;

    mtx_float a, b;
    bool alloc_a = false, alloc_b = false;

    if (to_mantix_float(v, prec, &a, &alloc_a) < 0 || to_mantix_float(w, prec, &b, &alloc_b) < 0) {
        if (alloc_a) mtx_clear(&a);
        if (alloc_b) mtx_clear(&b);
        Py_RETURN_NOTIMPLEMENTED;
    }

    size_t target_prec = a.precision > b.precision ? a.precision : b.precision;
    MantixFloatObject *res = mantix_float_alloc(target_prec);
    if (res == NULL) {
        if (alloc_a) mtx_clear(&a);
        if (alloc_b) mtx_clear(&b);
        return NULL;
    }

    mtx_add(&res->val, &a, &b, MTX_ROUND_TO_NEAREST_EVEN);

    if (alloc_a) mtx_clear(&a);
    if (alloc_b) mtx_clear(&b);
    return (PyObject *)res;
}

static PyObject *MantixFloat_sub(PyObject *v, PyObject *w)
{
    if (__builtin_expect(Py_IS_TYPE(v, &MantixFloat_Type) && Py_IS_TYPE(w, &MantixFloat_Type), 1)) {
        MantixFloatObject *a = (MantixFloatObject *)v;
        MantixFloatObject *b = (MantixFloatObject *)w;
        size_t target_prec = a->val.precision > b->val.precision ? a->val.precision : b->val.precision;
        MantixFloatObject *res = mantix_float_alloc(target_prec);
        if (__builtin_expect(res == NULL, 0)) return NULL;
        mtx_sub(&res->val, &a->val, &b->val, MTX_ROUND_TO_NEAREST_EVEN);
        return (PyObject *)res;
    }

    size_t prec = 53U;
    if (Py_IS_TYPE(v, &MantixFloat_Type)) prec = ((MantixFloatObject *)v)->val.precision;
    else if (Py_IS_TYPE(w, &MantixFloat_Type)) prec = ((MantixFloatObject *)w)->val.precision;

    mtx_float a, b;
    bool alloc_a = false, alloc_b = false;

    if (to_mantix_float(v, prec, &a, &alloc_a) < 0 || to_mantix_float(w, prec, &b, &alloc_b) < 0) {
        if (alloc_a) mtx_clear(&a);
        if (alloc_b) mtx_clear(&b);
        Py_RETURN_NOTIMPLEMENTED;
    }

    size_t target_prec = a.precision > b.precision ? a.precision : b.precision;
    MantixFloatObject *res = mantix_float_alloc(target_prec);
    if (res == NULL) {
        if (alloc_a) mtx_clear(&a);
        if (alloc_b) mtx_clear(&b);
        return NULL;
    }

    mtx_sub(&res->val, &a, &b, MTX_ROUND_TO_NEAREST_EVEN);

    if (alloc_a) mtx_clear(&a);
    if (alloc_b) mtx_clear(&b);
    return (PyObject *)res;
}

static PyObject *MantixFloat_mul(PyObject *v, PyObject *w)
{
    if (__builtin_expect(Py_IS_TYPE(v, &MantixFloat_Type) && Py_IS_TYPE(w, &MantixFloat_Type), 1)) {
        MantixFloatObject *a = (MantixFloatObject *)v;
        MantixFloatObject *b = (MantixFloatObject *)w;
        size_t target_prec = a->val.precision > b->val.precision ? a->val.precision : b->val.precision;
        MantixFloatObject *res = mantix_float_alloc(target_prec);
        if (__builtin_expect(res == NULL, 0)) return NULL;
        mtx_mul(&res->val, &a->val, &b->val, MTX_ROUND_TO_NEAREST_EVEN);
        return (PyObject *)res;
    }

    size_t prec = 53U;
    if (Py_IS_TYPE(v, &MantixFloat_Type)) prec = ((MantixFloatObject *)v)->val.precision;
    else if (Py_IS_TYPE(w, &MantixFloat_Type)) prec = ((MantixFloatObject *)w)->val.precision;

    mtx_float a, b;
    bool alloc_a = false, alloc_b = false;

    if (to_mantix_float(v, prec, &a, &alloc_a) < 0 || to_mantix_float(w, prec, &b, &alloc_b) < 0) {
        if (alloc_a) mtx_clear(&a);
        if (alloc_b) mtx_clear(&b);
        Py_RETURN_NOTIMPLEMENTED;
    }

    size_t target_prec = a.precision > b.precision ? a.precision : b.precision;
    MantixFloatObject *res = mantix_float_alloc(target_prec);
    if (res == NULL) {
        if (alloc_a) mtx_clear(&a);
        if (alloc_b) mtx_clear(&b);
        return NULL;
    }

    mtx_mul(&res->val, &a, &b, MTX_ROUND_TO_NEAREST_EVEN);

    if (alloc_a) mtx_clear(&a);
    if (alloc_b) mtx_clear(&b);
    return (PyObject *)res;
}

static PyObject *MantixFloat_iadd(PyObject *v, PyObject *w)
{
    if (Py_IS_TYPE(v, &MantixFloat_Type) && Py_IS_TYPE(w, &MantixFloat_Type)) {
        MantixFloatObject *self = (MantixFloatObject *)v;
        MantixFloatObject *other = (MantixFloatObject *)w;
        mtx_add(&self->val, &self->val, &other->val, MTX_ROUND_TO_NEAREST_EVEN);
        Py_INCREF(v);
        return v;
    }
    return MantixFloat_add(v, w);
}

static PyObject *MantixFloat_isub(PyObject *v, PyObject *w)
{
    if (Py_IS_TYPE(v, &MantixFloat_Type) && Py_IS_TYPE(w, &MantixFloat_Type)) {
        MantixFloatObject *self = (MantixFloatObject *)v;
        MantixFloatObject *other = (MantixFloatObject *)w;
        mtx_sub(&self->val, &self->val, &other->val, MTX_ROUND_TO_NEAREST_EVEN);
        Py_INCREF(v);
        return v;
    }
    return MantixFloat_sub(v, w);
}

static PyObject *MantixFloat_imul(PyObject *v, PyObject *w)
{
    if (Py_IS_TYPE(v, &MantixFloat_Type) && Py_IS_TYPE(w, &MantixFloat_Type)) {
        MantixFloatObject *self = (MantixFloatObject *)v;
        MantixFloatObject *other = (MantixFloatObject *)w;
        mtx_mul(&self->val, &self->val, &other->val, MTX_ROUND_TO_NEAREST_EVEN);
        Py_INCREF(v);
        return v;
    }
    return MantixFloat_mul(v, w);
}

static PyObject *MantixFloat_neg(MantixFloatObject *self)
{
    MantixFloatObject *res = mantix_float_alloc(self->val.precision);
    if (res == NULL) return NULL;
    mtx_neg(&res->val, &self->val);
    return (PyObject *)res;
}

static PyObject *MantixFloat_abs(MantixFloatObject *self)
{
    MantixFloatObject *res = mantix_float_alloc(self->val.precision);
    if (res == NULL) return NULL;
    mtx_abs(&res->val, &self->val);
    return (PyObject *)res;
}

static int MantixFloat_bool(MantixFloatObject *self)
{
    return !mtx_is_zero(&self->val);
}

static PyObject *MantixFloat_float(MantixFloatObject *self)
{
    double d = mtx_get_f64(&self->val, MTX_ROUND_TO_NEAREST_EVEN);
    return PyFloat_FromDouble(d);
}

static PyObject *MantixFloat_richcompare(PyObject *v, PyObject *w, int op)
{
    size_t prec = 53U;
    if (Py_IS_TYPE(v, &MantixFloat_Type)) prec = ((MantixFloatObject *)v)->val.precision;
    else if (Py_IS_TYPE(w, &MantixFloat_Type)) prec = ((MantixFloatObject *)w)->val.precision;

    mtx_float a, b;
    bool alloc_a = false, alloc_b = false;

    if (to_mantix_float(v, prec, &a, &alloc_a) < 0 || to_mantix_float(w, prec, &b, &alloc_b) < 0) {
        if (alloc_a) mtx_clear(&a);
        if (alloc_b) mtx_clear(&b);
        Py_RETURN_NOTIMPLEMENTED;
    }

    int cmp = mtx_cmp(&a, &b);

    if (alloc_a) mtx_clear(&a);
    if (alloc_b) mtx_clear(&b);

    Py_RETURN_RICHCOMPARE(cmp, 0, op);
}

static PyObject *MantixFloat_repr(MantixFloatObject *self)
{
    double d = mtx_get_f64(&self->val, MTX_ROUND_TO_NEAREST_EVEN);
    return PyUnicode_FromFormat("MantixFloat(%g, prec=%zu)", d, self->val.precision);
}

static PyObject *MantixFloat_str(MantixFloatObject *self)
{
    double d = mtx_get_f64(&self->val, MTX_ROUND_TO_NEAREST_EVEN);
    char buf[64];
    snprintf(buf, sizeof(buf), "%.17g", d);
    return PyUnicode_FromString(buf);
}

static PyObject *MantixFloat_to_f32(MantixFloatObject *self, PyObject *Py_UNUSED(ignored))
{
    float f = mtx_get_f32(&self->val, MTX_ROUND_TO_NEAREST_EVEN);
    return PyFloat_FromDouble((double)f);
}

static PyObject *MantixFloat_to_f64(MantixFloatObject *self, PyObject *Py_UNUSED(ignored))
{
    double d = mtx_get_f64(&self->val, MTX_ROUND_TO_NEAREST_EVEN);
    return PyFloat_FromDouble(d);
}

static PyObject *MantixFloat_is_zero(MantixFloatObject *self, PyObject *Py_UNUSED(ignored))
{
    if (mtx_is_zero(&self->val)) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject *MantixFloat_is_normalized(MantixFloatObject *self, PyObject *Py_UNUSED(ignored))
{
    if (mtx_is_normalized(&self->val)) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject *MantixFloat_get_precision(MantixFloatObject *self, void *closure)
{
    return PyLong_FromSize_t(self->val.precision);
}

static PyObject *MantixFloat_get_exponent(MantixFloatObject *self, void *closure)
{
    return PyLong_FromLongLong(self->val.exponent);
}

static PyObject *MantixFloat_get_negative(MantixFloatObject *self, void *closure)
{
    if (self->val.negative) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyGetSetDef MantixFloat_getseters[] = {
    {"precision", (getter)MantixFloat_get_precision, NULL, "Precision in bits", NULL},
    {"exponent", (getter)MantixFloat_get_exponent, NULL, "Binary exponent", NULL},
    {"negative", (getter)MantixFloat_get_negative, NULL, "Sign is negative", NULL},
    {NULL}
};

static PyMethodDef MantixFloat_methods[] = {
    {"to_f32", (PyCFunction)MantixFloat_to_f32, METH_NOARGS, "Convert to 32-bit single precision float"},
    {"to_f64", (PyCFunction)MantixFloat_to_f64, METH_NOARGS, "Convert to 64-bit double precision float"},
    {"is_zero", (PyCFunction)MantixFloat_is_zero, METH_NOARGS, "Check if value is canonical zero"},
    {"is_normalized", (PyCFunction)MantixFloat_is_normalized, METH_NOARGS, "Check if value is normalized"},
    {NULL}
};

static PyNumberMethods MantixFloat_as_number = {
    .nb_add = MantixFloat_add,
    .nb_subtract = MantixFloat_sub,
    .nb_multiply = MantixFloat_mul,
    .nb_negative = (unaryfunc)MantixFloat_neg,
    .nb_absolute = (unaryfunc)MantixFloat_abs,
    .nb_bool = (inquiry)MantixFloat_bool,
    .nb_float = (unaryfunc)MantixFloat_float,
    .nb_inplace_add = MantixFloat_iadd,
    .nb_inplace_subtract = MantixFloat_isub,
    .nb_inplace_multiply = MantixFloat_imul,
};

static PyTypeObject MantixFloat_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "mantix._core.Float",
    .tp_doc = "Mantix arbitrary-precision floating-point number",
    .tp_basicsize = sizeof(MantixFloatObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = MantixFloat_new,
    .tp_init = (initproc)MantixFloat_init,
    .tp_dealloc = (destructor)MantixFloat_dealloc,
    .tp_repr = (reprfunc)MantixFloat_repr,
    .tp_str = (reprfunc)MantixFloat_str,
    .tp_as_number = &MantixFloat_as_number,
    .tp_richcompare = MantixFloat_richcompare,
    .tp_methods = MantixFloat_methods,
    .tp_getset = MantixFloat_getseters,
};

/* Fast FMA (Fused Multiply-Add) function: res = a * b + c */
static PyObject *mantix_py_fma(PyObject *self, PyObject *args)
{
    PyObject *obj_a, *obj_b, *obj_c;
    if (!PyArg_ParseTuple(args, "OOO", &obj_a, &obj_b, &obj_c)) {
        return NULL;
    }
    if (Py_IS_TYPE(obj_a, &MantixFloat_Type) &&
        Py_IS_TYPE(obj_b, &MantixFloat_Type) &&
        Py_IS_TYPE(obj_c, &MantixFloat_Type)) {
        MantixFloatObject *a = (MantixFloatObject *)obj_a;
        MantixFloatObject *b = (MantixFloatObject *)obj_b;
        MantixFloatObject *c = (MantixFloatObject *)obj_c;

        size_t p = a->val.precision;
        if (b->val.precision > p) p = b->val.precision;
        if (c->val.precision > p) p = c->val.precision;

        MantixFloatObject *res = mantix_float_alloc(p);
        if (res == NULL) return NULL;

        mtx_float tmp;
        mtx_init(&tmp, p);
        mtx_mul(&tmp, &a->val, &b->val, MTX_ROUND_TO_NEAREST_EVEN);
        mtx_add(&res->val, &tmp, &c->val, MTX_ROUND_TO_NEAREST_EVEN);
        mtx_clear(&tmp);

        return (PyObject *)res;
    }

    PyErr_SetString(PyExc_TypeError, "fma requires three MantixFloat arguments");
    return NULL;
}

/* Fast batch dot product: dot(v1, v2) */
static PyObject *mantix_py_dot(PyObject *self, PyObject *args)
{
    PyObject *seq1, *seq2;
    if (!PyArg_ParseTuple(args, "OO", &seq1, &seq2)) {
        return NULL;
    }
    Py_ssize_t n1 = PySequence_Size(seq1);
    Py_ssize_t n2 = PySequence_Size(seq2);
    if (n1 < 0 || n2 < 0 || n1 != n2) {
        PyErr_SetString(PyExc_ValueError, "Sequences must have the same non-negative length");
        return NULL;
    }

    MantixFloatObject *acc = mantix_float_alloc(53U);
    if (acc == NULL) return NULL;
    mtx_set_zero(&acc->val);

    mtx_float prod;
    mtx_init(&prod, 53U);

    for (Py_ssize_t i = 0; i < n1; ++i) {
        PyObject *item1 = PySequence_GetItem(seq1, i);
        PyObject *item2 = PySequence_GetItem(seq2, i);
        if (item1 == NULL || item2 == NULL) {
            Py_XDECREF(item1); Py_XDECREF(item2);
            Py_DECREF(acc); mtx_clear(&prod);
            return NULL;
        }

        mtx_float v1, v2;
        bool a1 = false, a2 = false;
        if (to_mantix_float(item1, 53U, &v1, &a1) == 0 && to_mantix_float(item2, 53U, &v2, &a2) == 0) {
            mtx_mul(&prod, &v1, &v2, MTX_ROUND_TO_NEAREST_EVEN);
            mtx_add(&acc->val, &acc->val, &prod, MTX_ROUND_TO_NEAREST_EVEN);
        }
        if (a1) mtx_clear(&v1);
        if (a2) mtx_clear(&v2);
        Py_DECREF(item1); Py_DECREF(item2);
    }
    mtx_clear(&prod);
    return (PyObject *)acc;
}

static PyMethodDef mantix_core_methods[] = {
    {"fma", mantix_py_fma, METH_VARARGS, "Fused multiply-add: fma(a, b, c) -> a * b + c"},
    {"dot", mantix_py_dot, METH_VARARGS, "Vector dot product: dot(seq1, seq2)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef mantix_core_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "mantix._core",
    .m_doc = "Core C extension module for Mantix arbitrary-precision library",
    .m_size = -1,
    .m_methods = mantix_core_methods,
};

PyMODINIT_FUNC PyInit__core(void)
{
    PyObject *m;
    if (PyType_Ready(&MantixFloat_Type) < 0) {
        return NULL;
    }

    m = PyModule_Create(&mantix_core_module);
    if (m == NULL) {
        return NULL;
    }

    Py_INCREF(&MantixFloat_Type);
    if (PyModule_AddObject(m, "Float", (PyObject *)&MantixFloat_Type) < 0) {
        Py_DECREF(&MantixFloat_Type);
        Py_DECREF(m);
        return NULL;
    }

    PyModule_AddIntConstant(m, "ROUND_NEAREST_EVEN", MTX_ROUND_TO_NEAREST_EVEN);
    PyModule_AddIntConstant(m, "ROUND_TOWARD_ZERO", MTX_ROUND_TOWARD_ZERO);
    PyModule_AddIntConstant(m, "ROUND_TOWARD_POSITIVE", MTX_ROUND_TOWARD_POSITIVE);
    PyModule_AddIntConstant(m, "ROUND_TOWARD_NEGATIVE", MTX_ROUND_TOWARD_NEGATIVE);

    return m;
}
