import numpy as np
import typing

class Cardinal(typing.TypeVar, _root=True):
    __slots__ = ('__value__')
    def __init__(self, value):
        self.__value__ = value
        if not isinstance(value, int) or value <= 0:
            raise TypeError("Cardinal must be a positive integer")
    def __repr__(self):
        return 'Cardinal[%d]' % (self.__value__,)
    def __reduce__(self):
        return (Cardinal, (self.__value__,))

class Array(typing.Generic[typing.TypeVar('shape'), typing.TypeVar('VT')]):
    def __class_getitem__(cls, params):
        if not isinstance(params, tuple) or len(params) != 2:
            raise TypeError("Array requires two parameters: shape, value type")
        shape, value_type = params
        shape = typing.Tuple[tuple(Cardinal(s) if isinstance(s, int) else typing.TypeVar(s) if isinstance(s, str) else s for s in shape)]
        if isinstance(value_type, str):
            value_type = typing.TypeVar(value_type)
        if not isinstance(value_type, typing.TypeVar):
            value_type = np.dtype(value_type).type
        return super().__class_getitem__((shape, value_type))

class Equals:
    __slots__ = ['t1', 't2']
    def __new__(cls, t1, t2):
        t1 = t1.__value__ if isinstance(t1, Cardinal) else t1.__name__ if isinstance(t1, typing.TypeVar) else t1
        t2 = t2.__value__ if isinstance(t2, Cardinal) else t2.__name__ if isinstance(t2, typing.TypeVar) else t2
        if isinstance(t1, int) and isinstance(t2, int):
            return t1 == t2
        if not isinstance(t1, (int, str)) and t1 is t2:
            return True
        condition = object.__new__(Equals)
        condition.t1 = t1
        condition.t2 = t2
        return condition
    def __repr__(self):
        lhs = 'a.%s' % self.t1 if isinstance(self.t1, str) else str(self.t1)
        rhs = 'b.%s' % self.t2 if isinstance(self.t2, str) else str(self.t2)
        return '(%s == %s)' % (lhs, rhs)

class All:
    __slots__ = ['clauses']
    def __new__(cls, *clauses):
        clauses = [cc for c in clauses for cc in (c.clauses if isinstance(c, All) else c if hasattr(c, '__iter__') else (c,)) if cc is not True]
        if len(clauses) == 0:
            return True
        elif len(clauses) == 1:
            return clauses[0]
        elif any(c is False for c in clauses):
            return False
        else:
            condition = object.__new__(All)
            condition.clauses = clauses
            return condition
    def __repr__(self):
        return 'All(%s)' % ', '.join(map(repr, self.clauses))

class Any:
    __slots__ = ['clauses']
    def __new__(cls, *clauses):
        clauses = [cc for c in clauses for cc in (c.clauses if isinstance(c, Any) else c if hasattr(c, '__iter__') else (c,)) if cc is not False]
        if len(clauses) == 0:
            return False
        elif len(clauses) == 1:
            return clauses[0]
        elif any(c is True for c in clauses):
            return True
        elif any(isinstance(c, All) for c in clauses):
            # put in conjunctive normal form
            return All(map(Any, itertools.product(*((c.clauses if isinstance(c, All) else (c,)) for c in clauses))))
        else:
            condition = object.__new__(Any)
            condition.clauses = clauses
            return condition
    def __repr__(self):
        return 'Any(%s)' % ', '.join(map(repr, self.clauses))

class Subtype:
    def __new__(cls, t1, t2):
        if isinstance(t1, str): t1 = typing.TypeVar(t1)
        if isinstance(t2, str): t2 = typing.TypeVar(t2)

        if isinstance(t1, typing.TypeVar) or isinstance(t2, typing.TypeVar): # n.b. Cardinal is a subclass of TypeVar
            return Equals(t1, t2)

        if typing.Any in (t1, t2):
            return True

        left_types = t1.__args__ if isinstance(t1, typing._GenericAlias) and t1.__origin__ is typing.Union else (t1,)
        right_types = t2.__args__ if isinstance(t2, typing._GenericAlias) and t2.__origin__ is typing.Union else (t2,)

        if len(left_types) * len(right_types) > 1:
            return All(Any(Subtype(l, r) for r in right_types) for l in left_types)

        if not isinstance(t1, typing._GenericAlias) or not isinstance(t2, typing._GenericAlias):
            return t1 == t2

        t1 = left_types[0]
        t2 = right_types[0]

        o1, a1 = t1.__origin__, t1.__args__
        o2, a2 = t2.__origin__, t2.__args__

        if not issubclass(o1, o2):
            return False

        if o1 is typing.Callable:
            if o2 is not typing.Callable or len(a1) != len(a2):
                return False
            return All(Subtype(a1[-1], a2[-1]), (Subtype(aa2, aa1) for aa1, aa2 in zip(a1[:-1], a2[:-1])))

        if o1 is tuple:
            if o2 is not tuple or len(a1) != len(a2):
                return False
            return All(Subtype(aa1, aa2) for aa1, aa2 in zip(a1, a2))

        if o1 is Array:
            s1, vt1 = a1
            s2, vt2 = a2
            s1 = s1.__args__
            s2 = s2.__args__
            if o2 is not Array or len(s1) != len(s2):
                return False
            return All((Equals(ss1, ss2) for ss1, ss2 in zip(s1, s2)), Equals(vt1, vt2))

        raise ValueError('Unhandled subtype condition: %s <= %s' % (t1, t2))

#def sat(condition, ns1, ns2):
#    # make a list of variables
#    if condition is True:
#        return ()
#    elif isinstance(condition, Equals):
#        return (
#    stack = []
#    for 
