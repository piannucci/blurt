import numpy as np
import typing
import collections.abc

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


class Variable(typing.NamedTuple):
    target_id : int
    name : str
    target : object

    def __repr__(self):
        target = '*' if self.target is None else self.target
        return '%s.%s' % (target, self.name)

    def bound_to(self, target):
        if not hasattr(target, self.name):
            return Variable(id(target), self.name, target)
        else:
            return Value(getattr(target, self.name))

    def __hash__(self):
        return hash((self.__class__, self.target_id, self.name))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.target is other.target and self.name == other.name


class Value(typing.NamedTuple):
    value : object


class UnsatisfiableError(Exception):
    pass


class Condition:
    def reversed(self):
        if isinstance(self, bool):
            return self
        if isinstance(self, (Any, All)):
            return self.__class__(*(Condition.reversed(c) for c in self.clauses))
        if isinstance(self, Identity):
            return Identity(self.rhs, self.lhs)
        if isinstance(self, LeftAssignment):
            return RightAssignment(self.rhs, self.lhs)
        if isinstance(self, RightAssignment):
            return LeftAssignment(self.rhs, self.lhs)
        raise TypeError()

    def bound_to(self, lhs, rhs):
        # Attach specific namespaces to variable names
        if isinstance(self, bool):
            return self
        if isinstance(self, (Any, All)):
            return self.__class__(*(Condition.bound_to(c, lhs, rhs) for c in self.clauses))
        if isinstance(self, Identity):
            lhs = self.lhs.bound_to(lhs)
            rhs = self.rhs.bound_to(rhs)
        elif isinstance(self, LeftAssignment):
            lhs = self.lhs.bound_to(lhs)
            rhs = self.rhs
        elif isinstance(self, RightAssignment):
            lhs = self.lhs
            rhs = self.rhs.bound_to(rhs)
        else:
            raise TypeError()
        return Equals(lhs, rhs)

    def literals(self):
        if isinstance(self, (Identity, LeftAssignment, RightAssignment)):
            yield self
        elif isinstance(self, (All, Any)):
            for c in self.clauses:
                yield from literals(c)
        raise TypeError()

    def sat(Φ, a=None):
        a = Assignment() if a is None else a
        Φ = a.evaluate(Φ)
        if Φ is True:
            return a
        if Φ is False:
            raise UnsatisfiableError()
        while not isinstance(Φ, bool):
            clauses = Φ.clauses if isinstance(Φ, All) else (Φ,)
            units = [c for c in clauses if not isinstance(c, Any)]
            if not units:
                break
            for l in units:
                a.add_literal(l)
            Φ = a.evaluate(Φ)
        if Φ is True:
            return a
        if Φ is False:
            raise UnsatisfiableError()
        for l in Condition.literals(Φ):
            aa = a.copy()
            aa.add_literal(l)
            try:
                return Condition.sat(Φ, aa)
            except UnsatisfiableError:
                pass
        raise UnsatisfiableError()


class Identity(typing.NamedTuple, Condition):
    lhs : Variable
    rhs : Variable

    def __repr__(self):
        return '(a.%s == b.%s)' % (self.lhs.name, self.rhs.name)


class LeftAssignment(typing.NamedTuple, Condition):
    lhs : Variable
    rhs : Value

    def __repr__(self):
        return '(a.%s := %s)' % (self.lhs.name, self.rhs.value)


class RightAssignment(typing.NamedTuple, Condition):
    lhs : Value
    rhs : Variable

    def __repr__(self):
        return '(b.%s := %s)' % (self.rhs.name, self.lhs.value)


def forwardRefToTypeVar(t):
    return typing.TypeVar(t.__forward_arg__) if isinstance(t, typing.ForwardRef) else t


class Equals:
    @staticmethod
    def fieldFromTypeArg(t):
        t = forwardRefToTypeVar(t)
        if isinstance(t, (Variable, Value)):
            return t
        elif isinstance(t, Cardinal):
            return Value(t.__value__)
        elif isinstance(t, typing.TypeVar):
            return Variable(0, t.__name__, None)
        else:
            return Value(t)

    def __new__(cls, lhs, rhs):
        lhs = Equals.fieldFromTypeArg(lhs)
        rhs = Equals.fieldFromTypeArg(rhs)
        if isinstance(lhs, Value) and isinstance(rhs, Value):
            return lhs == rhs
        elif isinstance(lhs, Value):
            return RightAssignment(lhs, rhs)
        elif isinstance(rhs, Value):
            return LeftAssignment(lhs, rhs)
        else:
            return Identity(lhs, rhs)


class All(Condition):
    __slots__ = ['clauses']

    def __new__(cls, *clauses):
        clauses = [
            cc
            for c in clauses
            for cc in (
                c.clauses if isinstance(c, All) else (c,)
            )
            if cc is not True
        ]
        if len(clauses) == 0:
            return True
        elif len(clauses) == 1:
            return clauses[0]
        elif any(c is False for c in clauses):
            return False
        else:
            condition = super().__new__(cls)
            condition.clauses = clauses
            return condition

    def __repr__(self):
        return 'All(%s)' % ', '.join(map(repr, self.clauses))


class Any(Condition):
    __slots__ = ['clauses']

    def __new__(cls, *clauses):
        clauses = [
            cc
            for c in clauses
            for cc in (
                c.clauses if isinstance(c, Any) else (c,)
            )
            if cc is not False
        ]
        if len(clauses) == 0:
            return False
        elif len(clauses) == 1:
            return clauses[0]
        elif any(c is True for c in clauses):
            return True
        elif any(isinstance(c, All) for c in clauses):
            # put in conjunctive normal form
            return All(*map(Any, itertools.product(*((c.clauses if isinstance(c, All) else (c,)) for c in clauses))))
        else:
            condition = super().__new__(cls)
            condition.clauses = clauses
            return condition

    def __repr__(self):
        return 'Any(%s)' % ', '.join(map(repr, self.clauses))


class Subtype:
    def __new__(cls, t1, t2):
        t1 = forwardRefToTypeVar(t1)
        t2 = forwardRefToTypeVar(t2)
        if isinstance(t1, str): t1 = typing.TypeVar(t1)
        if isinstance(t2, str): t2 = typing.TypeVar(t2)

        if isinstance(t1, typing.TypeVar) or isinstance(t2, typing.TypeVar): # n.b. Cardinal is a subclass of TypeVar
            return Equals(t1, t2)

        if typing.Any in (t1, t2):
            return True

        left_types = t1.__args__ if isinstance(t1, typing._GenericAlias) and t1.__origin__ is typing.Union else (t1,)
        right_types = t2.__args__ if isinstance(t2, typing._GenericAlias) and t2.__origin__ is typing.Union else (t2,)

        if len(left_types) * len(right_types) > 1:
            return All(*(Any(*(Subtype(l, r) for r in right_types)) for l in left_types))

        if not isinstance(t1, typing._GenericAlias) or not isinstance(t2, typing._GenericAlias):
            return t1 == t2

        t1 = left_types[0]
        t2 = right_types[0]

        o1, a1 = t1.__origin__, t1.__args__
        o2, a2 = t2.__origin__, t2.__args__

        if not issubclass(o1, o2):
            return False

        if o1 is collections.abc.Callable:
            if o2 is not collections.abc.Callable or len(a1) != len(a2):
                return False
            return All(Subtype(a1[-1], a2[-1]), *(Condition.reversed(Subtype(aa2, aa1)) for aa1, aa2 in zip(a1[:-1], a2[:-1])))

        if o1 is tuple:
            if o2 is not tuple or len(a1) != len(a2):
                return False
            return All(*(Subtype(aa1, aa2) for aa1, aa2 in zip(a1, a2)))

        if o1 is Array:
            s1, vt1 = a1
            s2, vt2 = a2
            s1 = s1.__args__
            s2 = s2.__args__
            if o2 is not Array or len(s1) != len(s2):
                return False
            return All(Equals(vt1, vt2), *(Equals(ss1, ss2) for ss1, ss2 in zip(s1, s2)))

        raise ValueError('Unhandled subtype condition: %s <= %s' % (t1, t2))


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        if self.parent[x] is not x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(x, y):
        x, y = self.find(x), self.find(y)
        if x is y:
            return
        if self.rank[x] < self.rank[y]:
            self.parent[x] = y
            return y
        elif self.rank[x] > self.rank[y]:
            self.parent[y] = x
            return x
        else:
            self.parent[y] = x
            self.rank[x] += 1
            return x

    def copy(self):
        other = UnionFind()
        other.parent.update(self.parent)
        other.rank.update(self.rank)

    def keys(self):
        return self.parent.keys()


class Assignment:
    def __init__(self):
        self.sets = UnionFind()
        self.values = {}

    def value(self, x):
        return self.values.get(self.sets.find(x))

    def add_identity(self, x, y):
        # Try to update this assignment with the information that x==y.
        # Iff this would contradict some other assignment, return False.
        x, y = self.sets.find(x), self.sets.find(y)
        if x is y:
            return True
        if x in self.values and y in self.values and self.values[x] != self.values[y]:
            return False
        v = self.values.pop(x, self.values.pop(y, None)) 
        x = self.sets.union(x, y)
        if v is not None:
            self.values[x] = v
        return True

    def add_value(self, x, y):
        # Try to update this assignment with the information that x:=y.
        # Iff this would contradict some other assignment, return False.
        x = self.sets.find(x)
        if x in self.values:
            return self.values[x] == y
        self.values[x] = y
        return True

    def add_literal(self, literal):
        if isinstance(literal, Identity):
            return self.add_identity(literal.lhs, literal.rhs)
        elif isinstance(literal, LeftAssignment):
            return self.add_value(literal.lhs, literal.rhs.value)
        elif isinstance(literal, RightAssignment):
            return self.add_value(literal.rhs, literal.lhs.value)

    def copy(self):
        other = super().__new__(self.__class__)
        other.sets = self.sets.copy()
        other.values = self.values.copy()

    def evaluate(self, condition):
        if isinstance(condition, bool):
            return condition
        if isinstance(condition, (Any, All)):
            return condition.__class__(*(self.evaluate(c) for c in condition.clauses))
        if isinstance(condition, Identity):
            lhs = self.sets.find(condition.lhs)
            rhs = self.sets.find(condition.rhs)
            if lhs is rhs:
                return True
            lhs = self.values.get(lhs)
            rhs = self.values.get(rhs)
            if lhs is not None and rhs is not None:
                return lhs == rhs
            return condition
        if isinstance(condition, LeftAssignment):
            lhs = self.value(condition.lhs)
            return lhs == condition.rhs.value if lhs is not None else condition
        if isinstance(condition, RightAssignment):
            rhs = self.value(condition.rhs)
            return rhs == condition.lhs.value if rhs is not None else condition
        raise TypeError()

    def apply(self):
        for variable in self.sets.keys():
            setattr(variable.target, variable.name, self.value(variable))
