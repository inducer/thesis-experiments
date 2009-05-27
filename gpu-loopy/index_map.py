from __future__ import division

import numpy
from pytools import Record, memoize_method
from pymbolic.mapper.dependency import DependencyMapper
from pymbolic.mapper.c_code import CCodeMapper
from pymbolic.mapper.stringifier import PREC_NONE

have_cuda = False

if have_cuda:
    import pycuda.autoinit
    import pycuda.driver as cuda
    import pycuda.compiler as compiler
    import pycuda.gpuarray as gpuarray
    import pycuda.curandom as curandom




AXES = ["x", "y", "z", "w"]




class BLOCK_IDX_TAG:
    def __init__(self, axis=None):
        self.axis = axis

    def __repr__(self):
        if self.axis is None:
            return "BLOCK_IDX"
        else:
            return "BLOCK_IDX(%d)" % self.axis

class THREAD_IDX_TAG:
    def __init__(self, axis=None):
        self.axis = axis

    def __repr__(self):
        if self.axis is None:
            return "THREAD_IDX"
        else:
            return "THREAD_IDX(%d)" % self.axis

class LoopDimension(Record):
    __slots__ = ["name", "length", "tag"]

    def __init__(self, name, length, tag=None):
        Record.__init__(self, name=name, length=length, tag=tag)




class LoopDomain(Record):
    __slots__ = ["dims"]

    def name_to_idx(self, name):
        for i, dim in enumerate(self.dims):
            if dim.name == name:
                return i
        else: 
            raise KeyError("invalid dimension name: %s" % name)

    def name_to_dim(self, name):
        for dim in self.dims:
            if dim.name == name:
                return dim
        else: 
            raise KeyError("invalid dimension name: %s" % name)

    def tag_to_idx(self, tag):
        for i, dim in enumerate(self.dims):
            if dim.tag == tag:
                return i
        else: 
            raise KeyError("invalid tag: %s" % tag)

    def indices_by_tag_type(self, tag_type):
        return [i for i, dim in enumerate(self.dims)
                if isinstance(dim.tag, tag_type)]

    def dims_by_tag_type(self, tag_type):
        return [dim for dim in self.dims
                if isinstance(dim.tag, tag_type)]

    def dims_by_tag(self, tag):
        return [dim for dim in self.dims if dim.tag == tag]

    def set_dim(self, idx, new_dim):
        return self.copy(dims=
                self.dims[:idx] 
                + [new_dim] 
                + self.dims[(idx+1):])

    def change_dim(self, idx, **kwargs):
        return self.set_dim(idx, self.dims[idx].copy(**kwargs))

    def note_substitute(self, old_var, new_expr):
        pass

    def move(self, from_idx, to_idx):
        new_dims = self.dims[:idx] + self.dims[(idx+1):]
        if from_idx > to_idx:
            to_idx -= 1
        new_dims.insert(to_idx, self.dims[from_idx])
        return self.copy(dims=new_dims)



        
class LoopKernel(LoopDomain):
    @memoize_method
    def all_indices(self):
        return set(dim.name for dim in self.dims)

    @memoize_method
    def output_indices(self):
        dm = DependencyMapper(include_subscripts=False)

        output_indices = set()
        for lvalue, expr in self.instructions:
            output_indices.update(
                    set(v.name for v in dm(lvalue)) 
                    & self.all_indices())

        return output_indices

    @memoize_method
    def output_dimensions(self):
        return [dim for dim in self.dims_by_tag(None)
                if dim.name in self.output_indices()]

    @memoize_method
    def reduction_dimensions(self):
        return [dim for dim in self.dims_by_tag(None)
                if dim.name not in self.output_indices()]

    @memoize_method
    def input_vectors(self):
        dm = DependencyMapper(include_subscripts=False)

        input_vectors = set()
        for lvalue, expr in self.instructions:
            input_vectors.update(
                    set(v.name for v in dm(expr)) 
                    - self.all_indices())
        return input_vectors

    @memoize_method
    def output_vectors(self):
        dm = DependencyMapper(include_subscripts=False)

        output_vectors = set()
        for lvalue, expr in self.instructions:
            output_vectors.update(
                    set(v.name for v in dm(lvalue)) 
                    - self.all_indices())
        return list(output_vectors)

    def _subst_insns(self, old_var, new_expr):
        from pymbolic.mapper.substitutor import substitute

        subst_map = {old_var: new_expr}

        return [(substitute(lvalue, subst_map),
            substitute(expr, subst_map))
            for lvalue, expr in self.instructions]

    def substitute(self, old_var, new_expr):
        return self.copy(instructions=self._subst_insns(old_var, new_expr))

    def split_dimension(self, idx, inner_length, outer_name=None, inner_name=None,
            outer_tag=None, inner_tag=None):
        dim = self.dims[idx]

        if outer_name is None:
            outer_name = dim.name+"_outer"
        if inner_name is None:
            inner_name = dim.name+"_inner"

        assert dim.length % inner_length == 0

        from pymbolic import var
        self.substitute(dim.name, 
                var(inner_name) 
                + var(outer_name)*inner_length)

        return self.copy(dims=
                self.dims[:idx] + [
                    LoopDimension(
                        name=dim.name, 
                        length=dim.length//inner_length,
                        tag=outer_tag),
                    LoopDimension(
                        name=inner_name, 
                        length=inner_length,
                        tag=inner_tag),
                    ]
                + self.dims[(idx+1):])




def make_loop_kernel(dims, insns):
    from pymbolic import parse
    insns = [(parse(lvalue), parse(expr)) 
            for lvalue, expr in insns]

    return LoopKernel(dims=dims, instructions=insns)




def generate_thread_index_assignment_numberings(kernel):
    thread_idx_dim_indices = kernel.indices_by_tag_type(
            THREAD_IDX_TAG)

    if thread_idx_dim_indices:
        from pytools import generate_unique_permutations

        for perm in generate_unique_permutations(
                tuple(range(len(thread_idx_dim_indices)))):

            new_kernel = kernel
            for dim_i, thread_axis in zip(
                    thread_idx_dim_indices,
                    perm):
                new_kernel = new_kernel.change_dim(
                        dim_i, tag=THREAD_IDX_TAG(thread_axis))

            yield new_kernel
    else:
        # nothing assigned to thread indices? not interested.
        pass




def generate_dim_assignments(kernel, idx=0,
        no_thread_indices=set()):
    if idx >= len(kernel.dims):
        for knl in generate_thread_index_assignment_numberings(
                kernel):
            yield knl
        return

    dim = kernel.dims[idx]

    assert dim.length >= 2

    for knl in generate_dim_assignments(kernel, idx+1,
            no_thread_indices=no_thread_indices):
        yield knl

    from pymbolic import var

    block_idx_dim_count = len(kernel.dims_by_tag_type(BLOCK_IDX_TAG))

    if dim.name in kernel.output_indices() \
            and block_idx_dim_count < 2 :
        for knl in generate_dim_assignments(
                kernel.change_dim(idx, 
                    tag=BLOCK_IDX_TAG(block_idx_dim_count)),
                idx+1,
                no_thread_indices=no_thread_indices):
            yield knl

    # try to assign to thread indices
    thread_idx_dims = kernel.dims_by_tag_type(THREAD_IDX_TAG)
    thread_idx_dims_count = len(thread_idx_dims)

    from pytools import product
    assigned_block_size  = product(tid.length
            for tid in thread_idx_dims)
    leftover_block_size = 512 // assigned_block_size

    if (dim.name in kernel.output_indices()
            and dim.name not in no_thread_indices
            and thread_idx_dims_count < 3 
            and leftover_block_size > 1):
        my_block_length = 1
        while my_block_length < dim.length:
            my_block_length *= 2
            if my_block_length > dim.length:
                my_block_length = dim.length

            if my_block_length > leftover_block_size:
                break

            if dim.length % my_block_length != 0:
                break

            new_length = dim.length//my_block_length
            
            if new_length > 1:
                outer_name = dim.name+"_outer"
                inner_name = dim.name+"_inner"

                for knl in generate_dim_assignments(
                        kernel.split_dimension(idx, 
                            inner_length=new_length,
                            outer_name=outer_name,
                            outer_tag=THREAD_IDX_TAG(),
                            inner_name=inner_name),
                        idx+1,
                        no_thread_indices=(
                            no_thread_indices | set([inner_name]))):
                    yield knl
            else:
                for knl in generate_dim_assignments(
                        kernel.change_dim(idx, tag=THREAD_IDX_TAG),
                        idx+1,
                        no_thread_indices=no_thread_indices):
                    yield knl




class LoopyCCodeMapper(CCodeMapper):
    def __init__(self, kernel):
        CCodeMapper.__init__(self)
        self.kernel = kernel

    def map_subscript(self, expr, enclosing_prec):
        from pymbolic.primitives import Variable
        if (isinstance(expr.aggregate, Variable)
                and expr.aggregate.name in self.kernel.input_vectors()):
            return "tex1Dfetch(tex_%s, %s)" % (
                    expr.aggregate.name,
                    self.rec(expr.index, PREC_NONE))
        else:
            return CCodeMapper.map_subscript(self, expr, enclosing_prec)

    def map_variable(self, expr, enclosing_prec):
        try:
            dim = self.kernel.name_to_dim(expr.name)
        except KeyError:
            return CCodeMapper.map_variable(self, expr, enclosing_prec)
        else:
            if isinstance(dim.tag, THREAD_IDX_TAG):
                return "threadIdx."+AXES[dim.tag.axis]
            elif isinstance(dim.tag, BLOCK_IDX_TAG):
                return "blockIdx."+AXES[dim.tag.axis]
            else:
                return CCodeMapper.map_variable(self, expr, enclosing_prec)







def generate_code(kernel):
    from codepy.cgen import FunctionBody, FunctionDeclaration, \
            Typedef, POD, Value, Pointer, Module, Block, \
            Initializer, Assign, Statement, For

    from pymbolic.primitives import Subscript
    ccm = LoopyCCodeMapper(kernel)

    inner = Block([])
    for lvalue, expr in kernel.instructions:
        assert isinstance(lvalue, Subscript)
        name = lvalue.aggregate.name
        inner.append(Statement("tmp_%s += %s"
            % (name, ccm(expr, PREC_NONE))))

    for dim in kernel.reduction_dimensions():
        inner = For(
                "int %s = 0" % dim.name,
                "%s < %s" % (dim.name, dim.length),
                "++%s" % dim.name, inner)

    inner = Block(
            [Initializer(POD(numpy.float32, 
                "tmp_"+lvalue.aggregate.name), 0)
                for lvalue, expr in kernel.instructions]
            +[inner]+
            [Assign(
                ccm(lvalue, PREC_NONE),
                "tmp_"+lvalue.aggregate.name)
                for lvalue, expr in kernel.instructions])

    for loop in kernel.output_dimensions():
        inner = For(
                "int %s = 0" % loop.name,
                "%s < %s" % (loop.name, loop.length),
                "++%s" % loop.name, inner)

    from codepy.cgen.cuda import CudaGlobal

    mod = Module()

    for v in kernel.input_vectors():
        mod.append(
                Value("texture<float, 1, cudaReadModeElementType>",
                    "tex_"+v));

    mod.append(
        FunctionBody(
            CudaGlobal(FunctionDeclaration(
                Value("void", "loopy_kernel"),
                [Pointer(POD(numpy.float32, name)) 
                    for name in kernel.output_vectors()])),
            Block([inner])))

    return str(mod)




def main():
    n = 16*34
    if have_cuda:
        a = curandom.rand((n, n))
        b = curandom.rand((n, n))
        c = gpuarray.empty_like(a)

    def bogus_launcher(grid, kernel, texref_lookup):
        a.bind_to_texref_ext(texref_lookup["a"])
        b.bind_to_texref_ext(texref_lookup["b"])
        kernel.prepared_call(grid, c.gpudata)

    k = make_loop_kernel([
        LoopDimension("i", n),
        LoopDimension("j", n),
        LoopDimension("k", n),
        ], [ 
        ("c[i+16*34*j]", "a[i+16*34*k]*b[k+16*34*j]") 
        ])

    for knl in generate_dim_assignments(k):
        for d in knl.dims:
            print d
        print
        print generate_code(knl)
        raw_input()




if __name__ == "__main__":
    main()
