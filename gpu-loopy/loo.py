from __future__ import division

import numpy
from pytools import Record, memoize_method
from pymbolic.mapper.dependency import DependencyMapper
from pymbolic.mapper.c_code import CCodeMapper
from pymbolic.mapper.stringifier import PREC_NONE
from pymbolic.mapper import CombineMapper



# TODO: Correctness checking
# TODO: More freedom for data types of input and output vectors
# TODO: Try different kernels
# TODO: Non-multiple loop splits





SMEM_BYTES = 16384 - 256 # allow for parameter space
MAX_THREADS_PER_BLOCK = 512

try:
    import pycuda.autoinit
except:
    HAVE_CUDA = False
else:
    HAVE_CUDA = True
    import pycuda.driver as cuda
    import pycuda.compiler as compiler
    import pycuda.gpuarray as gpuarray
    import pycuda.curandom as curandom




AXES = ["x", "y", "z", "w"]




class IndexTag(object):
    pass

class BLOCK_IDX_TAG(IndexTag):
    def __init__(self, axis=None):
        self.axis = axis

    def __repr__(self):
        if self.axis is None:
            return "BLOCK_IDX"
        else:
            return "BLOCK_IDX(%d)" % self.axis

    def __eq__(self, other):
        return (self.__class__ == other.__class__
                and self.axis == other.axis)

    def __ne__(self, other):
        return not self.__eq__(other)

class THREAD_IDX_TAG(IndexTag):
    def __init__(self, axis=None):
        self.axis = axis

    def __repr__(self):
        if self.axis is None:
            return "THREAD_IDX"
        else:
            return "THREAD_IDX(%d)" % self.axis

    def __eq__(self, other):
        return (self.__class__ == other.__class__
                and self.axis == other.axis)

    def __ne__(self, other):
        return not self.__eq__(other)

class LoopDimension(Record):
    __slots__ = ["name", "length", "tag"]

    def __init__(self, name, length, tag=None):
        Record.__init__(self, name=name, length=length, tag=tag)

        if tag is not None:
            assert isinstance(tag, IndexTag)

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        if self.tag is not None:
            return "LD(%r, %d, %s)" % (self.name, self.length, self.tag)
        else:
            return "LD(%r, %d)" % (self.name, self.length)





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
        raise KeyError("invalid tag: %s" % tag)

    def tag_to_dim(self, tag):
        return self.dims[self.tag_to_idx(tag)]

    def indices_by_tag_type(self, tag_type):
        return [i for i, dim in enumerate(self.dims)
                if isinstance(dim.tag, tag_type)]

    def dims_by_tag_type(self, tag_type):
        return [dim for dim in self.dims
                if isinstance(dim.tag, tag_type)]

    def ordered_dim_by_tag_type(self, tag_type):
        result = []
        from itertools import count
        for i in count():
            try:
                dim = self.tag_to_dim(tag_type(i))
            except KeyError:
                return result
            else:
                result.append(dim)

        return result

    def dims_by_tag(self, tag):
        return [dim for dim in self.dims if dim.tag == tag]

    def set_dim(self, idx, new_dim):
        return self.copy(dims=
                self.dims[:idx] 
                + [new_dim] 
                + self.dims[(idx+1):])

    def change_dim(self, idx, **kwargs):
        return self.set_dim(idx, self.dims[idx].copy(**kwargs))

    def move(self, from_idx, to_idx):
        new_dims = self.dims[:idx] + self.dims[(idx+1):]
        if from_idx > to_idx:
            to_idx -= 1
        new_dims.insert(to_idx, self.dims[from_idx])
        return self.copy(dims=new_dims)



        
class LoopKernel(LoopDomain):
    # possible attributes:
    # - dims from LoopDomain
    # - instructions
    # - prefetch
    # - schedule

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
        return [dim for dim in self.dims if dim.name in self.output_indices()]

    @memoize_method
    def reduction_dimensions(self):
        return [dim for dim in self.dims if dim.name not in self.output_indices()]

    def grid_dim(self):
        dims = self.ordered_dim_by_tag_type(BLOCK_IDX_TAG)
        return [dim.length for dim in dims] + [1]*(2-len(dims))

    def block_dim(self):
        dims = self.ordered_dim_by_tag_type(THREAD_IDX_TAG)
        return [dim.length for dim in dims] + [1]*(3-len(dims))

    def thread_count(self):
        from pytools import product
        return product(self.block_dim())

    def block_count(self):
        from pytools import product
        return product(self.grid_dim())

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

    def is_prefetch_variable(self, varname):
        if hasattr(self, "prefetch"):
            for pf in self.prefetch.itervalues():
                for pfdim in pf.dims:
                    if pfdim.name == varname:
                        return True

        return False

    def _subst_prefetch(self, old_var, new_expr):
        from pymbolic.mapper.substitutor import substitute
        subst_map = {old_var: new_expr}

        result = {}
        for pf in self.prefetch.itervalues():
            for pfdim in pf.dims:
                if pfdim.name == old_var:
                    raise RuntimeError("can't substitute prefetch dimension %s"
                            % old_var)

            new_pf = pf.copy(index_expr=substitute(pf.index_expr, subst_map))
            result[pf.input_vector, new_pf.index_expr] = new_pf

        return result

    def substitute(self, old_var, new_expr):
        copy = self.copy(instructions=self._subst_insns(old_var, new_expr))
        if hasattr(self, "prefetch"):
            copy.prefetch = self._subst_prefetch(old_var, new_expr)

        if hasattr(self, "schedule"):
            for sched_item in self.schedule:
                if (isinstance(sched_item, LoopDimension) 
                        and sched_item.name == old_var):
                    raise RuntimeError("can't substitute already-scheduled variable: %s"
                            % old_var)
                    
        return copy

    def split_dimension(self, idx, inner_length, outer_name=None, inner_name=None,
            outer_tag=None, inner_tag=None):
        dim = self.dims[idx]

        if outer_name is None:
            outer_name = dim.name+"_outer"
        if inner_name is None:
            inner_name = dim.name+"_inner"

        assert dim.length % inner_length == 0

        from pymbolic import var
        tgt_expr = var(inner_name) + var(outer_name)*inner_length

        return self \
                .substitute(dim.name, tgt_expr) \
                .copy(dims=
                        self.dims[:idx] + [
                            LoopDimension(
                                name=outer_name, 
                                length=dim.length//inner_length,
                                tag=outer_tag),
                            LoopDimension(
                                name=inner_name, 
                                length=inner_length,
                                tag=inner_tag),
                            ]
                        + self.dims[(idx+1):]), tgt_expr




def make_loop_kernel(dims, insns):
    from pymbolic import parse

    def parse_if_necessary(v):
        if isinstance(v, str):
            return parse(v)
        else:
            return v

    insns = [
            (parse_if_necessary(lvalue), 
                parse_if_necessary(expr)) 
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
    leftover_block_size = MAX_THREADS_PER_BLOCK // assigned_block_size

    if (dim.name in kernel.output_indices()
            and dim.name not in no_thread_indices
            and dim.tag is None
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

                new_kernel, tgt_expr = kernel.split_dimension(idx, 
                            outer_name=outer_name,
                            inner_length=my_block_length,
                            inner_name=inner_name,
                            inner_tag=THREAD_IDX_TAG(),
                            )
                for knl in generate_dim_assignments(new_kernel, idx,
                        no_thread_indices=(
                            no_thread_indices | set([outer_name]))):
                    yield knl
            else:
                for knl in generate_dim_assignments(
                        kernel.change_dim(idx, tag=THREAD_IDX_TAG()),
                        idx+1,
                        no_thread_indices=no_thread_indices):
                    yield knl




# prefetch-related ------------------------------------------------------------
def total_prefetch_size(kernel):
    return sum(pf.size() for pf in kernel.prefetch.itervalues())

class PrefetchDescriptor(Record):
    # possible attributes:
    # - input_vector
    # - index_expr
    # - dims
    # - dim_storage_lengths

    def size(self):
        from pytools import product
        return 4*product(dim.length for dim in self.dims)

    @memoize_method
    def free_variables(self):
        return set(var.name 
                for var in DependencyMapper()(self.index_expr)
                ) - set(dim.name for dim in self.dims)



def with_added_prefetch_dim(kernel, ivec, iexpr, dim):
    new_prefetch = kernel.prefetch.copy()
    if (ivec, iexpr) in new_prefetch:
        old_pf_descr = new_prefetch[ivec, iexpr]
        new_prefetch[ivec, iexpr] = old_pf_descr.copy(
                dims=old_pf_descr.dims + [dim])
    else:
        new_prefetch[ivec, iexpr] = PrefetchDescriptor(
                input_vector=ivec,
                index_expr=iexpr,
                dims=[dim])

    new_kernel = kernel.copy(prefetch=new_prefetch)

    if total_prefetch_size(new_kernel) <= SMEM_BYTES:
        return new_kernel
    else:
        return None




def generate_prefetch_sizes(kernel, ivec, iexpr, prefetch_dims):
    if not prefetch_dims:
        yield kernel
        return

    dim = prefetch_dims[0]

    if (isinstance(dim.tag, THREAD_IDX_TAG) 
            or kernel.is_prefetch_variable(dim.name)):
        new_kernel = with_added_prefetch_dim(kernel, ivec, iexpr, dim)
        if new_kernel is not None:
            for knl in generate_prefetch_sizes(
                    new_kernel, ivec, iexpr, prefetch_dims[1:]):
                yield knl
    else:
        prefetch_length = 2
        while prefetch_length < dim.length:
            if prefetch_length > dim.length:
                prefetch_length = dim.length

            if dim.length % prefetch_length != 0:
                break

            outer_length = dim.length//prefetch_length
            if outer_length > 1:
                # split the dimension, then generate prefetch
                inner_name = dim.name+"_prefetch"

                new_kernel, tgt_expr = kernel.split_dimension(
                        kernel.name_to_idx(dim.name),
                        inner_length=prefetch_length,
                        inner_name=inner_name)

                from pymbolic import var
                from pymbolic.mapper.substitutor import substitute

                new_iexpr = substitute(iexpr, {var(dim.name): tgt_expr})
                new_kernel = with_added_prefetch_dim(
                        new_kernel, ivec, new_iexpr,
                        LoopDimension(inner_name, prefetch_length))

                if new_kernel is not None:
                    for knl in generate_prefetch_sizes(new_kernel, 
                            ivec, new_iexpr, prefetch_dims[1:]):
                        yield knl
            else:
                # prefetch the whole dimension
                new_kernel = with_added_prefetch_dim(kernel, ivec, iexpr, dim)
                if new_kernel is not None:
                    for knl in generate_prefetch_sizes(
                            new_kernel, ivec, iexpr, prefetch_dims[1:]):
                        yield knl

            prefetch_length *= 2




class VariableIndexExpressionCollector(CombineMapper):
    def __init__(self, tgt_vector_name):
        self.tgt_vector_name = tgt_vector_name

    def combine(self, values):
        from pytools import flatten
        return set(flatten(values))

    def map_constant(self, expr):
        return set()

    def map_algebraic_leaf(self, expr):
        return set()

    def map_subscript(self, expr):
        from pymbolic.primitives import Variable
        assert isinstance(expr.aggregate, Variable)

        if expr.aggregate.name == self.tgt_vector_name:
            return set([expr.index])
        else:
            return CombineMapper.map_subscript(self, expr)




def generate_single_vector_kernel_prefetch_choices(kernel, ivec,
        force_prefetch):
    from pytools import flatten
    index_exprs = set(flatten(
        VariableIndexExpressionCollector(ivec)(expression)
        for lvalue, expression in kernel.instructions
        ))

    for index_expr in index_exprs:
        dm = DependencyMapper()

        involved_dims = list(set(kernel.name_to_dim(idx.name)
            for idx in dm(index_expr)))

        prefetch_dims = [dim
                for dim in involved_dims
                if isinstance(dim.tag, THREAD_IDX_TAG)]
        uncertain_dims = [dim
                for dim in involved_dims
                if not isinstance(dim.tag, (THREAD_IDX_TAG, BLOCK_IDX_TAG))]

        if force_prefetch == True:
            choices = [[True]*len(uncertain_dims)]
        elif force_prefetch == False:
            choices = [[True]*len(uncertain_dims)]
        else:
            assert force_prefetch is None
            from pytools import generate_nonnegative_integer_tuples_below as gnitt
            choices = gnitt(2, len(uncertain_dims))

        for flags in choices:
            my_prefetch_dims = prefetch_dims + [
                    udim for udim, flag in zip(uncertain_dims, flags)
                    if flag]
            for knl in generate_prefetch_sizes(kernel, 
                    ivec, index_expr, my_prefetch_dims):
                yield knl





def optimize_prefetch(kernel):
    new_prefetch = {}

    pf_list = kernel.prefetch.values()

    from pytools import partition2, product

    for i_pf, pf in enumerate(pf_list):
        # reorder to get threadIdx.x to the bottom
        thread_pf_dims, non_thread_pf_dims = partition2(
                (isinstance(pfdim.tag, THREAD_IDX_TAG), pfdim)
                for pfdim in pf.dims)
        thread_pf_dims.sort(key=lambda pfdim: pfdim.tag.axis)

        # try and avoid bank conflicts
        pfdims = non_thread_pf_dims+thread_pf_dims[::-1]
        dim_storage_lengths = [pfdim.length for pfdim in pfdims]

        if thread_pf_dims and thread_pf_dims[0].tag.axis == 0:
            tx_dim = thread_pf_dims[0]

            # see if we can afford the smem expense in avoiding bank conflicts
            other_pf_sizes = sum(opf.size() 
                    for opf in pf_list[:i_pf]+pf_list[i_pf+1:])
            other_dim_sizes = 4*product(odim.length
                    for odim in thread_pf_dims[1:]+non_thread_pf_dims)

            if (tx_dim.length % 2 == 0 
                    and other_pf_sizes+other_dim_sizes*(tx_dim.length+1) 
                    < SMEM_BYTES):
                dim_storage_lengths[-1] += 1

        new_prefetch[pf.input_vector, pf.index_expr] = \
                pf.copy(dims=pfdims, dim_storage_lengths=dim_storage_lengths)

    return kernel.copy(prefetch=new_prefetch)






def generate_kernel_prefetch_choices(kernel, ivecs, prefetch_hints):
    if ivecs:
        for knl in generate_single_vector_kernel_prefetch_choices(
                kernel, ivecs[0], prefetch_hints.get(ivecs[0])):
            for subknl in generate_kernel_prefetch_choices(
                    knl, ivecs[1:], prefetch_hints):
                yield subknl
    else:
        yield kernel




def generate_all_prefetching_kernels(kernel, prefetch_hints):
    kernel = kernel.copy(prefetch={})

    from pytools import generate_nonnegative_integer_tuples_below as gnitt
    ivecs = kernel.input_vectors()

    certain_yes_ivecs = []
    uncertain_ivecs = []

    for iv in ivecs:
        hint = prefetch_hints.get(iv)
        if hint == True:
            certain_yes_ivecs.append(iv)
        elif hint is None:
            uncertain_ivecs.append(iv)

    for flags in gnitt(2, len(uncertain_ivecs)):
        for knl in generate_kernel_prefetch_choices(kernel,
                certain_yes_ivecs
                + [ivec for flag, ivec in zip(flags, uncertain_ivecs) if flag],
                prefetch_hints):
            yield optimize_prefetch(knl)






# loop scheduling -------------------------------------------------------------
def generate_loop_schedules(kernel):
    prev_schedule = getattr(kernel, "schedule", 
            kernel.dims_by_tag_type(BLOCK_IDX_TAG)
            + kernel.dims_by_tag_type(THREAD_IDX_TAG))

    already_scheduled = set(sch_item 
            for sch_item in prev_schedule
            if isinstance(sch_item, LoopDimension))

    # have a schedulable prefetch? load, schedule it
    scheduled_names = set(dim.name for dim in already_scheduled)

    had_usable_prefetch = False
    scheduled_thread_dim_names = set(
            dim.name for dim in already_scheduled
            if isinstance(dim.tag, THREAD_IDX_TAG))

    for pf in kernel.prefetch.itervalues():
        # already scheduled? never mind then.
        if pf in prev_schedule:
            continue

        # a free variable not known yet? then we're not ready
        if not pf.free_variables() <= scheduled_names:
            continue

        # a prefetch variable already scheduled, but not borrowable?
        # (only thread index variables are borrowable)
        pf_loop_names = set(dim.name for dim in pf.dims)

        if pf_loop_names & (already_scheduled - scheduled_thread_dim_names):
            # dead end: we won't be able to schedule this prefetch
            # in this branch. at least one of its loop dimensions
            # was already scheduled, and that dimension is not 
            # borrowable.
            print "UNSCHEDULABLE:"
            print_kernel_info(kernel)
            raw_input()
            return

        new_kernel = kernel.copy(schedule=prev_schedule+[pf])
        for knl in generate_loop_schedules(new_kernel):
            had_usable_prefetch = True
            yield knl

    if had_usable_prefetch:
        return

    # Build set of potentially schedulable variables
    schedulable = set(kernel.dims)

    # Don't re-schedule already scheduled variables
    schedulable -= already_scheduled

    # Don't schedule reduction variables until all output
    # variables are taken care of. Once they are, schedule
    # output writing.
    serial_output_dims = set(od for od in kernel.output_dimensions() 
            if od.tag is None)

    if not serial_output_dims <= already_scheduled:
        schedulable -= set(kernel.reduction_dimensions())
    else:
        if not any(isinstance(sch_item, WriteOutput) 
                for sch_item in prev_schedule):
            kernel = kernel.copy(
                    schedule=prev_schedule + [WriteOutput()])
            prev_schedule = kernel.schedule

    # Don't schedule variables that are prefetch axes 
    # for not-yet-scheduled prefetches.
    unsched_prefetch_axes = set(dim
            for pf in kernel.prefetch.itervalues()
            if pf not in prev_schedule
            for dim in pf.dims)
    schedulable -= unsched_prefetch_axes

    if schedulable:
        # have a schedulable variable? schedule a loop for it, recurse
        for dim in schedulable:
            new_kernel = kernel.copy(schedule=prev_schedule+[dim])
            for knl in generate_loop_schedules(new_kernel):
                yield knl
    else:
        # all loop dimensions and prefetches scheduled?
        # great! yield the finished product if it is complete

        all_dims_scheduled = len(already_scheduled) == len(kernel.dims)
        all_pf_scheduled =  len(set(sch_item for sch_item in prev_schedule
            if isinstance(sch_item, PrefetchDescriptor))) == len(kernel.prefetch)
        output_scheduled = len(set(sch_item for sch_item in prev_schedule
            if isinstance(sch_item, WriteOutput))) == 1

        if all_dims_scheduled and all_pf_scheduled and output_scheduled:
            yield kernel
    
    



# register prefetches ---------------------------------------------------------
class AllIndexExpressionCollector(CombineMapper):
    def combine(self, values):
        from pytools import flatten
        return set(flatten(values))

    def map_constant(self, expr):
        return set()

    def map_algebraic_leaf(self, expr):
        return set()

    def map_subscript(self, expr):
        from pymbolic.primitives import Variable
        assert isinstance(expr.aggregate, Variable)

        return set([expr])




def insert_register_prefetches(kernel):
    reg_pf = {}

    total_loop_count = len(kernel.all_indices())
    known_vars = set()

    unused_index_exprs = set()
    for tgt, expr in kernel.instructions:
        unused_index_exprs |= AllIndexExpressionCollector()(expr)
    unused_index_exprs = [
            (iexpr, set(v.name for v in DependencyMapper()(iexpr.index)))
            for iexpr in unused_index_exprs]

    schedule = kernel.schedule[:]

    sched_index = 0
    loop_count = 0
    while sched_index < len(schedule):
        sched_item = schedule[sched_index]
        if isinstance(sched_item, LoopDimension):
            known_vars.add(sched_item.name)
            loop_count += 1
        sched_index += 1

        if loop_count < total_loop_count:
            i = 0
            while i < len(unused_index_exprs):
                iexpr, index_deps = unused_index_exprs[i]
                if (index_deps <= known_vars 
                        and (iexpr.aggregate.name, iexpr.index)
                            not in kernel.prefetch):
                    unused_index_exprs.pop(i)
                    new_name = "reg_prefetch_"+iexpr.aggregate.name+str(sched_index)
                    reg_pf[iexpr] = new_name
                    schedule.insert(sched_index,
                            RegisterPrefetch(
                                index_expr=iexpr, new_name=new_name))
                    sched_index += 1
                else:
                    i += 1

    return kernel.copy(schedule=schedule, register_prefetch=reg_pf)




# code generation -------------------------------------------------------------
class LoopyCCodeMapper(CCodeMapper):
    def __init__(self, kernel, get_prefetch_name):
        def constant_mapper(c):
            if isinstance(c, float):
                return "%sf" % repr(c)
            else:
                return repr(c)

        CCodeMapper.__init__(self, constant_mapper=constant_mapper)
        self.kernel = kernel
        self.get_prefetch_name = get_prefetch_name

    def map_subscript(self, expr, enclosing_prec):
        from pymbolic.primitives import Variable
        if (isinstance(expr.aggregate, Variable)
                and expr.aggregate.name in self.kernel.input_vectors()):
            try:
                pf = self.kernel.prefetch[expr.aggregate.name, expr.index]
            except KeyError:
                try:
                    return self.kernel.register_prefetch[expr]
                except KeyError:
                    return "tex1Dfetch(tex_%s, %s)" % (
                            expr.aggregate.name,
                            self.rec(expr.index, PREC_NONE))
            else:
                return self.get_prefetch_name(pf)+"".join(
                        "[%s]" % dim.name for dim in pf.dims)
        else:
            return CCodeMapper.map_subscript(self, expr, enclosing_prec)





class WriteOutput(Record):
    pass

class RegisterPrefetch(Record):
    __slots__ = ["index_expr", "new_name"]




def make_fetch_index_expr(kernel, exclude):
    from pymbolic import var
    expr = 0
    for dim in kernel.ordered_dim_by_tag_type(THREAD_IDX_TAG)[::-1]:
        if dim in exclude:
            continue

        expr = expr*dim.length + var("threadIdx." + AXES[dim.tag.axis])

    return expr




def generate_code(kernel):
    from codepy.cgen import FunctionBody, FunctionDeclaration, \
            Typedef, POD, Value, Pointer, Module, Block, \
            Initializer, Assign, Statement, For, ArrayOf, \
            Define, If, Line, Comment, MaybeUnused

    from codepy.cgen.cuda import CudaGlobal, CudaShared

    S = Statement

    from pymbolic.primitives import Subscript
    from pymbolic import var

    # prefetch name assignment

    def get_prefetch_name(pf):
        try:
            return prefetch_names[pf]
        except KeyError:
            nm = "prefetch_%s_%d" % (pf.input_vector, len(prefetch_names))
            prefetch_names[pf] = nm
            return nm

    prefetch_names = {}

    ccm = LoopyCCodeMapper(kernel, get_prefetch_name)

    # write innermost loop body
    inner = Block([])
    for lvalue, expr in kernel.instructions:
        assert isinstance(lvalue, Subscript)
        name = lvalue.aggregate.name
        inner.append(S("tmp_%s += %s"
            % (name, ccm(expr, PREC_NONE))))

    thread_count = kernel.thread_count()

    # we're progressing from the innermost (last in the schedule) 
    # to the outermost loop

    schedule = kernel.schedule[::-1]
    for sched_index, sched_item in enumerate(schedule):
        # find surrounding schedule items
        if sched_index > 0:
            next_inner_sched_item = schedule[sched_index-1]
        else:
            next_inner_sched_item = None
            
        if sched_index+1 < len(schedule):
            next_outer_sched_item = schedule[sched_index+1]
        else:
            next_outer_sched_item = None

        # write code for loops
        if isinstance(sched_item, LoopDimension):
            dim = sched_item
            if dim.tag is None:
                inner = For(
                        "int %s = 0" % dim.name,
                        "%s < %s" % (dim.name, dim.length),
                        "++%s" % dim.name, inner)

        # write code for output writes
        elif isinstance(sched_item, WriteOutput):
            inner = Block(
                    [Initializer(POD(numpy.float32, 
                        "tmp_"+lvalue.aggregate.name), 0)
                        for lvalue, expr in kernel.instructions]
                    +[inner]+
                    [Assign(
                        ccm(lvalue, PREC_NONE),
                        "tmp_"+lvalue.aggregate.name)
                        for lvalue, expr in kernel.instructions])

        # write code for prefetches
        elif isinstance(sched_item, PrefetchDescriptor):
            pf = sched_item
            pf_name = get_prefetch_name(pf)

            # build smem array declarator
            smem_pf_array = POD(numpy.float32, pf_name)
            for l in pf.dim_storage_lengths:
                smem_pf_array = ArrayOf(smem_pf_array, l)
            smem_pf_array = CudaShared(smem_pf_array)

            # figure out dimensions
            from pytools import partition2
            thread_pf_dims, non_thread_pf_dims = partition2(
                    (isinstance(dim.tag, THREAD_IDX_TAG), dim)
                    for dim in pf.dims)

            # start writing fetch code block
            fetch_block = Block([
                    Initializer(
                        MaybeUnused(POD(numpy.uint32, "fetch_idx")),
                        make_fetch_index_expr(kernel, thread_pf_dims))
                    ])
                        
            # Prefetch needs to take types of dimensions into account:
            #
            # - non-thread prefetch dimensions (1)
            # - thread prefetch dimensions (2)
            # - non-prefetch thread dimensions (3)
            #
            # Type (1) comprises a pool of fetches that must be efficiently
            # fetched with the "extra parallelism" available through (3).

            from pytools import product
            non_thread_pf_size = product(dim.length for dim in non_thread_pf_dims)
            thread_pf_size = product(dim.length for dim in thread_pf_dims)
            non_pf_thread_size = product(
                    dim.length for dim in kernel.dims_by_tag_type(THREAD_IDX_TAG)
                    if dim not in thread_pf_dims)

            # generate indexing substitution map and smem assignment indices
            pf_indices = []
            pf_idx_subst_map = {}
            prev_dim_sizes = 1
            for i, pf_dim in enumerate(pf.dims):
                if isinstance(pf_dim.tag, THREAD_IDX_TAG):
                    dim_expr = "threadIdx.%s" % AXES[pf_dim.tag.axis]
                    pf_indices.append(dim_expr)
                    pf_idx_subst_map[pf_dim.name] = var(dim_expr)
                else:
                    dim_expr = var("fetch_idx") / prev_dim_sizes

                    if [pf_subdim 
                            for pf_subdim in pf.dims[i+1:]
                            if isinstance(pf_subdim, THREAD_IDX_TAG)]:
                        dim_expr = dim_expr % pf_dim.length

                    pf_indices.append(str(dim_expr))
                    pf_idx_subst_map[pf_dim.name] = dim_expr
                    prev_dim_sizes *= pf_dim.length

            assert prev_dim_sizes == non_thread_pf_size

            # generate smem assignment statement
            from pymbolic.mapper.substitutor import substitute
            pf_assignment = Assign(
                    pf_name + "".join("[%s]" % dexpr 
                        for dexpr in pf_indices),
                    "tex1Dfetch(tex_%s, %s)" 
                    % (pf.input_vector,
                        substitute(pf.index_expr, pf_idx_subst_map))
                    )

            # generate fetch loop

            # fetch_base and fetch_idx don't include thread_pf_size
            fetch_base = 0
            while fetch_base + non_pf_thread_size <= non_thread_pf_size:
                fetch_block.append(pf_assignment)
                if fetch_base + non_pf_thread_size < non_thread_pf_size:
                    fetch_block.append(
                            S("fetch_idx += %d" % non_pf_thread_size))
                fetch_base += non_pf_thread_size
            if fetch_base < non_thread_pf_size:
                from pytools import product
                fetch_block.append(
                        If("fetch_idx < %d" % non_thread_pf_size,
                            pf_assignment))

            new_block = Block([
                    Comment("prefetch dimensions: " + ", ".join(
                        "%s[%d]" % (pfdim.name, pfdim.length) for pfdim in pf.dims)),
                    Comment("  ... direct-thread prefetch dims: " + ", ".join(
                        pfdim.name for pfdim in thread_pf_dims)),
                    Line(),
                    ])

            # omit head sync primitive if we just came out of a prefetch
            if not isinstance(next_outer_sched_item, PrefetchDescriptor):
                new_block.append(S("__syncthreads()"))
            else:
                new_block.append(Comment("next outer schedule item is a prefetch: "
                    "no sync needed"))

            new_block.extend([
                Line(),
                smem_pf_array,
                fetch_block,
                Line(),
                ])

            # omit tail sync primitive if we're headed into another prefetch
            if not isinstance(next_inner_sched_item, PrefetchDescriptor):
                new_block.append(S("__syncthreads()"))
            else:
                new_block.append(Comment("next inner schedule item is a prefetch: "
                    "no sync needed"))

            new_block.extend([Line(),inner])

            inner = new_block

        elif isinstance(sched_item, RegisterPrefetch):
            inner = Block([
                Initializer(POD(numpy.float32,
                    sched_item.new_name),
                    "tex1Dfetch(tex_%s, %s)"
                    % (sched_item.index_expr.aggregate.name,
                        ccm(sched_item.index_expr.index, PREC_NONE))),
                inner])

        else:
            raise ValueError("invalid schedule item encountered")

    mod = Module()

    # add texture declarations
    for v in kernel.input_vectors():
        mod.append(
                Value("texture<float, 1, cudaReadModeElementType>",
                    "tex_"+v))

    # symbolic names for thread and block indices
    mod.extend([Line()]
            + [Define(dim.name, "blockIdx.%s /* 0..%d */"
                % (AXES[dim.tag.axis], dim.length-1))
                for dim in kernel.dims_by_tag_type(BLOCK_IDX_TAG)]
            + [Define(dim.name, "threadIdx.%s /* 0..%d */"
                % (AXES[dim.tag.axis], dim.length-1))
                for dim in kernel.dims_by_tag_type(THREAD_IDX_TAG)]
            + [Line()])

    # construct function
    mod.append(
        FunctionBody(
            CudaGlobal(FunctionDeclaration(
                Value("void", "loopy_kernel"),
                [Pointer(POD(numpy.float32, name)) 
                    for name in kernel.output_vectors()])),
            Block([inner])))

    return str(mod)




# debugging -------------------------------------------------------------------
def print_kernel_info(knl):
    print "PREFETCH", total_prefetch_size(knl)
    for pf in knl.prefetch.itervalues():
        print "   %s[%s]: %s" % (pf.input_vector, pf.index_expr, pf.dims)
    print
    print "Scheduling: ---------------------"
    for sched_item in knl.schedule:
        print sched_item
    print

    for ld in knl.dims:
        print ld
    print
    for t, e in knl.instructions:
        print "%s <- %s" % (t, e)




# speed measurement -----------------------------------------------------------
def make_gpu_function_and_texrefs(kernel, code):
    mod = compiler.SourceModule(code)
    texref_lookup = dict(
            (iv, mod.get_texref("tex_"+iv))
            for iv in kernel.input_vectors())
    func = mod.get_function("loopy_kernel")
    func.prepare("P" * len(kernel.output_vectors()),
            kernel.block_dim(), texrefs=texref_lookup.values())

    return func, texref_lookup




class CompiledKernel:
    def __init__(self, kernel):
        self.kernel = kernel
        self.code = generate_code(kernel)
        self.gpu_function, self.texref_lookup = \
                make_gpu_function_and_texrefs(kernel, self.code)

    def time_run(self, launcher, warmup_rounds=2, timing_rounds=5):
        for i in range(warmup_rounds):
            launcher(self.kernel.grid_dim(), 
                    self.gpu_function, 
                    self.texref_lookup)
        evt_start = cuda.Event()
        evt_end = cuda.Event()
        evt_start.record()
        for i in range(timing_rounds):
            launcher(self.kernel.grid_dim(), 
                    self.gpu_function, 
                    self.texref_lookup)
        evt_end.record()
        evt_end.synchronize()

        return evt_end.time_since(evt_start)*1e-3/timing_rounds




# driver ----------------------------------------------------------------------
def generate_all_kernels(orig_kernel, min_threads=None, min_blocks=None,
        prefetch_hints={}):
    for knl in generate_dim_assignments(orig_kernel):
        if min_threads is not None and knl.thread_count() < min_threads:
            continue
        if min_blocks is not None and knl.block_count() < min_blocks:
            continue

        if False:
            for d in knl.dims:
                print d
            print "-------------------------------------------------------"
            raw_input()
        else:
            for pf_knl in generate_all_prefetching_kernels(knl,
                    prefetch_hints):
                for sch_knl in generate_loop_schedules(pf_knl):
                    yield insert_register_prefetches(sch_knl)




def drive_timing_run(kernel_generator, launch, flop_count=None):
    soln_count = 0
    for kernel in kernel_generator:

        compiled = CompiledKernel(kernel)
        elapsed = compiled.time_run(launch)

        print "-----------------------------------------------"
        print "SOLUTION #%d" % soln_count
        print "-----------------------------------------------"
        print compiled.code
        print "-----------------------------------------------"
        print "time: %f" % elapsed
        if flop_count is not None:
            print "gflops/s: %f (#%d)" % (
                    flop_count/elapsed/1e9, soln_count)
        print "-----------------------------------------------"

        soln_count += 1

    print "%d solutions" % soln_count




def show_kernel_codes(kernel_generator):
    soln_count = 0

    for kernel in kernel_generator:
        print "-----------------------------------------------"
        print "SOLUTION #%d" % soln_count
        print "-----------------------------------------------"
        print generate_code(kernel)
        soln_count += 1

    print "%d solutions" % soln_count





def main_matrix_mul():
    n = 16*34
    from pymbolic import var
    a, b, c, i, j, k = [var(s) for s in "abcijk"]

    k = make_loop_kernel([
        LoopDimension("i", n),
        LoopDimension("j", n),
        LoopDimension("k", n),
        ], [ 
        (c[i+n*j], a[i+n*k]*b[k+n*j]) 
        ])

    gen_kwargs = {
            "min_threads": 128,
            "min_blocks": 32,
            }

    if True and HAVE_CUDA:
        if HAVE_CUDA:
            a = curandom.rand((n, n))
            b = curandom.rand((n, n))
            c = gpuarray.empty_like(a)

        def launcher(grid, kernel, texref_lookup):
            a.bind_to_texref_ext(texref_lookup["a"])
            b.bind_to_texref_ext(texref_lookup["b"])
            kernel.prepared_call(grid, c.gpudata)

        drive_timing_run(
                generate_all_kernels(k, **gen_kwargs), 
                launcher, 2*n**3)
    else:
        show_kernel_codes(generate_all_kernels(k, **gen_kwargs))




def main_elwise_scaled_matrix_mul():
    Np = 64
    K = 2000
    from pymbolic import var
    m, u, v, g, i, j, k = [var(s) for s in "muvgijk"]

    knl = make_loop_kernel([
        LoopDimension("i", Np),
        LoopDimension("j", Np),
        LoopDimension("k", K),
        ], [ 
        (v[i+Np*k], m[i+Np*j]*u[j+Np*k]*g[k]) 
        ])

    gen_kwargs = {
            "min_threads": 128,
            "min_blocks": 32,
            "prefetch_hints": {"g": False, "m":True},
            }

    if True and HAVE_CUDA:
        if HAVE_CUDA:
            g = curandom.rand((K))
            u = curandom.rand((Np, K))
            m = curandom.rand((Np, Np))
            v = gpuarray.empty_like(u)

        def launcher(grid, kernel, texref_lookup):
            g.bind_to_texref_ext(texref_lookup["g"])
            u.bind_to_texref_ext(texref_lookup["u"])
            m.bind_to_texref_ext(texref_lookup["m"])
            kernel.prepared_call(grid, v.gpudata)

        drive_timing_run(
                generate_all_kernels(knl, **gen_kwargs), 
                launcher, 2*Np**2*K)
    else:
        show_kernel_codes(generate_all_kernels(knl, **gen_kwargs))




def main_transpose():
    n = 16*48
    from pymbolic import var
    a, b, i, j = [var(s) for s in "abij"]

    k = make_loop_kernel([
        LoopDimension("i", n),
        LoopDimension("j", n),
        ], [ 
        (b[i+n*j], a[j+n*i]) 
        ])

    gen_kwargs = {
            "min_threads": 128,
            "min_blocks": 32,
            }

    if True and HAVE_CUDA:
        if HAVE_CUDA:
            a = curandom.rand((n, n))
            b = gpuarray.empty_like(a)

        def launcher(grid, kernel, texref_lookup):
            a.bind_to_texref_ext(texref_lookup["a"])
            kernel.prepared_call(grid, b.gpudata)

        drive_timing_run(
                generate_all_kernels(k, **gen_kwargs), 
                launcher, 0)
    else:
        show_kernel_codes(generate_all_kernels(k, **gen_kwargs))




def main_volume_d_dx():
    Np =  128
    K = 22000
    from pymbolic import var

    D, g, u, p, i, j, k = [var(s) for s in "Dgupijk"]

    dim = 3

    ker = make_loop_kernel([
        LoopDimension("k", K),
        LoopDimension("j", Np),
        LoopDimension("i", Np),
        ], [ 
            (p[j+Np*k], 
                sum(g[dim*(j+Np*k)+i] * D[dim*(j+Np*i)+i] for i in range(3)) 
                * u[i+Np*k])
            ])

    gen_kwargs = {
            "min_threads": 128,
            "min_blocks": 32,
            }

    if True and HAVE_CUDA:
        if HAVE_CUDA:
            u = curandom.rand((Np, K))
            p = curandom.rand((Np, K))
            g = curandom.rand((Np*3, K))
            D = curandom.rand((Np*3, Np))

        def launcher(grid, kernel, texref_lookup):
            u.bind_to_texref_ext(texref_lookup["u"])
            g.bind_to_texref_ext(texref_lookup["g"])
            D.bind_to_texref_ext(texref_lookup["D"])
            kernel.prepared_call(grid, p.gpudata)

        drive_timing_run(
                generate_all_kernels(ker, **gen_kwargs),
                launcher, Np*Np*K)
    else:
        show_kernel_codes(generate_all_kernels(ker, **gen_kwargs))




if __name__ == "__main__":
    main_elwise_scaled_matrix_mul()
