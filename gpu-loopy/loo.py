from pytools import Record
import numpy
from pymbolic.mapper.c_code import CCodeMapper
from pymbolic.mapper.evaluator import EvaluationMapper
from pymbolic.mapper.stringifier import PREC_NONE

have_cuda = False

if have_cuda:
    import pycuda.autoinit
    import pycuda.driver as cuda
    import pycuda.compiler as compiler
    import pycuda.gpuarray as gpuarray
    import pycuda.curandom as curandom

from pytools import memoize_method

class IndexDescriptor(Record):
    __slots__ = ["name", "start", "stop", "is_output"]

class IndexAssignment(Record):
    __slots__ = ["name", "start", "stop"]

    def __len__(self):
        return self.stop-self.start

class ForLoopAssignment(IndexAssignment):
    pass

class BlockIndexAssignment(IndexAssignment):
    __slots__ = ["axis"]
    pass

class ThreadIndexAssignment(IndexAssignment):
    __slots__ = ["axis"]

class IndexSubsitution(Record): 
    __slots__ = ["old_variable", "new_expr"]




AXES = ["x", "y", "z", "w"]




def generate_thread_index_assignment_numberings(assignments):
    ti_ass = [ass for ass in assignments
            if isinstance(ass, ThreadIndexAssignment)]
    other_ass = [ass for ass in assignments
            if not isinstance(ass, ThreadIndexAssignment)]

    if not ti_ass:
        yield assignments
    else:
        from pytools import generate_unique_permutations

        from pymbolic import var
        for perm in generate_unique_permutations(
                tuple(range(len(ti_ass)))):
            from pytools import flatten
            yield other_ass + list(flatten([
                tia.copy(axis=p_i),
                IndexSubsitution(
                    old_variable=tia.name,
                    new_expr=var("threadIdx."+AXES[p_i]))
                ]
                for tia, p_i in zip(ti_ass, perm)))




def generate_domain_assignments(domain, done_assignments=[],
        output_indices=set(), no_thread_indices=set()):
    if not domain:
        for ass in generate_thread_index_assignment_numberings(
                done_assignments):
            yield ass, output_indices
        return

    name, length = domain[0]

    assert length >= 2

    props = dict(name=name, start=0, stop=length)

    for ass in generate_domain_assignments(
            domain[1:], 
            done_assignments=(
                done_assignments+[ForLoopAssignment(**props)]),
            output_indices=output_indices,
            no_thread_indices=no_thread_indices):
        yield ass

    from pymbolic import var

    block_idx_ass_count = sum(1 for ass in done_assignments
            if isinstance(ass, BlockIndexAssignment))
    if name in output_indices and block_idx_ass_count < 2 :
        for ass in generate_domain_assignments(
                domain[1:], 
                done_assignments=(
                    done_assignments+[
                        IndexSubsitution(
                            old_variable=name,
                            new_expr=var("blockIdx."+AXES[
                                block_idx_ass_count])),
                        BlockIndexAssignment(
                            axis=block_idx_ass_count, **props)]),
                output_indices=output_indices,
                no_thread_indices=no_thread_indices):
            yield ass

    # try to assign to thread indices
    assigned_thread_indices  = sum(1 
            for ass in done_assignments
            if isinstance(ass, ThreadIndexAssignment))
    from pytools import product
    assigned_block_size  = product(len(ass)
            for ass in done_assignments
            if isinstance(ass, ThreadIndexAssignment))
    leftover_block_size = 512 // assigned_block_size

    if (name in output_indices 
            and name not in no_thread_indices
            and assigned_thread_indices < 3 
            and leftover_block_size > 1):
        my_block_length = 1
        while my_block_length < length:
            my_block_length *= 2
            if my_block_length > length:
                my_block_length = length

            if my_block_length > leftover_block_size:
                break

            if length % my_block_length == 0:
                new_name = name+"_b"
                leftover_name = name+"_l"
                new_length = length//my_block_length
                
                ass_add = [ThreadIndexAssignment(
                    name=new_name, start=0, stop=my_block_length)]

                if new_length > 1:
                    domain_add = [ (leftover_name, new_length) ]
                    ass_add.append(IndexSubsitution(
                        old_variable=name,
                        new_expr=(var(new_name)
                            +my_block_length*var(leftover_name))))
                    addl_output_indices = set([leftover_name])
                    addl_no_thread_indices = set([leftover_name])
                else:
                    domain_add = []
                    addl_output_indices = set()
                    addl_no_thread_indices = set()

                for ass in generate_domain_assignments(
                        domain_add+domain[1:], 
                        done_assignments=(
                            done_assignments+ass_add),
                        output_indices=(
                            output_indices | addl_output_indices),
                        no_thread_indices=(
                            no_thread_indices | addl_no_thread_indices)
                        ):
                    yield ass




def full_subst(subst_map, expr):
    from pymbolic.mapper.substitutor import substitute
    while True:
        last_expr = expr
        expr = substitute(expr, subst_map)
        if last_expr == expr:
            return expr




class TexturingCCodeMapper(CCodeMapper):
    def __init__(self, input_vectors):
        CCodeMapper.__init__(self)
        self.input_vectors = input_vectors

    def map_subscript(self, expr, enclosing_prec):
        from pymbolic.primitives import Variable
        if (isinstance(expr.aggregate, Variable)
                and expr.aggregate.name in self.input_vectors):
            return "tex1Dfetch(tex_%s, %s)" % (
                    expr.aggregate.name,
                    self.rec(expr.index, PREC_NONE))
        else:
            return CCodeMapper.map_subscript(self, expr, enclosing_prec)




class KernelVariant:
    def __init__(self, 
            idx_assignments, transformed_output_indices, kernel):
        self.subst_map = dict((s.old_variable, s.new_expr) 
                for s in idx_assignments
                if isinstance(s, IndexSubsitution))

        self.idx_assignments = idx_assignments
        self.output_indices = transformed_output_indices

        self.insns = [
                (full_subst(self.subst_map, lvalue),
                    full_subst(self.subst_map, expr))
                for lvalue, expr in kernel.insns]

        self.for_loops = [ass for ass in self.idx_assignments
                if isinstance(ass, ForLoopAssignment)]
        self.output_loops = [fl for fl in self.for_loops 
                if fl.name in self.output_indices]
        self.reduction_loops = [fl for fl in self.for_loops 
                if fl.name not in self.output_indices]

        self.kernel = kernel

        find_reuse(self, self.insns[0][1])
        raw_input()

    @memoize_method
    def thread_axes(self):
        thread_axes = [None]*3
        for tia in [ass for ass in self.idx_assignments
                if isinstance(ass, ThreadIndexAssignment)]:
            thread_axes[tia.axis] = tia

        return thread_axes

    @memoize_method
    def block_axes(self):
        block_axes = [None]*2
        for bia in [ass for ass in self.idx_assignments
                if isinstance(ass, BlockIndexAssignment)]:
            block_axes[bia.axis] = bia
        return block_axes

    def block_size(self):
        return tuple(
                1 if bia is None else len(bia)
                for bia in self.thread_axes())

    def grid_size(self):
        return tuple(
                1 if gia is None else len(gia)
                for gia in self.block_axes())

    @memoize_method
    def code(self):
        from codepy.cgen import FunctionBody, FunctionDeclaration, \
                Typedef, POD, Value, Pointer, Module, Block, \
                Initializer, Assign, Statement, For

        from pymbolic.primitives import Subscript
        ccm = TexturingCCodeMapper(self.kernel.input_vectors)

        inner = Block([])
        for lvalue, expr in self.insns:
            assert isinstance(lvalue, Subscript)
            name = lvalue.aggregate.name
            inner.append(Statement("tmp_%s += %s"
                % (name, ccm(expr, PREC_NONE))))

        for loop in self.reduction_loops:
            inner = For(
                    "int %s = %s" % (loop.name, loop.start),
                    "%s < %s" % (loop.name, loop.stop),
                    "++%s" % loop.name, inner)

        inner = Block(
                [Initializer(POD(numpy.float32, 
                    "tmp_"+lvalue.aggregate.name), 0)
                    for lvalue, expr in self.insns]
                +[inner]+
                [Assign(
                    ccm(lvalue, PREC_NONE),
                    "tmp_"+lvalue.aggregate.name)
                    for lvalue, expr in self.insns])

        for loop in self.output_loops:
            inner = For(
                    "int %s = %s" % (loop.name, loop.start),
                    "%s < %s" % (loop.name, loop.stop),
                    "++%s" % loop.name, inner)

        from codepy.cgen.cuda import CudaGlobal

        mod = Module()

        for v in self.kernel.input_vectors:
            mod.append(
                    Value("texture<float, 1, cudaReadModeElementType>",
                        "tex_"+v));

        mod.append(
            FunctionBody(
                CudaGlobal(FunctionDeclaration(
                    Value("void", "loopy_kernel"),
                    [Pointer(POD(numpy.float32, name)) 
                        for name in self.kernel.output_vectors])),
                Block([inner])))

        return str(mod)

    def func_and_texrefs(self):
        mod = compiler.SourceModule(self.code())
        texref_lookup = dict(
                (iv, mod.get_texref("tex_"+iv))
                for iv in self.kernel.input_vectors)
        func = mod.get_function("loopy_kernel")
        func.prepare("P" * len(self.kernel.output_vectors),
                self.block_size(), texrefs=texref_lookup.values())

        return func, texref_lookup




class ReuseDetectingEvaluationMapper(EvaluationMapper):
    def __init__(self, context):
        EvaluationMapper.__init__(self, context)
        self.reuse_map = {}
        # variable -> index -> [count, exprs]

        self.max_reuse_map = {}

    def map_subscript(self, expr):
        from pymbolic.primitives import Variable
        if (isinstance(expr.aggregate, Variable)):
            var_name = expr.aggregate.name
            var_reuse_dict = self.reuse_map.setdefault(
                    var_name, {})
            idx_reuse_data = var_reuse_dict.setdefault(
                    self.rec(expr.index), [0, []])
            idx_reuse_data[0] += 1
            idx_reuse_data[1].append(
                (expr.index, self.context.copy()))

            self.max_reuse_map[var_name] = max(
                    self.max_reuse_map.get(var_name, 0),
                    idx_reuse_data[0])

            return 0
        else:
            return EvaluationMapper.map_subscript(self, expr)
        



def find_reuse(kv, expr):
    print expr

    t_axes_names = ["threadIdx."+AXES[i] 
            for i, ta in enumerate(kv.thread_axes()) if ta is not None]

    context = dict(("blockIdx."+AXES[i], 0) 
            for i, bia in enumerate(kv.block_axes()) if bia is not None)

    context.update(dict(
        (ol.name, 0) for ol in kv.output_loops))

    redloop_names, redloop_bounds = zip(*[
        (fl.name, min(16, len(fl))) for fl in kv.reduction_loops
        ])

    from pytools import generate_nonnegative_integer_tuples_below as gnitb
    mapper = ReuseDetectingEvaluationMapper(context)
    for rli in gnitb(redloop_bounds):
        mapper.context.update(dict(zip(redloop_names, rli)))
        for ti in gnitb(kv.block_size()):
            mapper.context.update(dict(zip(t_axes_names, ti)))
            mapper(expr)

    for var, reuse in mapper.reuse_map.iteritems():
        max_reuse = mapper.max_reuse_map[var]
        if max_reuse == 1:
            continue
        print "VARIABLE %s max re-use: %d" % (var, max_reuse)
        for i, (reuse_count, reuse_info) in reuse.iteritems():
            print "  ", i, reuse_count
            for idx_expr, ctx in reuse_info:
                print "    %s | %s" % (idx_expr, ctx)




class LoopyKernel:
    def __init__(self, domain, insns, bogus_launcher, flop_count):
        from pymbolic import parse
        self.insns = [(parse(lvalue), parse(expr)) 
                for lvalue, expr in insns]

        from pymbolic.mapper.dependency import DependencyMapper
        dm = DependencyMapper(include_subscripts=False)

        from pymbolic import var
        indices = set(idx[0] for idx in domain)

        self.output_indices = set()
        for lvalue, expr in self.insns:
            self.output_indices.update(
                    set(v.name for v in dm(lvalue)) 
                    & all_indices)

        self.input_vectors = set()
        self.output_vectors = set()
        for lvalue, expr in self.insns:
            self.input_vectors.update(
                    set(v.name for v in dm(expr)) 
                    - all_indices)
            self.output_vectors.update(
                    set(v.name for v in dm(lvalue)) 
                    - all_indices)

        # these actually need to be ordered
        self.output_vectors = list(self.output_vectors)

        from pytools import product

        timings = []
        for ass, transformed_output_indices in \
                generate_domain_assignments(domain, 
                        output_indices=self.output_indices):
            kv = KernelVariant(
                    ass, transformed_output_indices, self)
            if product(kv.block_size()) >= 128:
                func, texref_lookup = kv.func_and_texrefs()

                for i in range(1):
                    bogus_launcher(kv.grid_size(), func, texref_lookup)
                evt_start = cuda.Event()
                evt_end = cuda.Event()
                evt_start.record()
                for i in range(2):
                    bogus_launcher(kv.grid_size(), func, texref_lookup)
                evt_end.record()
                evt_end.synchronize()

                elapsed = evt_end.time_since(evt_start)*1e-3
                
                print "-----------------------------------------------"
                print "grid", kv.grid_size()
                print "block", kv.block_size
                print
                print kv.code()
                print "-----------------------------------------------"
                print "time: %f" % elapsed
                print "gflops/s: %f" % (flop_count/elapsed/1e9)
                print "-----------------------------------------------"
                timings.append(("%s %s" % (kv.grid_size(), kv.block_size()), elapsed))

        if False:
            from matplotlib.pyplot import plot, xticks, savefig
            timings.sort(key=lambda e: e[1])
            labels, times = zip(*timings)
            times = numpy.array(times)
            flops = flop_count/times

            x_points = arange(len(times))
            plot(x_points, times, "o")
            xticks(x_points, labels)
            savefig("times.png")

            clf()
            plot(x_points, flops, "o")
            xticks(x_points, labels)
            savefig("times.png")

            print "%d solutions" % soln_count


    def __call__(self, **vars):
        pass





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

    k = LoopyKernel([
        ("i", n),
        ("j", n),
        ("k", n),
        ],
        [ ("c[i+16*34*j]", "a[i+16*34*k]*b[k+16*34*j]") ],
        bogus_launcher,
        flop_count=2*n**3)




if __name__ == "__main__":
    main()
