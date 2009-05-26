from pytools import Record
import numpy

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
    pass

class IndexSubsitution(Record):
    __slots__ = ["old_variable", "new_expr"]
    pass




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




def block_size(assignments):
    bs = [1]*3
    for tia in [ass for ass in assignments
            if isinstance(ass, ThreadIndexAssignment)]:
        bs[tia.axis] = len(tia)

    return bs




def grid_size(assignments):
    gs = [1]*2
    for bia in [ass for ass in assignments
            if isinstance(ass, BlockIndexAssignment)]:
        gs[bia.axis] = len(bia)

    return gs




def full_subst(subst_map, expr):
    from pymbolic.mapper.substitutor import substitute
    while True:
        last_expr = expr
        expr = substitute(expr, subst_map)
        if last_expr == expr:
            return expr




class LoopyKernel:
    def __init__(self, domain, insns):
        from pymbolic import parse
        self.insns = [(parse(lvalue), parse(expr)) 
                for lvalue, expr in insns]

        from pymbolic.mapper.dependency import DependencyMapper
        dm = DependencyMapper(include_subscripts=False)

        from pymbolic import var
        all_indices = set(idx[0] for idx in domain)

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

        soln_count = 0
        for ass, transformed_output_indices in \
                generate_domain_assignments(domain, 
                        output_indices=self.output_indices):
            if product(block_size(ass)) >= 128:
                print "-----------------------------------------------"
                print "grid", grid_size(ass)
                print "block", block_size(ass)
                print
                print self.generate_code(ass, transformed_output_indices)

                soln_count += 1

        print "%d solutions" % soln_count

    def generate_code(self, assignments, output_indices):
        from codepy.cgen import FunctionBody, FunctionDeclaration, \
                Typedef, POD, Value, Pointer, Module, Block, \
                Initializer, Assign, Statement, For

        for_loops = [ass for ass in assignments
                if isinstance(ass, ForLoopAssignment)]
        output_loops = [fl for fl in for_loops 
                if fl.name in output_indices]
        reduction_loops = [fl for fl in for_loops 
                if fl.name not in output_indices]

        # construct fully resolved subst_map
        subst_map = dict((s.old_variable, s.new_expr) 
                for s in assignments
                if isinstance(s, IndexSubsitution))

        from pymbolic.primitives import Subscript
        from pymbolic.mapper.c_code import CCodeMapper
        from pymbolic.mapper.stringifier import PREC_NONE
        ccm = CCodeMapper()

        inner = Block([])
        for lvalue, expr in self.insns:
            assert isinstance(lvalue, Subscript)
            name = lvalue.aggregate.name
            inner.append(Statement("tmp_%s += %s"
                % (name, ccm(full_subst(subst_map, expr), PREC_NONE))))

        for loop in reduction_loops:
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
                    ccm(full_subst(subst_map, lvalue), PREC_NONE),
                    "tmp_"+lvalue.aggregate.name)
                    for lvalue, expr in self.insns])

        for loop in output_loops:
            inner = For(
                    "int %s = %s" % (loop.name, loop.start),
                    "%s < %s" % (loop.name, loop.stop),
                    "++%s" % loop.name, inner)

        from codepy.cgen.cuda import CudaGlobal

        mod = Module()

        for v in self.input_vectors:
            mod.append(
                    Value("texture<float32, 1, cudaReadModeElement>",
                        "tex_"+v));

        mod.append(
            FunctionBody(
                CudaGlobal(FunctionDeclaration(
                    Value("void", "loopy_kernel"),
                    [Pointer(POD(numpy.float32, name)) for name in self.output_vectors])),
                Block([inner])))

        return str(mod)

    def __call__(self, **vars):
        pass

def main():
    from pymbolic import parse
    k = LoopyKernel([
        ("i", 16*34),
        ("j", 16*34),
        ("k", 16*34),
        ],
        [ ("c[i+16*34*j]", "a[i+16*34*k]*b[k+16*34*j]") ]
        )





if __name__ == "__main__":
    main()
