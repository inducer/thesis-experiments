from __future__ import division
from pytools import one
orders = sorted(one(x) for x in q("select distinct dg_order from runs"))
el_counts = sorted(one(x) for x in q("select distinct element_count from runs"))

errs = dict(
        ((dg_order, element_count), err)
        for dg_order, element_count, err in q(
            "select dg_order, element_count, max($l1_err_rho)"
            " where abs(viscosity_scale-0.4)< 1e-5 and smoother like '%Vertexwise%'"
            " group by dg_order, element_count"))

from pytools import Table
tbl = Table()
tbl.add_row([''] + ["$N=%d$" % n for n in orders]
        + ['EOC'])


def tex_number(f):
    import re
    return re.sub(r"e+?(-?)0*([0-9]+)", r"\cdot 10^{\1\2}", "$%.3e$" % f)

from hedge.tools.convergence import estimate_order_of_convergence

for k in el_counts:
    row_errs = []

    for n in orders:
        err = errs.get((n,k), None)
        if err is not None:
            row_errs.append(err)
        else:
            break

    _, row_o = estimate_order_of_convergence(
            orders[:len(row_errs)], row_errs)


    tbl.add_row(['$h/%.0f$' % ((k-1)/(el_counts[0]-1))] 
            + [tex_number(err) for err in row_errs] 
            + ["n/a"] * (len(orders)-len(row_errs))
            + ['%.2f' % row_o])

eoc_row = [""]
for n in orders:
    col_errs = []

    for k in el_counts:
        err = errs.get((n,k), None)
        if err is not None:
            col_errs.append(err)
        else:
            break
    _, col_o = estimate_order_of_convergence(
            el_counts[:len(col_errs)], col_errs)
    eoc_row.append('%.2f' % col_o)

tbl.add_row(eoc_row+[''])


print tbl.latex()
