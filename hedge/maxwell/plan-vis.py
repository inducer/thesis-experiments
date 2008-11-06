from pylab import *
rc('font', size='20')
title("Work Distribution Study")
xlabel("$w_p$")
ylabel("Execution time [$ms$]")
annotate("Local differentiation, matrix-in-shared,\n"
        "order 4, with microblocking\n"
        "point size denotes $w_i\in\{1,\ldots,4\}$\n" ,
        (0.03, 0.97), 
        fontsize='16',
        xycoords='axes fraction',
        ha='left', va='top',
        )
db.scatter_cursor(db.q("select parallel, value/1e-3, 30*inline, serial from data where type='smem_matrix' and mb_elements=4"))
cb = colorbar()
cb.set_label("$w_s$")
xlim((14,33))
savefig("plan-vis.pdf")
