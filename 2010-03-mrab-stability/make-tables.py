from math import pi

for name, k, mucrit in [
        ("k2-mubig", 2, ">=0.5"),
        ("k2-musmall", 2, "<=0.5"),
        ("k3-musmall", 3, "<=0.333"),
        ("k4-musmall", 4, "<=0.2"),
        ]:
    tbl = table_from_cursor(q("select method,round(min(dt),3),"
                " round(avg(dt),3),round(max(dt),3) from data "
                " where angle=? and offset >= ? "
                " and offset <= ?+1e-5 and ratio %s and substep_count=?"
                " group by method "
                " order by min(dt) desc" % mucrit, 
                (0.05*pi, 0.3*pi, 0.7*pi, k)))
    del tbl.rows[0]

    open("stability-tables/stabtab-%s.tex" % name, "w").write(tbl.latex())
