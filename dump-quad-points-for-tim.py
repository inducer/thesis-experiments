#! /usr/bin/env python

from hedge.quadrature import SimplexCubature

outf = open("gm_quadpoints.m", "w")

for s in range(1, 10):
    sc = SimplexCubature(s, 3)

    outf.write("o%dquad = [\n" % (2*s+1))

    for p, w in zip(sc.points, sc.weights):
        outf.write(" ".join(repr(x) for x in list(p)+[w])+"\n")
    outf.write("]\n\n")
