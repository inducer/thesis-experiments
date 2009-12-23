set isosamples 50
l1(r,s)=(r+1)/2
l2(r,s)=(s+1)/2
l3(r,s)=-(r+s)/2

d = 2

# "dirichlet hump", zero at boundary, one at barycenter
dhump(r,s) = (d+1)**(d+1)*l1(r,s)*l2(r,s)*l3(r,s)

# eN: one on edge N, zero on all other edges
e3(r,s) = -4*l1(r,s)*l2(r,s)/((l1(r,s)-l2(r,s))**2-1)
e2(r,s) = -4*l1(r,s)*l3(r,s)/((l1(r,s)-l3(r,s))**2-1)
e1(r,s) = -4*l2(r,s)*l3(r,s)/((l2(r,s)-l3(r,s))**2-1)

# ehfuncN: hump along edge N, zero on all other edges
eh3(r,s) = (d**d)*l1(r,s)*l2(r,s)
eh2(r,s) = (d**d)*l1(r,s)*l3(r,s)
eh1(r,s) = (d**d)*l2(r,s)*l3(r,s)

splot [r=-1:1][s=-1:1] r+s>0?0/0: dhump(r,s)
