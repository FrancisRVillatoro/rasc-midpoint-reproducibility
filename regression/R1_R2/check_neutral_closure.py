#!/usr/bin/env python3
"""Exact linear-algebra regression separating invariant suppression from symmetry closure."""
import sympy as s
L=s.Matrix([[0,1,0],[0,0,0],[0,0,0]])
Z=s.Matrix([1,0,0])
invariant=s.Matrix([[0,1,0]])
g=s.Matrix([1,0,1]); bad=s.Matrix([1,1,0])
assert invariant*L==s.zeros(1,3)
assert (invariant*g)[0]==0 and L*g==s.zeros(3,1)
assert Z.cross(g)!=s.zeros(3,1) # no unique symmetry rate for g
assert (invariant*bad)[0]!=0 and L*bad!=s.zeros(3,1)
print('L =',L)
print('invariant*g =',invariant*g,'L*g =',L*g)
print('g is in ker L but outside Ran Z: no t^2 forcing, yet not only symmetry drift')
print('R2 NEUTRAL-CLOSURE PASS')
