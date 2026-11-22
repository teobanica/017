# mascagni.py: projective quantum isometries of the complex Grassmannians (section 4).
# (1) checks at N=4 that r_4 cap Sym_tau = V + W, with dims 84 and 20, inside (C^N)^{tensor 4};
# (2) computes exactly, over Q(N), the projections P_V, P_W in the basis of the 24 walled
#     Brauer diagrams on 8 points;
# (3) for the generators TL + P_V and TL + (P_V + P_W), finds numerically at N=5 a derivation
#     (rotations, adjoints, products) of the half-liberated crossing sigma (x) id, then replays it
#     exactly over Q(N), asserts that the residual vanishes identically, factors all the
#     denominators and pivots exactly, and lists the integers N where the certificate could fail.
# Writes mascagni.txt.
# ---- diag.py
# Brauer diagrams on 8 points: top j0..j3 (points 0..3), bottom i0..i3 (points 4..7)
def pairings(pts):
    if not pts: yield []; return
    a=pts[0]
    for k in range(1,len(pts)):
        b=pts[k]; rest=pts[1:k]+pts[k+1:]
        for p in pairings(rest): yield [(a,b)]+p
ALL=[tuple(sorted(tuple(sorted(x)) for x in p)) for p in pairings(list(range(8)))]
def color(pt):  # V positions: 0,2 (top) and 4,6 (bottom); top V counts +1, bottom V counts -1 ...
    pos=pt%4
    return 1 if pos in (0,2) else -1
def walled(p):
    # U_N-intertwiner: through string (top-bottom) must connect same color; cap/cup must connect opposite colors
    for a,b in p:
        ta,tb=a<4,b<4
        if ta!=tb:
            if color(a)!=color(b): return False
        else:
            if color(a)==color(b): return False
    return True
WB=[p for p in ALL if walled(p)]
def compose(p,q):
    # p on top (points 0..3 top, 4..7 bottom), q below; p's bottom glued to q's top
    # nodes: ('p',x),('q',x)
    adj={}
    def link(u,v): adj.setdefault(u,[]).append(v); adj.setdefault(v,[]).append(u)
    for a,b in p: link(('p',a),('p',b))
    for a,b in q: link(('q',a),('q',b))
    for k in range(4): link(('p',4+k),('q',k))  # glue
    ext=[('p',k) for k in range(4)]+[('q',4+k) for k in range(4)]
    def newlabel(v): return v[1] if v[0]=='p' else v[1]
    seen=set(); res=[]
    for s in ext:
        if s in seen: continue
        prev=None; cur=s; seen.add(cur)
        while True:
            nxt=[v for v in adj[cur] if v!=prev] if prev is not None else adj[cur]
            # move along: each internal node has degree 2 (one diagram edge, one glue)
            if prev is None: nb=adj[cur][0]
            else:
                nbs=[v for v in adj[cur] if v!=prev]
                nb=nbs[0]
            prev,cur=cur,nb; seen.add(cur)
            if cur in ext and cur!=s:
                res.append(tuple(sorted((newlabel(s),newlabel(cur))))); break
    # count loops among internal nodes not seen
    internal=[('p',4+k) for k in range(4)]+[('q',k) for k in range(4)]
    loops=0
    for s in internal:
        if s in seen: continue
        loops+=1; prev=None; cur=s
        while True:
            seen.add(cur)
            nbs=adj[cur] if prev is None else [v for v in adj[cur] if v!=prev]
            if len(nbs)==0: break
            nb=nbs[0]
            if nb in seen: break
            prev,cur=cur,nb
    return tuple(sorted(res)),loops
CYC=[0,1,2,3,7,6,5,4]   # cyclic order: j0..j3, i3..i0
def rotate(p):
    # xi'_{c_k} pairs (k,l) iff xi pairs (k+1,l+1)
    pos={pt:k for k,pt in enumerate(CYC)}
    res=[]
    for a,b in p:
        ka,kb=pos[a]-1,pos[b]-1
        res.append(tuple(sorted((CYC[ka%8],CYC[kb%8]))))
    return tuple(sorted(res))
def adjoint(p):
    sw=lambda x:(x+4)%8
    return tuple(sorted(tuple(sorted((sw(a),sw(b)))) for a,b in p))
def matrix(p,N):
    d=N**4; M=np.zeros((d,d))
    for J in itertools.product(range(N),repeat=4):
        for I in itertools.product(range(N),repeat=4):
            lab=list(J)+list(I)
            if all(lab[a]==lab[b] for a,b in p):
                r=0
                for x in J: r=r*N+x
                c=0
                for x in I: c=c*N+x
                M[r,c]=1
    return M

# ---- wbalg.py
# Walled Brauer algebra WB_4(N) on 24 diagrams, generic coefficients (numeric or exact)
IDX={p:k for k,p in enumerate(WB)}
n=len(WB)
# structure constants
MUL={}
for a,p in enumerate(WB):
    for b,q in enumerate(WB):
        r,l=compose(p,q); MUL[(a,b)]=(IDX[r],l)
ROT=[IDX[rotate(p)] for p in WB]
ADJ=[IDX[adjoint(p)] for p in WB]
def diagram_from_perm(perm):
    # permutation operator X e_I = e_{I permuted}: output position k gets input position perm[k]
    return tuple(sorted(tuple(sorted((k,4+perm[k]))) for k in range(4)))
def E(i):  # cup-cap at positions i,i+1 (top and bottom), identity elsewhere
    pairs=[(i,i+1),(4+i,4+i+1)]+[(k,4+k) for k in range(4) if k not in (i,i+1)]
    return tuple(sorted(tuple(sorted(x)) for x in pairs))
def F(a,b):  # cup-cap at positions a,b
    pairs=[(a,b),(4+a,4+b)]+[(k,4+k) for k in range(4) if k not in (a,b)]
    return tuple(sorted(tuple(sorted(x)) for x in pairs))

# ---- rf.py
# exact rational functions in one variable N over Q
from fractions import Fraction as Fr
def ptrim(a):
    a=list(a)
    while a and a[-1]==0: a.pop()
    return tuple(a)
def padd(a,b):
    m=max(len(a),len(b)); return ptrim([(a[i] if i<len(a) else 0)+(b[i] if i<len(b) else 0) for i in range(m)])
def pneg(a): return tuple(-x for x in a)
def pmul(a,b):
    if not a or not b: return ()
    r=[Fr(0)]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        if x==0: continue
        for j,y in enumerate(b): r[i+j]+=x*y
    return ptrim(r)
def pdivmod(a,b):
    a=list(a); q=[Fr(0)]*max(len(a)-len(b)+1,1)
    while len(a)>=len(b) and a:
        c=a[-1]/b[-1]; k=len(a)-len(b); q[k]=c
        for i,y in enumerate(b): a[i+k]-=c*y
        a=list(ptrim(a))
    return ptrim(q),ptrim(a)
def pgcd(a,b):
    while b:
        _,r=pdivmod(a,b); a,b=b,r
    return tuple(x/a[-1] for x in a) if a else ()
def peval(a,x):
    r=Fr(0)
    for c in reversed(a): r=r*x+c
    return r
class RF:
    __slots__=('n','d')
    def __init__(s,n,d=(Fr(1),),norm=True):
        n=ptrim(n); d=ptrim(d)
        if norm:
            if not n: d=(Fr(1),)
            else:
                g=pgcd(n,d)
                if len(g)>1:
                    n,_=pdivmod(n,g); d,_=pdivmod(d,g)
                c=d[-1]; n=tuple(x/c for x in n); d=tuple(x/c for x in d)
        s.n=n; s.d=d
    @staticmethod
    def const(c): return RF((Fr(c),))
    @staticmethod
    def var(): return RF((Fr(0),Fr(1)))
    def __add__(s,o):
        if not isinstance(o,RF): o=RF.const(o)
        if s.d==o.d: return RF(padd(s.n,o.n),s.d)
        return RF(padd(pmul(s.n,o.d),pmul(o.n,s.d)),pmul(s.d,o.d))
    __radd__=__add__
    def __neg__(s): return RF(pneg(s.n),s.d,False)
    def __sub__(s,o): return s+(-o if isinstance(o,RF) else RF.const(-o))
    def __mul__(s,o):
        if not isinstance(o,RF): o=RF.const(o)
        return RF(pmul(s.n,o.n),pmul(s.d,o.d))
    __rmul__=__mul__
    def inv(s): return RF(s.d,s.n)
    def __truediv__(s,o):
        if not isinstance(o,RF): o=RF.const(o)
        return s*o.inv()
    def iszero(s): return not s.n
    def ev(s,x): return peval(s.n,x)/peval(s.d,x)
    def __repr__(s): return "(%s)/(%s)"%(list(map(str,s.n)),list(map(str,s.d)))
def solve(A,b):
    # exact Gaussian elimination over RF; A list of rows (lists of RF); returns x (particular), consistency flag, pivots dens
    m=len(A); k=len(A[0])
    M=[list(A[i])+[b[i]] for i in range(m)]
    piv=[];r=0;pivvals=[]
    for c in range(k):
        p=next((i for i in range(r,m) if not M[i][c].iszero()),None)
        if p is None: continue
        M[r],M[p]=M[p],M[r]
        pv=M[r][c]; pivvals.append(pv)
        inv=pv.inv(); M[r]=[x*inv for x in M[r]]
        for i in range(m):
            if i!=r and not M[i][c].iszero():
                f=M[i][c]; M[i]=[x-f*y for x,y in zip(M[i],M[r])]
        piv.append(c); r+=1
        if r==m: break
    consistent=all(M[i][k].iszero() for i in range(r,m))
    x=[RF.const(0)]*k
    for i,c in enumerate(piv): x[c]=M[i][k]
    return x,consistent,piv,pivvals

# ---- main
import itertools, numpy as np, time
from fractions import Fraction as Fr
out=open("mascagni.txt","w")
def say(s):
    print(s); out.write(s+"\n"); out.flush()
# (1) dimension check at N=4
N=4; d=N**4
def idx(I):
    r=0
    for x in I: r=r*N+x
    return r
allI=list(itertools.product(range(N),repeat=4))
C=[]
for (p,q) in [(0,1),(1,2),(2,3)]:
    M=np.zeros((N*N,d))
    for I in allI:
        if I[p]==I[q]:
            rest=[I[k] for k in range(4) if k not in (p,q)]
            M[rest[0]*N+rest[1],idx(I)]+=1
    C.append(M)
C=np.vstack(C)
_,s,vh=np.linalg.svd(C); R4=vh[(s>1e-9).sum():].T
Tm=np.zeros((d,d))
for I in allI: Tm[idx([I[2],I[3],I[0],I[1]]),idx(I)]=1
A=np.vstack([np.eye(d)-R4@R4.T,(np.eye(d)-Tm)/2])
_,s2,_=np.linalg.svd(A)
rng=np.random.default_rng(0); vecs=[]
for _ in range(400):
    x=rng.normal(size=N)+1j*rng.normal(size=N); y=rng.normal(size=N)+1j*rng.normal(size=N)
    y=y-(x@y)/(x@x)*x; vecs.append(np.kron(np.kron(np.kron(x,y),x),y))
sv=np.linalg.svd(np.array(vecs).T,compute_uv=False)
say("N=4: dim r_4 = %d, dim(r_4 cap Sym_tau) = %d, dim V = %d" % (R4.shape[1],(s2<1e-9).sum(),(sv>1e-8*sv[0]).sum()))
# (2) exact projections over Q(N)
NV=RF.var(); ZERO=RF.const(0)
POW=[RF.const(1)]
for _ in range(10): POW.append(POW[-1]*NV)
def mul(x,y,num=None):
    z=[0.0]*n if num is not None else [ZERO]*n
    for a in range(n):
        if (x[a]==0 if num is not None else x[a].iszero()): continue
        for b in range(n):
            if (y[b]==0 if num is not None else y[b].iszero()): continue
            r,l=MUL[(a,b)]
            z[r]=z[r]+x[a]*y[b]*(num**l if num is not None else POW[l])
    return z
def rot(x):
    z=[0*x[0]]*n
    for a in range(n): z[ROT[a]]=z[ROT[a]]+x[a]
    return z
def adj(x):
    z=[0*x[0]]*n
    for a in range(n): z[ADJ[a]]=z[ADJ[a]]+x[a]
    return z
def basis(p):
    z=[ZERO]*n; z[IDX[p]]=RF.const(1); return z
ID=diagram_from_perm([0,1,2,3]); S13=diagram_from_perm([2,1,0,3]); S24=diagram_from_perm([0,3,2,1])
def lm(y):
    cols=[mul(y,basis(WB[b])) for b in range(n)]
    return [[cols[b][a] for b in range(n)] for a in range(n)]
def proj(sign):
    A=[];b=[]
    for y in [basis(S13),basis(S24)]:
        M=lm(y)
        for a in range(n):
            A.append([M[a][c]-(RF.const(sign) if a==c else ZERO) for c in range(n)]); b.append(ZERO)
    for (p,q) in [(0,1),(0,3),(2,1),(2,3)]:
        M=lm(basis(F(p,q)))
        for a in range(n): A.append(M[a]); b.append(ZERO)
    e=[ZERO]*n; e[IDX[ID]]=RF.const(1); A.append(e); b.append(RF.const(Fr(1,4)))
    x,cons,piv,pv=solve(A,b)
    assert cons and len(piv)==n
    return x
PV=proj(1); PW=proj(-1)
for name,P in [("P_V",PV),("P_W",PW)]:
    say("%s in the walled Brauer basis (pairs of the 8 points, top 0-3, bottom 4-7):" % name)
    for k in range(n):
        if not P[k].iszero(): say("  %s  %s" % (WB[k],P[k]))
    sq=mul(P,P)
    say("  idempotent: %s" % all((a-b).iszero() for a,b in zip(sq,P)))
# (3) certificates
def numeric(x,Nn): return np.array([float(c.ev(Fr(Nn))) for c in x])
TLg=[('I',basis(ID)),('E0',basis(E(0))),('E1',basis(E(1))),('E2',basis(E(2)))]
for gname,P in [("TL + P_V",PV),("TL + P_V + P_W",[a+b for a,b in zip(PV,PW)])]:
    Nn=5.0
    gens=TLg+[('P',P)]
    target=numeric(basis(S13),5)
    words=[];vals=[];B=np.zeros((0,n))
    def add(w,v):
        global B
        w2=v-B.T@(B@v) if len(B) else v.copy(); nv=np.linalg.norm(w2)
        if nv>1e-8*max(1,np.linalg.norm(v)):
            B=np.vstack([B,w2/nv]); words.append(w); vals.append(v); return True
        return False
    for k,g in gens: add(('gen',k),numeric(g,5))
    inspan=lambda: np.linalg.norm(target-B.T@(B@target))<1e-9
    k=0; done=False
    while k<len(vals) and not done:
        x=vals[k]
        for w,v in [(('rot',k),np.array(rot(list(x)))),(('adj',k),np.array(adj(list(x))))]:
            add(w,v)
            if inspan(): done=True; break
        if done: break
        for j in range(k+1):
            for w,v in [(('mul',k,j),np.array(mul(list(x),list(vals[j]),num=Nn))),(('mul',j,k),np.array(mul(list(vals[j]),list(x),num=Nn)))]:
                add(w,v)
                if inspan(): done=True; break
            if done: break
        k+=1
    keep=list(range(len(vals)))
    for i in reversed(range(len(vals))):
        trial=[j for j in keep if j!=i]
        At=np.array([vals[j] for j in trial]).T
        y,*_=np.linalg.lstsq(At,target,rcond=None)
        if np.linalg.norm(At@y-target)<1e-9: keep=trial
    # exact replay
    gd=dict(gens); ex=[]
    for w in words:
        if w[0]=='gen': ex.append(gd[w[1]])
        elif w[0]=='rot': ex.append(rot(ex[w[1]]))
        elif w[0]=='adj': ex.append(adj(ex[w[1]]))
        else: ex.append(mul(ex[w[1]],ex[w[2]]))
    A=[[ex[j][a] for j in keep] for a in range(n)]
    x,cons,piv,pv=solve(A,basis(S13))
    # exact check of the certificate: the residual must vanish identically in Q(N)
    comb=[ZERO]*n
    for j,c in zip(keep,x):
        comb=[a+c*b for a,b in zip(comb,ex[j])]
    resid=[a-b for a,b in zip(comb,basis(S13))]
    assert cons and all(r.iszero() for r in resid)
    say("generators %s: derivation with %d words, %d kept; sigma (x) id = combination over Q(N): %s, exact residual zero: %s" % (gname,len(words),len(keep),cons,all(r.iszero() for r in resid)))
    say("   all words, with their definitions (gen = generator, rot/adj = rotation/adjoint of word k, mul k j = product of words k and j):")
    for i,w in enumerate(words): say("     [%d] %s" % (i,w))
    say("   kept words and their coefficients:")
    for j,c in zip(keep,x): say("     [%d] %s  coefficient %s" % (j,str(words[j]),c))
    dens=[c.d for c in x]+[p.n for p in pv]+[p.d for p in pv]+[c.d for c in P]
    # exact factorization against the candidate factors, and exact integer roots
    CAND=[("N",(0,1)),("N+1",(1,1)),("N-1",(-1,1)),("N+2",(2,1)),("N-2",(-2,1)),("N+3",(3,1)),("N-3",(-3,1)),
          ("N^2-6",(-6,0,1)),("N^2-7",(-7,0,1)),("N^2-8",(-8,0,1))]
    found=set(); introots=set()
    for dd in dens:
        a=tuple(Fr(c) for c in dd)
        if len(a)<=1: continue
        for name,cf in CAND:
            b=tuple(Fr(c) for c in cf)
            while len(a)>1:
                q,r=pdivmod(a,b)
                if r: break
                a=q; found.add(name)
        assert len(a)==1, "unexpected irreducible factor %s" % (a,)
        # integer roots of the original polynomial, by the rational root theorem
        from math import lcm
        den=lcm(*[c.denominator for c in dd]); ic=[int(c*den) for c in dd]
        k=0
        while ic[k]==0: k+=1
        if k: introots.add(0)
        c0=abs(ic[k])
        for t in range(1,c0+1):
            if c0%t==0:
                for z in (t,-t):
                    if peval(dd,Fr(z))==0: introots.add(z)
    say("   exact factors of all denominators and pivots: %s" % sorted(found))
    say("   integer roots: %s; exceptional integers N>=4: %s" % (sorted(introots),[z for z in introots if z>=4]))
    assert not [z for z in introots if z>=4]
out.close()
