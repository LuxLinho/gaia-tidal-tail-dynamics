"""Project 07 geometry and reproducibility helpers; upstream products read only."""
from pathlib import Path
import hashlib
import importlib.util
import json
import platform
import astropy
import galpy
import numpy as np
from astropy import units as u
from astropy.table import Table

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/interim/project07'
RESULTS=ROOT/'results/project07'
GC=ROOT/'data/interim/project04/stock2_galactocentric_phase_space.ecsv'
REFERENCE=ROOT/'data/interim/project05/stock2_cluster_reference.ecsv'
ZONES=ROOT/'data/interim/project06/stock2_candidate_tidal_zones.ecsv'
JACOBI=ROOT/'data/interim/project06/stock2_jacobi_radius.ecsv'
GEOMETRY=DATA/'stock2_clustercentric_geometry.ecsv'
AXES=DATA/'stock2_morphology_axes.ecsv'
BOOT=DATA/'stock2_orientation_bootstrap.ecsv'
FIELDS=['local_radial_pc','local_prograde_pc','local_vertical_pc']
SEED=20260921
N_BOOT=1000
SOFTWARE=dict(python=platform.python_version(),numpy=np.__version__,astropy=astropy.__version__,galpy=galpy.__version__)
FRAME='Basis columns in Astropy GC axes: e_R=(X/R,Y/R,0), e_pro=(Y/R,-X/R,0), e_Z=(0,0,1). B^T B=I, det(B)=-1. Outward/prograde/north is LEFT-HANDED. Coordinates are (r-r_ref) dot each fixed axis, in pc, not differences of cylindrical R or phi.'


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

p04=load_module('project07_upstream_tensor','src/project04/03_local_tidal_field.py')

def read(path): return Table.read(path,format='ascii.ecsv')

def arr(t,key,unit=None): return np.asarray(t[key].quantity.to_value(unit or t[key].unit),float)

def xyz(t): return np.column_stack([arr(t,q,u.pc) for q in FIELDS])

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def hashes(paths): return {str(p.relative_to(ROOT)):sha(p) for p in paths}

def plain(value):
    if isinstance(value,dict): return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [plain(v) for v in value]
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    return value

def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n')

def save(table,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    table.write(path,format='ascii.ecsv',overwrite=True)
    return read(path)

def metadata(stage,inputs):
    return dict(project=stage,frame=FRAME,software=SOFTWARE,inputs=hashes(inputs))

def snapshot():
    paths=[]
    for folder in ('data','results','src','tests','docs'):
        paths += [p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'project07' not in str(p.relative_to(ROOT))]
    return hashes(paths)

def check_hashes(h):
    for name,expected in h.items():
        if sha(ROOT/name)!=expected: raise ValueError(f'Protected upstream file changed: {name}')

def protect(): check_hashes(json.loads((RESULTS/'upstream_integrity.json').read_text())['sha256'])

def basis(ref):
    pos=np.array([ref[q][0] for q in ['x_gc_kpc','y_gc_kpc','z_gc_kpc']]); x,y,z=pos
    R=np.hypot(x,y)
    if R<=0: raise ValueError('Undefined radial basis')
    B=np.column_stack(([x/R,y/R,0.],[y/R,-x/R,0.],[0.,0.,1.]))
    np.testing.assert_allclose(B.T@B,np.eye(3),atol=1e-14)
    np.testing.assert_allclose(np.linalg.det(B),-1.,atol=1e-14)
    vel=np.array([ref[q][0] for q in ['vx_gc_kms','vy_gc_kms','vz_gc_kms']])
    if vel@B[:,1]<=0: raise ValueError('Chosen prograde direction is inconsistent with reference')
    return B,pos,vel

def unit(v):
    v=np.asarray(v,float); length=np.linalg.norm(v)
    if not np.isfinite(length) or length<=0: raise ValueError('Undefined direction')
    return v/length

def acute(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    a=a/np.linalg.norm(a,axis=-1,keepdims=True); b=b/np.linalg.norm(b,axis=-1,keepdims=True)
    return np.degrees(np.arccos(np.clip(np.abs(np.sum(a*b,axis=-1)),0,1)))

def directions(ref):
    B,pos,vel=basis(ref)
    return dict(radial_outward=np.array([1.,0,0]),radial_inward=np.array([-1.,0,0]),
                galactic_centre_toward=B.T@unit(-pos),galactic_centre_away=B.T@unit(pos),
                prograde=np.array([0.,1.,0]),orbital_velocity=B.T@unit(vel),vertical=np.array([0.,0,1.]))

def groups(table):
    r=arr(table,'r_cluster_pc',u.pc); rj=table.meta['r_J_pc']
    c=np.asarray(table['homogenized_class'])
    return dict(full=np.ones(len(table),bool),C=c=='C',L=c=='L',T=c=='T',
                extended_LT=np.isin(c,['L','T']),inner=r<rj,intermediate=(r>=rj)&(r<2*rj),
                outer=r>=2*rj,outside_rJ=r>=rj)

def eigensystem(matrix):
    values,vectors=np.linalg.eigh(matrix)
    order=np.argsort(values)[::-1]; values=values[order]; vectors=vectors[:,order]
    # Signs are display-only; all inference uses projectors or absolute dot products.
    for i in range(3):
        if vectors[np.argmax(np.abs(vectors[:,i])),i]<0: vectors[:,i]*=-1
    return values,vectors

def fit(points,robust=False):
    X=np.asarray(points,float)
    if X.ndim!=2 or X.shape[1]!=3 or len(X)<4 or not np.isfinite(X).all(): raise ValueError('Need >=4 finite 3D points')
    center=np.median(X,axis=0) if robust else X.mean(axis=0)
    Y=X-center
    if robust:
        norms=np.linalg.norm(Y,axis=1)
        Y=np.divide(Y,norms[:,None],out=np.zeros_like(Y),where=norms[:,None]>0)
        matrix=Y.T@Y/len(Y)
    else: matrix=Y.T@Y/(len(Y)-1)
    w,V=eigensystem(matrix)
    if w[0]<=0 or w[-1]<-1e-10: raise ValueError('Invalid shape scatter')
    w=np.maximum(w,0); lengths=np.sqrt(w)
    return dict(n=len(X),center=center,matrix=matrix,eigenvalues=w,vectors=V,scales=lengths,
                b_a=lengths[1]/lengths[0],c_a=lengths[2]/lengths[0],
                major_gap=(w[0]-w[1])/w[0],major_defined=bool((w[0]-w[1])/w[0]>1e-8))

def bootstrap(points,seed,n=N_BOOT):
    rng=np.random.default_rng(seed); output=[]; gaps=[]
    for _ in range(n):
        f=fit(points[rng.integers(0,len(points),size=len(points))])
        if not f['major_defined']: raise ValueError('Bootstrap major axis degenerate; review population')
        output.append(f['vectors'][:,0]); gaps.append(f['major_gap'])
    return np.array(output),np.array(gaps)

def axial_summary(vectors,reference):
    projector=np.einsum('ni,nj->ij',vectors,vectors)/len(vectors)
    w,V=eigensystem(projector)
    deviations=acute(vectors,reference)
    return dict(projector_mean=projector,consensus_axis=V[:,0],concentration=w[0],
                deviation_p50_deg=np.percentile(deviations,50),deviation_p68_deg=np.percentile(deviations,68),
                deviation_p95_deg=np.percentile(deviations,95))

def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.18,'savefig.dpi':220})
    return plt

def savefig(fig,folder,name):
    folder.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'): fig.savefig(folder/f'{name}.{ext}',bbox_inches='tight')
