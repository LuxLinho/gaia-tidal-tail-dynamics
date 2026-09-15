"""Project 04A: explicit Astropy Galactocentric context, without orbit dynamics."""
from pathlib import Path
import hashlib
import importlib.util
import json

import numpy as np
import astropy
from astropy import units as u
from astropy.coordinates import (SkyCoord, ICRS, Galactic, Galactocentric,
    CartesianRepresentation, CartesianDifferential, CylindricalRepresentation, CylindricalDifferential)
from astropy.table import Table

_spec=importlib.util.spec_from_file_location('project03d',Path(__file__).parents[1]/'project03/04_relative_morphology_coherence.py')
d=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d)
a,b,c,plt=d.a,d.b,d.c,d.plt
ROOT=d.ROOT
CONFIG=Path(__file__).with_name('galactocentric_parameters.json')
INPUT=ROOT/'data/interim/project03/stock2_relative_morphology.ecsv'
REFERENCE=d.REFERENCE
PCA=ROOT/'data/interim/project03/stock2_morphology_pca.ecsv'
OUTPUT=ROOT/'data/interim/project04'
FIGURES=ROOT/'results/project04/galactocentric_context'
POSITION=['x_gc_kpc','y_gc_kpc','z_gc_kpc']
VELOCITY=['vx_gc_kms','vy_gc_kms','vz_gc_kms']
UNITS={**dict.fromkeys(POSITION+['R_gc_kpc','r_gc_kpc'],'kpc'),'phi_gc_deg':'deg',
       **dict.fromkeys(VELOCITY+['v_gc_kms','v_R_gc_kms','v_phi_gc_kms'],'km / s')}
CONVENTION=('Astropy Galactocentric, right-handed, Galactic centre at origin. '
    'The solar-system origin is at negative X, Y=0, positive Z=z_sun; +X points '
    'from the solar projection through the centre toward the opposite side. '
    '+Y is approximately Galactic l=90 degrees and the adopted solar rotation direction; '
    '+Z is toward the northern side of the adopted midplane. Velocities are positive '
    'along the same axes. Solar height induces Astropy\'s small frame tilt. '
    'All five public frame parameters are explicit; no default parameter values are used.')
CYLINDER=('R=hypot(X,Y); phi=atan2(Y,X) in [-180,180] degrees, increasing from +X toward +Y. '
    'e_R=(X/R,Y/R,0) points outward in the plane; e_phi=(-Y/R,X/R,0) is increasing azimuth; '
    'e_Z=(0,0,1). v_R=(X*Vx+Y*Vy)/R; v_phi=(-Y*Vx+X*Vy)/R; cylindrical v_z=vz_gc_kms. '
    'With Sun at negative X and positive Vy, increasing azimuth opposes adopted local '
    'solar rotation: solar v_phi is negative. R=0 has undefined phi,v_R,v_phi (NaN).')
LABELS={q:q.replace('_gc_kpc','').replace('_gc_kms','')+(' [kpc]' if unit=='kpc' else ' [km/s]')
        for q,unit in UNITS.items()}
LABELS.update(phi_gc_deg='Galactocentric azimuth [deg]',R_gc_kpc='R [kpc]',z_gc_kpc='Z [kpc]',
              v_R_gc_kms=r'$v_R$ [km/s]',v_phi_gc_kms=r'$v_\phi$ [km/s]',vz_gc_kms=r'$v_z$ [km/s]')


def load_frame(path=CONFIG):
    config=json.loads(Path(path).read_text())
    def quantity(key):
        entry=config[key]
        return np.asarray(entry['value'])*u.Unit(entry['unit'])
    coord=config['galcen_coord']
    if coord['frame']!='icrs': raise ValueError('Galactic centre direction must be ICRS')
    frame=Galactocentric(galcen_distance=quantity('galcen_distance'),z_sun=quantity('z_sun'),
        galcen_v_sun=CartesianDifferential(quantity('galcen_v_sun')),
        galcen_coord=ICRS(ra=coord['ra']*u.Unit(coord['unit']),dec=coord['dec']*u.Unit(coord['unit'])),
        roll=quantity('roll'))
    if frame.galcen_distance<=0*u.kpc or abs(frame.z_sun)>=frame.galcen_distance:
        raise ValueError('Invalid Galactic distance/height')
    return frame,config


def galactic_cartesian(xyz,velocity=None):
    representation=CartesianRepresentation(xyz*u.pc)
    if velocity is not None:
        representation=representation.with_differentials(CartesianDifferential(velocity*u.km/u.s))
    return SkyCoord(Galactic(representation))


def cylindrical(xyz,velocity):
    x,y,z=xyz
    R=np.hypot(x,y)
    valid=R>0
    phi=np.full_like(R,np.nan)
    vR=np.full_like(R,np.nan)
    vphi=np.full_like(R,np.nan)
    phi[valid]=np.degrees(np.arctan2(y[valid],x[valid]))
    vR[valid]=(x[valid]*velocity[0,valid]+y[valid]*velocity[1,valid])/R[valid]
    vphi[valid]=(-y[valid]*velocity[0,valid]+x[valid]*velocity[1,valid])/R[valid]
    return dict(R_gc_kpc=R,r_gc_kpc=np.linalg.norm(xyz,axis=0),phi_gc_deg=phi,
                v_R_gc_kms=vR,v_phi_gc_kms=vphi,v_gc_kms=np.linalg.norm(velocity,axis=0))


def transform_table(table,frame,config):
    result=table.copy(copy_data=True)
    for q,unit in UNITS.items():
        if q in result.colnames: raise ValueError(f'Column already exists: {q}')
        result[q]=np.full(len(table),np.nan)*u.Unit(unit)
    xyz,pos=c.vectors(table,b.POSITION)
    velocity,vel=c.vectors(table,b.VELOCITY)
    transformed=galactic_cartesian(xyz[:,pos]).transform_to(frame)
    position=transformed.cartesian.xyz.to_value(u.kpc)
    for q,row in zip(POSITION,position):result[q][pos]=row
    if vel.any():
        transformed6d=galactic_cartesian(xyz[:,vel],velocity[:,vel]).transform_to(frame)
        for q,row in zip(VELOCITY,transformed6d.velocity.d_xyz.to_value(u.km/u.s)):result[q][vel]=row
    all_position=np.array([a.values(result[q]) for q in POSITION])
    all_velocity=np.array([a.values(result[q]) for q in VELOCITY])
    for q,values in cylindrical(all_position,all_velocity).items():result[q][:]=values
    result.meta.update(project04a_parameters=config,project04a_astropy_version=astropy.__version__,
                       project04a_convention=CONVENTION,project04a_cylindrical=CYLINDER,
                       project04a_input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
                       project04a_reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
                       project04a_pca_sha256=hashlib.sha256(PCA.read_bytes()).hexdigest())
    return result


def transform_reference(stored,frame):
    xyz=np.array([[stored[q][0]] for q in c.REF_POSITION])
    velocity=np.array([[stored[q][0]] for q in c.REF_VELOCITY])
    gc=galactic_cartesian(xyz,velocity).transform_to(frame)
    pos=gc.cartesian.xyz.to_value(u.kpc)
    vel=gc.velocity.d_xyz.to_value(u.km/u.s)
    ref=Table()
    for q,row in zip(POSITION,pos):ref[q]=row*u.kpc
    for q,row in zip(VELOCITY,vel):ref[q]=row*u.km/u.s
    for q,value in cylindrical(pos,vel).items():ref[q]=value*u.Unit(UNITS[q])
    return ref


def rotate_direction(axis,frame):
    # Transform two positions through Astropy, then cancel the translation.
    # A +/-1 kpc baseline limits cancellation without introducing hand-coded rotations.
    endpoints=galactic_cartesian(np.outer(axis,[-1000.,1000.])).transform_to(frame).cartesian.xyz.to_value(u.kpc)
    direction=(endpoints[:,1]-endpoints[:,0])/2
    np.testing.assert_allclose(np.linalg.norm(direction),1.,atol=1e-12)
    return direction/np.linalg.norm(direction)


def local_basis(position):
    x,y,_=position
    radius=np.hypot(x,y)
    if radius==0: raise ValueError('Local radial basis undefined at R=0')
    return np.array([[x/radius,y/radius,0],[-y/radius,x/radius,0],[0,0,1]])


def orientations(axis,reference):
    position=np.array([reference[q][0] for q in POSITION])
    velocity=np.array([reference[q][0] for q in VELOCITY])
    basis=local_basis(position)
    tangent=velocity/np.linalg.norm(velocity)
    rows=[]
    for name,direction in zip(['radial_outward','azimuth_increasing','vertical_positive','instantaneous_velocity'],[*basis,tangent]):
        dot=float(np.clip(np.dot(axis,direction),-1,1))
        rows.append(dict(direction=name,unit_x=direction[0],unit_y=direction[1],unit_z=direction[2],
                         signed_dot=dot,absolute_dot=abs(dot),directed_angle_deg=float(np.degrees(np.arccos(dot))),
                         unsigned_alignment_deg=float(np.degrees(np.arccos(abs(dot))))))
    output=Table(rows=rows)
    output['directed_angle_deg'].unit=u.deg
    output['unsigned_alignment_deg'].unit=u.deg
    output.meta.update(definition='Directed angle follows stored PCA sign; unsigned alignment treats it as an unoriented axis (0–90 deg).',
                       interpretation='Geometric context only; no tidal-origin inference or orbital-path alignment.')
    return basis,tangent,output


def solar_context(frame):
    sun=galactic_cartesian(np.zeros((3,1)),np.zeros((3,1))).transform_to(frame)
    xyz=sun.cartesian.xyz.to_value(u.kpc)
    velocity=sun.velocity.d_xyz.to_value(u.km/u.s)
    values=cylindrical(xyz,velocity)
    np.testing.assert_allclose(xyz[:,0],[-np.sqrt(frame.galcen_distance.to_value(u.kpc)**2-frame.z_sun.to_value(u.kpc)**2),0,frame.z_sun.to_value(u.kpc)],atol=1e-12)
    np.testing.assert_allclose(velocity[:,0],frame.galcen_v_sun.xyz.to_value(u.km/u.s),atol=1e-12)
    if values['v_phi_gc_kms'][0]>=0:raise ValueError('Solar azimuthal convention not as documented')
    return xyz[:,0],float(values['v_phi_gc_kms'][0])


def validate_inputs(table,reference,pca):
    # No calls to reference_frame(), spatial_pca(), or pca_populations().
    d.validate_input(table,reference)
    ext=pca[pca['population']=='L+T']
    if len(ext)!=3 or list(ext['component'])!=[1,2,3] or not np.all(ext['canonical_projection_basis']):
        raise ValueError('Invalid stored L+T PCA metadata')
    basis=np.array([ext[q] for q in ['e_x','e_y','e_z']]).T
    np.testing.assert_array_equal(basis,np.asarray(table.meta['project03d_basis']))
    if table.meta['project03d_reference_sha256']!=hashlib.sha256(REFERENCE.read_bytes()).hexdigest():
        raise ValueError('Project 03D reference hash mismatch')
    d.validate_projection(Table.read(d.INPUT),table,basis)
    return basis[0].copy()


def validate_cylindrical(table):
    xyz=np.array([a.values(table[q]) for q in POSITION])
    vel=np.array([a.values(table[q]) for q in VELOCITY])
    phi=np.radians(a.values(table['phi_gc_deg']))
    radius=a.values(table['R_gc_kpc'])
    np.testing.assert_allclose(radius*np.cos(phi),xyz[0],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(radius*np.sin(phi),xyz[1],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(a.values(table['r_gc_kpc']),np.linalg.norm(xyz,axis=0),rtol=1e-12)
    valid=np.all(np.isfinite(vel),axis=0)
    # Independent Astropy Cartesian -> cylindrical differential check.
    rep=CartesianRepresentation(xyz[:,valid]*u.kpc,differentials=CartesianDifferential(vel[:,valid]*u.km/u.s))
    cy=rep.represent_as(CylindricalRepresentation,CylindricalDifferential)
    diff=cy.differentials['s']
    np.testing.assert_allclose(a.values(table['v_R_gc_kms'])[valid],diff.d_rho.to_value(u.km/u.s),atol=1e-10)
    vphi=(cy.rho*diff.d_phi).to_value(u.km/u.s,equivalencies=u.dimensionless_angles())
    np.testing.assert_allclose(a.values(table['v_phi_gc_kms'])[valid],vphi,atol=1e-10)
    vR=a.values(table['v_R_gc_kms'])[valid]
    np.testing.assert_allclose(vR*np.cos(phi[valid])-vphi*np.sin(phi[valid]),vel[0,valid],atol=1e-10)
    np.testing.assert_allclose(vR*np.sin(phi[valid])+vphi*np.cos(phi[valid]),vel[1,valid],atol=1e-10)
    np.testing.assert_allclose(a.values(table['v_gc_kms'])[valid],np.linalg.norm(vel[:,valid],axis=0),atol=1e-10)


def validate_result(original,result,reference,frame):
    d.preserve_columns(original,result)
    if set(result.colnames)-set(original.colnames)!=set(UNITS):raise ValueError('Unexpected new quantities')
    finite_rv=np.isfinite(a.values(original['radial_velocity']))
    for q,unit in UNITS.items():
        if result[q].unit!=u.Unit(unit):raise ValueError(f'Wrong unit: {q}')
        expected=finite_rv if unit=='km / s' else np.ones(len(result),dtype=bool)
        np.testing.assert_array_equal(np.isfinite(a.values(result[q])),expected)
    xyz=np.array([a.values(result[q]) for q in POSITION])
    velocity=np.array([a.values(result[q]) for q in VELOCITY])
    refpos=np.array([reference[q][0] for q in POSITION])
    refvel=np.array([reference[q][0] for q in VELOCITY])
    distance=np.linalg.norm(xyz-refpos[:,None],axis=0)*1000
    speed=np.linalg.norm(velocity[:,finite_rv]-refvel[:,None],axis=0)
    distance_error=float(np.max(np.abs(distance-a.values(original['r_cluster_pc']))))
    speed_error=float(np.max(np.abs(speed-a.values(original['dv_cluster_kms'])[finite_rv])))
    np.testing.assert_allclose(distance,a.values(original['r_cluster_pc']),rtol=1e-11,atol=1e-8)
    np.testing.assert_allclose(speed,a.values(original['dv_cluster_kms'])[finite_rv],rtol=1e-11,atol=1e-9)
    validate_cylindrical(result)
    validate_cylindrical(reference)
    # Full Astropy inverse transform of stored GC Cartesian vectors.
    rep=CartesianRepresentation(xyz[:,finite_rv]*u.kpc,differentials=CartesianDifferential(velocity[:,finite_rv]*u.km/u.s))
    back=SkyCoord(frame.realize_frame(rep)).galactic
    original_xyz=np.array([a.values(original[q]) for q in b.POSITION[:3]])
    original_v=np.array([a.values(original[q]) for q in b.VELOCITY[:3]])
    np.testing.assert_allclose(back.cartesian.xyz.to_value(u.pc),original_xyz[:,finite_rv],atol=1e-8,rtol=1e-11)
    np.testing.assert_allclose(back.velocity.d_xyz.to_value(u.km/u.s),original_v[:,finite_rv],atol=1e-9,rtol=1e-11)
    return dict(max_distance_error_pc=distance_error,max_relative_speed_error_kms=speed_error)


def reference_metadata(reference,config,local,axis_old,axis_gc,tangent,angles,sun,sun_vphi,checks):
    result=reference.copy()
    for name,vector in [('e_R_gc',local[0]),('e_phi_gc',local[1]),('e_Z_gc',local[2]),
                        ('e1_ext_project03',axis_old),('e1_ext_gc',axis_gc),('velocity_unit_gc',tangent)]:
        result[name]=[vector]
    result['sun_position_gc_kpc']=[sun]*u.kpc
    result['sun_v_phi_kms']=[sun_vphi]*u.km/u.s
    for row in angles:
        for field in ['signed_dot','absolute_dot','directed_angle_deg','unsigned_alignment_deg']:
            q=f'{row["direction"]}_{field}'
            result[q]=[row[field]]
            if field.endswith('deg'):result[q].unit=u.deg
    result['max_distance_error_pc']=[checks['max_distance_error_pc']]*u.pc
    result['max_relative_speed_error_kms']=[checks['max_relative_speed_error_kms']]*u.km/u.s
    result.meta.update(adopted_parameters=config,astropy_version=astropy.__version__,convention=CONVENTION,
                       cylindrical_convention=CYLINDER,reference_source=str(REFERENCE.relative_to(ROOT)),
                       reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
                       pca_source=str(PCA.relative_to(ROOT)),pca_sha256=hashlib.sha256(PCA.read_bytes()).hexdigest(),
                       axis_transform='Astropy transform of +/-1 kpc positions along stored axis; difference cancels translation. No PCA refit.',
                       reference_columns='x_gc_kpc etc in this one-row product are the transformed fixed Project 03C reference, not sample medians.')
    return result


def summaries(table):
    def rows(mask,field,group):
        return [dict(group_column=field,group=group,n_position=int(mask.sum()),
                     n_velocity=int(np.isfinite(a.values(table['vx_gc_kms'])[mask]).sum()),
                     quantity=q,unit=unit,**a.percentile_summary(table[q][mask])) for q,unit in UNITS.items()]
    full=Table(rows=rows(np.ones(len(table),dtype=bool),'all','all'))
    groups=Table(rows=[row for field,labels in a.GROUPS.items() for group in labels
                      for row in rows(np.asarray(table[field])==group,field,group)])
    for t in (full,groups):t.meta.update(convention=CONVENTION,cylindrical=CYLINDER,statistics='Linear percentiles of finite values, no clipping; units per row.')
    return full,groups


def local_positions(table,reference,basis):
    position=np.array([a.values(table[q]) for q in POSITION])
    centre=np.array([reference[q][0] for q in POSITION])
    return basis@(position-centre[:,None])*1000


def scatter_arrays(table,x,y,xlabel,ylabel,equal=False):
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    for color,group in zip(a.COLORS,a.GROUPS['homogenized_class']):
        g=np.asarray(table['homogenized_class'])==group
        mask=g&np.isfinite(x)&np.isfinite(y)
        ax.scatter(x[mask],y[mask],s=12,alpha=.45,color=color,edgecolors='none',label=f'{group} (N={mask.sum()}/{g.sum()})')
    ax.set(xlabel=xlabel,ylabel=ylabel,title='Stock 2 — explicit Galactocentric frame')
    if equal:ax.set_aspect('equal',adjustable='box')
    ax.legend(fontsize=8,loc='upper left',bbox_to_anchor=(1.02,1))
    ax.grid(alpha=.18)
    return fig,ax


def galactic_location(table,reference,sun):
    fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
    ax.scatter(table['x_gc_kpc'],table['y_gc_kpc'],s=4,alpha=.25,label='Canonical population (1456)')
    ax.plot(0,0,'k+',ms=10,label='Galactic centre')
    ax.plot(sun[0],sun[1],marker='*',color='#E69F00',ms=10,linestyle='none',label='Sun / solar-system origin')
    ax.plot(reference['x_gc_kpc'][0],reference['y_gc_kpc'][0],marker='x',color='#D55E00',ms=8,linestyle='none',label='Fixed Stock 2 reference')
    extent=max(abs(np.array(table['x_gc_kpc'])).max(),abs(np.array(table['y_gc_kpc'])).max(),np.linalg.norm(sun[:2]))*1.1
    ax.set(xlim=(-extent,extent),ylim=(-extent,extent),xlabel='Galactocentric X [kpc]',ylabel='Galactocentric Y [kpc]',
           title='Galactic-scale location — individual cluster structure is unresolved here')
    ax.set_aspect('equal',adjustable='box')
    ax.legend(fontsize=8,loc='upper left',bbox_to_anchor=(1.02,1))
    ax.grid(alpha=.18)
    return fig,ax


def vector_diagnostic(axis,local,tangent):
    fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained')
    # Unit vectors expressed in local radial, increasing-azimuth, vertical coordinates.
    vectors=[local@axis,np.array([1.,0,0]),np.array([0.,1,0]),np.array([0.,0,1]),local@tangent]
    names=['Fixed L+T major spatial axis','Outward Galactic radial','Increasing Galactic azimuth','Positive Galactic vertical','Instantaneous velocity direction']
    colors=['#CC79A7','#D55E00','#0072B2','#009E73','#444444']
    labels=['Radial component','Azimuthal component','Vertical component']
    for ax,(i,j) in zip(axes,[(0,1),(0,2),(1,2)]):
        ax.plot(0,0,'k+',ms=8)
        for v,name,color in zip(vectors,names,colors):
            ax.plot([0,v[i]],[0,v[j]],color=color,label=name,linewidth=1.7)
            ax.annotate('',xy=(v[i],v[j]),xytext=(0,0),arrowprops=dict(arrowstyle='->',color=color,lw=1.2))
        ax.set(xlim=(-1.15,1.15),ylim=(-1.15,1.15),xlabel=labels[i],ylabel=labels[j])
        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.18)
    axes[1].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.2),ncol=2)
    fig.suptitle('Stock 2 local unit-vector context — direction only; no orbit integrated')
    return fig,axes


def generate_plots(table,reference,local,axis,tangent,sun,directory):
    outputs=[]
    def save(fig,stem):
        for ext in ('png','pdf'):
            path=directory/f'{stem}.{ext}';fig.savefig(path,dpi=200);outputs.append(path)
        plt.close(fig)
    fig,_=galactic_location(table,reference,sun);save(fig,'galactic_xy_location')
    fig,_=scatter_arrays(table,a.values(table['R_gc_kpc']),a.values(table['z_gc_kpc']),LABELS['R_gc_kpc'],LABELS['z_gc_kpc'],True)
    save(fig,'R_gc_vs_Z_gc_by_class')
    local_xyz=local_positions(table,reference,local)
    names=['radial','azimuthal','vertical']
    for i,j in [(0,1),(0,2),(1,2)]:
        fig,ax=scatter_arrays(table,local_xyz[i],local_xyz[j],f'Local {names[i]} displacement [pc]',f'Local {names[j]} displacement [pc]',True)
        ax.plot(0,0,'k+',ms=8)
        save(fig,f'local_{names[i]}_vs_{names[j]}_by_class')
    for x,y in [('v_R_gc_kms','v_phi_gc_kms'),('v_R_gc_kms','vz_gc_kms'),('v_phi_gc_kms','vz_gc_kms')]:
        fig,_=scatter_arrays(table,a.values(table[x]),a.values(table[y]),LABELS[x],LABELS[y]);save(fig,f'{x}_vs_{y}_by_class')
    fig,_=vector_diagnostic(axis,local,tangent);save(fig,'local_direction_alignment')
    return outputs


def protected_hashes():
    hashes=d.protected_hashes()
    directories=['data/interim/project03','results/project03','src/project01','src/project02','src/project03',
                 'data/processed','results/project01']
    for directory in directories:
        for p in (ROOT/directory).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (ROOT/'docs').glob('project0[123]*'):
        if p.is_file():hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return hashes


def make_report(config,reference,angles,checks,groups,outputs):
    lines=['Project 04A — Galactocentric transformation and Galactic context','='*70,
           'Adopted Galactic parameters (modelling assumptions, not measured Stock 2 properties):',
           json.dumps(config,indent=2),f'Astropy version: {astropy.__version__}',CONVENTION,CYLINDER,
           '', 'Fixed Project 03C reference transformed without recomputation:']
    for q,unit in UNITS.items():lines.append(f'{q}: {reference[q][0]:.10f} [{unit}]')
    lines+=['','Transformation validation:',f'Maximum positional invariance error: {checks["max_distance_error_pc"]:.12g} pc',
            f'Maximum relative-speed invariance error: {checks["max_relative_speed_error_kms"]:.12g} km/s',
            'Tolerance: position rtol=1e-11, atol=1e-8 pc; relative speed rtol=1e-11, atol=1e-9 km/s.',
            'Rows/unique IDs: 1456/1456. Positions: 1456. Velocities: 886. Missing velocities: 570.',
            'C/L/T: 940/184/332. Cartesian/cylindrical consistency and full 6D Astropy inverse transform passed.',
            f'Adopted Sun v_phi sign check: {reference["sun_v_phi_kms"][0]:.6f} km/s (negative).',
            '', 'Fixed L+T major-axis geometric context:']
    for row in angles:
        lines.append(f'{row["direction"]}: signed dot={row["signed_dot"]:.9f}, absolute dot={row["absolute_dot"]:.9f}, '
                     f'directed angle={row["directed_angle_deg"]:.6f} deg, unsigned alignment={row["unsigned_alignment_deg"]:.6f} deg')
    lines+=['Directed angles follow the stored PCA sign; unsigned angles are sign-invariant, 0–90 degrees.',
            'Alignment with instantaneous Galactocentric velocity is not orbital-path alignment or evidence of tidal origin.',
            '', 'Population summaries:']
    for row in groups:
        if row['quantity'] in ['R_gc_kpc','z_gc_kpc','v_R_gc_kms','v_phi_gc_kms','vz_gc_kms']:
            lines.append(f'{row["group_column"]}/{row["group"]}: N position={row["n_position"]}, N velocity={row["n_velocity"]}; '
                         f'{row["quantity"]} median={row["median"]:.8f} [{row["unit"]}]')
    lines+=['','Outputs:']+[str(p.relative_to(ROOT)) for p in outputs]
    lines+=['','Invariants:','rows == 1456: True','unique source_id == 1456: True','C == 940: True','L == 184: True','T == 332: True',
            'valid Galactocentric positions == 1456: True','valid Galactocentric velocities == 886: True',
            'missing-RV rows retain NaN velocities: True','Project 03 reference position unchanged: True',
            'Project 03 reference velocity unchanged: True','Project 03D PCA axis unchanged: True',
            'cluster-relative distances numerically preserved: True','cluster-relative speed magnitudes numerically preserved: True',
            'Project 01–03D products unchanged: True','no source removed: True','no clipping: True',
            'no membership criterion introduced: True','no orbit integrated: True','no Galactic potential instantiated: True',
            'no actions calculated: True','PROJECT 04A STATUS: PASS']
    return '\n'.join(lines)+'\n'


def main():
    before=protected_hashes()
    frame,config=load_frame()
    print('Explicit Galactic parameters before any transformation:',json.dumps(config),f'Astropy {astropy.__version__}',flush=True)
    table,stored,pca=Table.read(INPUT),Table.read(REFERENCE),Table.read(PCA)
    axis_old=validate_inputs(table,stored,pca)
    reference=transform_reference(stored,frame)
    result=transform_table(table,frame,config)
    checks=validate_result(table,result,reference,frame)
    axis_gc=rotate_direction(axis_old,frame)
    local,tangent,angles=orientations(axis_gc,reference)
    np.testing.assert_allclose(local@local.T,np.eye(3),atol=1e-12)
    sun,sun_vphi=solar_context(frame)
    reference=reference_metadata(reference,config,local,axis_old,axis_gc,tangent,angles,sun,sun_vphi,checks)
    full,groups=summaries(result)
    OUTPUT.mkdir(parents=True,exist_ok=True);FIGURES.mkdir(parents=True,exist_ok=True)
    outputs=[]
    products=[('stock2_galactocentric_phase_space',result),('stock2_galactocentric_reference',reference),
              ('stock2_galactocentric_summary',full),('stock2_galactocentric_group_summary',groups),('stock2_galactocentric_alignments',angles)]
    for stem,t in products:
        path=OUTPUT/f'{stem}.ecsv';t.write(path,overwrite=True);outputs.append(path)
    validate_result(table,Table.read(outputs[0]),Table.read(outputs[1]),frame)
    outputs+=generate_plots(result,reference,local,axis_gc,tangent,sun,FIGURES)
    if protected_hashes()!=before:raise RuntimeError('Earlier project products changed')
    report_path=FIGURES/'galactocentric_context_report.txt';outputs.append(report_path)
    report=make_report(config,reference,angles,checks,groups,outputs)
    report_path.write_text(report,encoding='utf-8');print(report)
    return result,reference,angles


if __name__=='__main__':main()
