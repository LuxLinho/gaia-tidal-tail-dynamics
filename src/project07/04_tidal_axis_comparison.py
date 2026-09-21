"""07D: inertial gravitational tensor, morphology alignment and synthesis."""
from geometry_support import *


def tidal_axes(ref):
    if galpy.__version__!='1.12.0' or p04.RO_KPC!=8. or p04.VO_KMS!=220.: raise ValueError('Potential convention changed')
    model=read(JACOBI).meta['model']
    if model['name']!='galpy MWPotential2014' or model['ro_kpc']!=8 or model['vo_kms']!=220: raise ValueError('Project06 potential mismatch')
    B,pos,vel=basis(ref)
    Tgc,H,b=p04.tidal_tensor_at(*pos)
    Tlocal=B.T@Tgc@B
    w,V=eigensystem(Tlocal)
    np.testing.assert_allclose(Tlocal,Tlocal.T,atol=1e-10)
    np.testing.assert_allclose(Tlocal@V,V*w,atol=1e-9,rtol=1e-12)
    return Tlocal,Tgc,w,V


def main():
    protect(); ref=read(REFERENCE); geometry=read(GEOMETRY); shapes=read(AXES); boot=read(BOOT)
    T,Tgc,w,V=tidal_axes(ref); B,pos,vel=basis(ref)
    rows=[]
    for i in range(3):
        gc=B@V[:,i]
        rows.append((i+1,w[i],*V[:,i],*gc,'stretching' if w[i]>0 else 'compression' if w[i]<0 else 'neutral'))
    tensor=Table(rows=rows,names=['component','eigenvalue','e_radial','e_prograde','e_vertical','e_astropy_X','e_astropy_Y','e_astropy_Z','interpretation'])
    tensor['eigenvalue'].unit=(u.km/u.s/u.kpc)**2
    tensor.meta=metadata('07D',[REFERENCE,JACOBI,ROOT/'src/project04/03_local_tidal_field.py'])
    tensor.meta.update(definition='T_ij=partial a_i/partial x_j=-partial_i partial_j Phi. INERTIAL gravitational differential acceleration at actual (R,z); positive eigenvalue stretches, negative compresses. No centrifugal/Coriolis terms.',
                       tensor_local=T.tolist(),tensor_astropy=Tgc.tolist(),
                       basis='Fixed local outward/prograde/north; symmetric rank-2 tensor transforms B^T T B even though det(B)=-1.',
                       potential='galpy 1.12.0 MWPotential2014; ro=8 kpc, vo=220 km/s',
                       distinction_from06='Project06 Jacobi denominator is an effective midplane circular approximation. It is not an eigenvalue of this actual-position inertial tensor.')
    save(tensor,DATA/'stock2_tidal_tensor_axes.ecsv')
    rows=[]; major_rows=[]
    for group in groups(geometry):
        br=boot[boot['population']==group]; bv=np.column_stack([br[q] for q in ['e_radial','e_prograde','e_vertical']])
        for method in ['covariance','spatial_sign']:
            fits=shapes[(shapes['population']==group)&(shapes['method']==method)]
            for fitrow in fits:
                axis=np.array([fitrow[q] for q in ['e_radial','e_prograde','e_vertical']])
                for i in range(3):
                    rows.append((group,method,int(fitrow['component']),i+1,float(acute(axis,V[:,i]))))
                    if method=='covariance' and fitrow['component']==1:
                        percent=np.percentile(acute(bv,V[:,i]),[2.5,16,50,84,97.5])
                        major_rows.append((group,i+1,float(acute(axis,V[:,i])),*percent))
    angles=Table(rows=rows,names=['population','method','spatial_component','tidal_component','angle_deg']); angles['angle_deg'].unit=u.deg
    angles.meta=metadata('07D axial comparisons',[AXES,DATA/'stock2_tidal_tensor_axes.ecsv'])
    save(angles,DATA/'stock2_tidal_axis_angles.ecsv')
    confidence=Table(rows=major_rows,names=['population','tidal_component','angle_deg','bootstrap_p025_deg','bootstrap_p16_deg','bootstrap_p50_deg','bootstrap_p84_deg','bootstrap_p975_deg'])
    for q in confidence.colnames[2:]: confidence[q].unit=u.deg
    confidence.meta=metadata('07D bootstrap alignment',[BOOT,DATA/'stock2_tidal_tensor_axes.ecsv'])
    save(confidence,DATA/'stock2_tidal_alignment_bootstrap.ecsv')
    write_json(RESULTS/'tidal_tensor/metadata.json',tensor.meta)
    orientation_plot(geometry,shapes,V)
    report(geometry,shapes,tensor,confidence)
    protect(); print('07D PASS; tidal eigenvalues:',w)


def orientation_plot(geometry,shapes,V):
    plt=plotting(); fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    pts=xyz(geometry); mask=groups(geometry)['outside_rJ']; dirs=directions(read(REFERENCE))
    fits=shapes[(shapes['population']=='outside_rJ')&(shapes['component']==1)]
    vectors=[]
    for method,color,ls,label in [('covariance','black','-','Outer standard major axis'),('spatial_sign','#8172b2','--','Outer spatial-sign axis')]:
        row=fits[fits['method']==method][0]
        vectors.append((np.array([row[q] for q in ['e_radial','e_prograde','e_vertical']]),color,ls,label,False))
    vectors += [(dirs['radial_outward'],'#c44e52',':','Galactic radial axis',False),
                (dirs['orbital_velocity'],'#dd8452','-','Instantaneous velocity',True)]
    for i,color in enumerate(['#55a868','#4c72b0','#937860']): vectors.append((V[:,i],color,'-.',f'Tidal principal axis {i+1}',False))
    labels=['Radial [pc]','Prograde [pc]','Vertical [pc]']
    for ax,(i,j) in zip(axes,[(0,1),(0,2),(1,2)]):
        ax.scatter(pts[mask,i],pts[mask,j],s=5,color='lightgrey',rasterized=True)
        for v,color,ls,label,directed in vectors:
            v=v*65
            if np.hypot(v[i],v[j])<.05: continue
            if directed:
                ax.annotate('',xy=(v[i],v[j]),xytext=(0,0),arrowprops=dict(arrowstyle='->',color=color,lw=1.6))
                ax.plot([],[],color=color,label=label)
            else: ax.plot([-v[i],v[i]],[-v[j],v[j]],color=color,ls=ls,lw=1.7,label=label)
        ax.scatter(0,0,marker='+',s=40,color='black'); ax.set(xlabel=labels[i],ylabel=labels[j]); ax.set_aspect('equal',adjustable='box')
    # One global legend with all vectors even if a projection vanishes.
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],color=color,ls=ls,label=label) for v,color,ls,label,directed in vectors]
    fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False,fontsize=9)
    fig.suptitle('Stock 2 — r ≥ r_J morphology and local directions\nAxes have arbitrary signs; 65 pc display length has no dynamical meaning',fontsize=12)
    savefig(fig,RESULTS/'directional_geometry','principal_axis_orientation'); plt.close(fig)


def report(geometry,shapes,tensor,confidence):
    summary=json.loads((RESULTS/'morphology/shape_summary.json').read_text())
    angles=read(DATA/'stock2_directional_angles.ecsv'); counts=read(DATA/'stock2_geometric_side_counts.ecsv')
    lines=['Project 07 — Cluster-centric Morphology & Tidal Geometry','='*65,FRAME,
           'Frozen 03C→04A reference; no parallax recalculation. All canonical 1456 sources retained.',
           f'Inherited r_J={geometry.meta["r_J_pc"]:.9f} pc; zones descriptive only.',
           'Standard PCA: mean-centred covariance, ddof=1, all positions unweighted; axis scales are RMS widths.',
           'Spatial-sign scatter: component-median centring and unit displacement vectors, no clipping; dimensionless shape only.',
           'Group means/medians do not replace the frozen frame centre.',
           f'Bootstrap: {N_BOOT} replicates per group; base seed {SEED}; empirical axial sampling stability, no Gaia uncertainty propagation.',
           'Angles are arccos(abs(dot)), 0–90 deg. True 3D GC direction and planar radial direction are distinguished.',
           '', 'Shape and leverage results:']
    for group,s in summary.items():
        f=s['standard']; b=s['bootstrap']
        lines += [f'{group} N={s["n"]}: RMS axes={f["scales"]} pc; b/a={f["b_a"]:.6f}, c/a={f["c_a"]:.6f}; major gap={f["major_gap"]:.5f}',
                  f'  major(local R,prograde,Z)={[row[0] for row in f["vectors"]]}',
                  f'  bootstrap deviation median/p68/p95={b["deviation_p50_deg"]:.3f}/{b["deviation_p68_deg"]:.3f}/{b["deviation_p95_deg"]:.3f} deg; robust axis shift={s["robust_axis_shift_deg"]:.3f} deg',
                  f'  max leave-one-out shift={s["leave_one_out_max_axis_shift_deg"]:.3f} deg; top {s["top_one_percent_n"]} objects contribute {s["top_one_percent_trace_fraction"]:.3%} of centred covariance trace.']
    lines += ['', 'Major-axis direction angles (point estimate; bootstrap 2.5–97.5 percentiles):']
    for row in angles:
        if row['direction'] in ['radial_outward','prograde','orbital_velocity','vertical','galactic_centre_toward']:
            lines.append(f'{row["population"]} -> {row["direction"]}: {row["standard_angle_deg"]:.3f} deg [{row["bootstrap_p025_deg"]:.3f}, {row["bootstrap_p975_deg"]:.3f}]; spatial-sign={row["spatial_sign_angle_deg"]:.3f}')
    lines += ['',tensor.meta['definition'],tensor.meta['distinction_from06'],'Tidal eigenpairs, local (R,prograde,Z), (km/s/kpc)^2:']
    for row in tensor: lines.append(f'{row["component"]}: {row["eigenvalue"]:.9f}; vector=({row["e_radial"]:.9f},{row["e_prograde"]:.9f},{row["e_vertical"]:.9f}); {row["interpretation"]}')
    lines += ['', 'Major morphology vs tidal axes (point estimate; bootstrap 2.5–97.5 percentiles):']
    for row in confidence:
        lines.append(f'{row["population"]} -> tidal {row["tidal_component"]}: {row["angle_deg"]:.3f} deg [{row["bootstrap_p025_deg"]:.3f}, {row["bootstrap_p975_deg"]:.3f}]')
    lines += ['', 'Geometric half-space counts (no tail membership inference):']
    for row in counts:
        if row['population'] in ['full','C','L','T']: lines.append(f'{row["population"]} {row["side"]}: {row["N_side"]}/{row["N_total"]}')
    lines += ['', 'Interpretation boundaries:',
              'Bootstrap dispersion measures conditional catalogue sampling, not statistical significance of an alignment or astrometric uncertainty.',
              'Spatial-sign orientation changes probe radial leverage and shape-definition sensitivity; they do not alone prove a small number of objects dominates.',
              'Leave-one-out influence and top-1% variance fractions quantify extreme-point leverage without deleting sources from any scientific product.',
              'A small angle must be interpreted with its bootstrap interval, eigenvalue gap, robust comparison and shell dependence.',
              'Morphology, geometric hemispheres and local tensor alignment do not confirm escaped/unbound stars, tidal tails, or a unique stripping mechanism.',
              'The local tensor is evaluated at one point; it is not assumed spatially constant over the entire extended population.',
              'Galactic potential, mass, reference, candidate positions and labels are inherited without error propagation or new membership cuts.',
              'NUMERICAL VALIDATION PASS; full tests and interpretation review recorded in verification.md.']
    (RESULTS/'project07_summary.txt').write_text('\n'.join(lines)+'\n')
    meta=metadata('07A–07D',[GC,REFERENCE,ZONES,JACOBI,GEOMETRY,AXES,BOOT,DATA/'stock2_tidal_tensor_axes.ecsv'])
    meta.update(frame_details=geometry.meta,bootstrap=dict(n=N_BOOT,base_seed=SEED,statistics='axial projector and acute-angle quantiles'),
                tidal_definition=tensor.meta,canonical_count=1456,shape_summary=summary)
    write_json(RESULTS/'project07_metadata.json',meta)


if __name__=='__main__': main()
