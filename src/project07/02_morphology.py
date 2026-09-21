"""07B: standard PCA, radial-leverage comparison, and axial bootstrap."""
from geometry_support import *


def main():
    protect(); table=read(GEOMETRY); points=xyz(table)
    rows=[]; bootrows=[]; summaries={}
    for gi,(group,mask) in enumerate(groups(table).items()):
        X=points[mask]; regular=fit(X); robust=fit(X,robust=True)
        if not regular['major_defined']: raise ValueError(f'Undefined major axis: {group}')
        for method,f in [('covariance',regular),('spatial_sign',robust)]:
            for i in range(3):
                row=dict(population=group,method=method,n=len(X),component=i+1,
                         eigenvalue=float(f['eigenvalues'][i]),scale=float(f['scales'][i]),
                         eigenvalue_unit='pc2' if method=='covariance' else 'dimensionless',
                         scale_unit='pc (one-sigma RMS)' if method=='covariance' else 'dimensionless',
                         e_radial=f['vectors'][0,i],e_prograde=f['vectors'][1,i],e_vertical=f['vectors'][2,i],
                         b_a=f['b_a'],c_a=f['c_a'],major_gap=f['major_gap'],major_defined=f['major_defined'],
                         centre_radial_pc=f['center'][0],centre_prograde_pc=f['center'][1],centre_vertical_pc=f['center'][2])
                for a in range(3):
                    for b in range(3): row[f'scatter_{a}{b}']=f['matrix'][a,b]
                rows.append(row)
        seed=SEED+gi
        samples,gaps=bootstrap(X,seed)
        axes=regular['vectors'][:,0]; stats=axial_summary(samples,axes)
        deviations=acute(samples,axes)
        for i,(v,gap,dev) in enumerate(zip(samples,gaps,deviations)):
            bootrows.append((group,i,seed,*v,gap,dev))
        # Influence audit only: refits omit one object at a time, but never alter
        # canonical/output populations. No selected leave-one-out fit is adopted.
        loo=np.array([acute(fit(np.delete(X,i,axis=0))['vectors'][:,0],axes) for i in range(len(X))])
        power=np.sum((X-X.mean(axis=0))**2,axis=1); n_top=max(1,int(np.ceil(.01*len(X))))
        top=np.argsort(power)[-n_top:]
        summaries[group]=dict(n=len(X),standard=regular,spatial_sign=robust,
                              robust_axis_shift_deg=float(acute(axes,robust['vectors'][:,0])),
                              bootstrap=stats,seed=seed,resamples=N_BOOT,
                              leave_one_out_max_axis_shift_deg=float(loo.max()),
                              max_influence_source_id=str(table['gaia_dr3_source_id'][mask][np.argmax(loo)]),
                              top_one_percent_n=n_top,top_one_percent_trace_fraction=float(power[top].sum()/power.sum()),
                              top_one_percent_source_ids=[str(v) for v in table['gaia_dr3_source_id'][mask][top]])
    axes=Table(rows=rows)
    for q in ['centre_radial_pc','centre_prograde_pc','centre_vertical_pc']: axes[q].unit=u.pc
    axes.meta=metadata('07B',[GEOMETRY])
    axes.meta.update(standard='Unweighted mean-centred sample covariance, ddof=1. Scales sqrt(eigenvalues) are RMS widths, not physical endpoints.',
                     robust='Coordinate-median-centred spatial-sign second moment: each nonzero centred vector normalized to unit length; zero vectors contribute zero. All rows retained. Dimensionless angular scatter, NOT a robust size estimate and not fully rotation-equivariant because of component medians.',
                     signs='Largest absolute component of each eigenvector made positive for display only; acute angles and vv^T statistics ignore signs.',
                     influence='Leave-one-out and top 1% covariance-trace diagnostics do not prune/redefine samples; no leave-one-out solution adopted.')
    save(axes,AXES)
    boot=Table(rows=bootrows,names=['population','replicate','seed','e_radial','e_prograde','e_vertical','major_gap','deviation_deg'])
    boot['deviation_deg'].unit=u.deg
    boot.meta=metadata('07B bootstrap',[GEOMETRY,AXES])
    boot.meta.update(resamples_per_population=N_BOOT,base_seed=SEED,
                     method='Resample each fixed descriptive population with replacement; refit standard mean-centred PCA. Axial mean from <vv^T>; deviations arccos(abs(dot)). Percentiles are empirical sampling stability, not Gaia uncertainty or formal alignment significance.')
    save(boot,BOOT)
    write_json(RESULTS/'morphology/shape_summary.json',summaries)
    make_plots(table,boot)
    protect()
    for group,s in summaries.items(): print(group,'b/a,c/a',s['standard']['b_a'],s['standard']['c_a'],'p95',s['bootstrap']['deviation_p95_deg'],'robust shift',s['robust_axis_shift_deg'])


def make_plots(table,boot):
    from matplotlib.patches import Circle
    plt=plotting(); points=xyz(table); rj=table.meta['r_J_pc']; dirs=directions(read(REFERENCE))
    colors={'C':'#4c72b0','L':'#dd8452','T':'#55a868'}
    labels=['Local outward radial [pc]','Local prograde tangential [pc]','Local vertical north [pc]']
    for i,j,name in [(0,1,'radial_tangential'),(0,2,'radial_vertical'),(1,2,'tangential_vertical')]:
        fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
        for cls,color in colors.items():
            mask=np.asarray(table['homogenized_class'])==cls
            ax.scatter(points[mask,i],points[mask,j],s=7,alpha=.6,color=color,label=f'{cls} (N={mask.sum()})',rasterized=True)
        for f,ls in [(1,'--'),(2,':')]: ax.add_patch(Circle((0,0),rj*f,fill=False,ls=ls,color='grey',label=f'{f if f==2 else ""}r_J scale guide'))
        ax.scatter(0,0,color='black',marker='+',s=60,label='Frozen centre')
        if name=='radial_tangential':
            for key,label,color in [('galactic_centre_toward','Toward Galactic centre','#c44e52'),('prograde','Prograde tangent','#8172b2'),('orbital_velocity','Instantaneous velocity','black')]:
                v=dirs[key]*55
                ax.annotate('',xy=(v[i],v[j]),xytext=(0,0),arrowprops=dict(arrowstyle='->',color=color,lw=1.5))
                ax.plot([],[],color=color,label=label)
        ax.set(xlabel=labels[i],ylabel=labels[j],title='Stock 2 — physical cluster-centric morphology')
        ax.set_aspect('equal',adjustable='box'); ax.legend(loc='upper left',bbox_to_anchor=(1,1),frameon=False,fontsize=9)
        ax.text(.5,-.16,'Scale circles do not define membership; arrows show projected directions.',transform=ax.transAxes,ha='center',fontsize=9)
        savefig(fig,RESULTS/'morphology',name); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
    ax.scatter(points[:,0],points[:,1],s=5,color='lightgrey',label='All literature candidates',rasterized=True)
    for group,color,label in [('intermediate','#dd8452','r_J ≤ r < 2r_J'),('outer','#55a868','r ≥ 2r_J')]:
        m=groups(table)[group]; ax.scatter(points[m,0],points[m,1],s=13,color=color,alpha=.8,label=f'{label} (N={m.sum()})',rasterized=True)
    for f,ls in [(1,'--'),(2,':')]: ax.add_patch(Circle((0,0),f*rj,fill=False,ls=ls,color='black'))
    ax.scatter(0,0,color='black',marker='+'); ax.set(xlabel=labels[0],ylabel=labels[1],title='Stock 2 — outer geometric zones (3D radius selection)')
    ax.set_aspect('equal',adjustable='box'); ax.legend(loc='upper left',bbox_to_anchor=(1,1),frameon=False)
    savefig(fig,RESULTS/'morphology','outer_population'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    for group,label,color in [('T','T','#55a868'),('outside_rJ','r ≥ r_J','#dd8452'),('outer','r ≥ 2r_J','#4c72b0')]:
        v=np.sort(np.asarray(boot['deviation_deg'][boot['population']==group]))
        ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',label=f'{label}: {len(v)} resamples',color=color)
    ax.set(xlabel='Acute major-axis deviation from original PCA [deg]',ylabel='Bootstrap cumulative fraction',ylim=(0,1.02),title='Stock 2 — axial orientation sampling stability')
    ax.legend(loc='lower right'); savefig(fig,RESULTS/'bootstrap','orientation_distribution'); plt.close(fig)


if __name__=='__main__': main()
