"""06C: descriptive 3D geometric zones, never membership or escape classes."""
from support import *

ZONE_LABELS=('r < r_J','r_J <= r < 2 r_J','r >= 2 r_J')


def zones(radius_pc,rj_pc):
    if not np.isfinite(rj_pc) or rj_pc<=0: raise ValueError('Invalid Jacobi radius')
    r=np.asarray(radius_pc,float)
    valid=np.isfinite(r)&(r>=0)
    zone=np.full(len(r),'unavailable',dtype='U24')
    zone[valid&(r<rj_pc)]=ZONE_LABELS[0]
    zone[valid&(r>=rj_pc)&(r<2*rj_pc)]=ZONE_LABELS[1]
    zone[valid&(r>=2*rj_pc)]=ZONE_LABELS[2]
    return zone,valid


def summarize(table):
    rows=[]
    groups=[('all','all',np.ones(len(table),bool))]
    for key in ('homogenized_class','catalogue_status'):
        groups += [(key,str(value),np.asarray(table[key])==value) for value in np.unique(table[key])]
    for column,label,mask in groups:
        available=mask & np.asarray(table['position_usable'])
        n=int(available.sum()); total=int(mask.sum())
        for zone in ZONE_LABELS:
            count=int(np.sum(available&(np.asarray(table['spatial_zone'])==zone)))
            rows.append((column,label,zone,total,n,total-n,count,count/n if n else np.nan))
    result=Table(rows=rows,names=['group_column','group','spatial_zone','N_total','N_calculable','N_unavailable','N_zone','fraction_of_calculable'])
    return result


def main():
    check_run_protection()
    ref,orbit,spatial=validate_inputs()
    rj=float(read(JACOBI)['r_J_now'][0])
    keys=['gaia_dr3_source_id','homogenized_class','catalogue_status','dx_cluster_pc','dy_cluster_pc','dz_cluster_pc','r_cluster_pc']
    output=spatial[keys].copy()
    xyz=np.column_stack([arr(output,q,u.pc) for q in keys[3:6]])
    computed=np.linalg.norm(xyz,axis=1)
    np.testing.assert_allclose(computed,arr(output,'r_cluster_pc',u.pc),atol=1e-9,rtol=1e-12,equal_nan=True)
    zone,valid=zones(computed,rj)
    output['position_usable']=valid
    output['r_over_r_J']=computed/rj*u.dimensionless_unscaled
    output['spatial_zone']=zone
    output.meta=metadata('06C',[SPATIAL,REF03,REFERENCE,JACOBI,MASTER])
    output.meta.update(radius_definition='Inherited full 3D barycentric Cartesian separation from frozen 03C centre; rotation into 04A does not change this norm.',
                       zones='Purely geometric descriptions, NOT membership/boundness/escape determinations.',
                       figure_projection='XY uses original 03B Galactic Cartesian axes in pc; projected circles are guides, while zones always use 3D radius.',
                       r_J_now_pc=rj,fraction_denominator='Calculable positions in each named group; missing positions retained as unavailable.')
    output=save(output,DATA/'stock2_candidate_tidal_zones.ecsv')
    summary=summarize(output); summary.meta=output.meta.copy()
    save(summary,DATA/'stock2_candidate_tidal_zone_counts.ecsv')
    plt=plotting()
    from matplotlib.patches import Circle
    colors={'C':'#4c72b0','L':'#dd8452','T':'#55a868'}
    folder=RESULTS/'candidate_extent'
    fig,ax=plt.subplots(figsize=(8,6.3),layout='constrained')
    for cls,color in colors.items():
        mask=np.asarray(output['homogenized_class'])==cls
        ax.scatter(xyz[mask,0],xyz[mask,1],s=8,alpha=.55,color=color,label=f'{cls} (N={mask.sum()})',rasterized=True)
    for factor,style in [(1,'--'),(2,':')]:
        ax.add_patch(Circle((0,0),factor*rj,fill=False,color='black',ls=style,lw=1.2,label=f'{factor if factor==2 else ""}r_J projection guide'))
    ax.scatter(0,0,marker='+',color='black',s=65,label='Frozen centre')
    ax.set(xlabel='Cluster-centric ΔX [pc]',ylabel='Cluster-centric ΔY [pc]',title='Stock 2 literature candidates — XY projection')
    ax.set_aspect('equal',adjustable='box'); ax.legend(loc='upper left',bbox_to_anchor=(1,1),frameon=False)
    ax.text(.5,-.15,'Dashed circles are scale guides, not membership boundaries.',transform=ax.transAxes,ha='center',fontsize=10)
    savefig(fig,folder,'candidate_xy'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
    ax.hist(computed[valid],bins=45,color='#4c72b0',alpha=.7,label=f'All calculable (N={valid.sum()})')
    for value,label,color,style in [(rj,'r_J now','black','--'),(2*rj,'2r_J now','black',':'),(MASS['comparison_tidal_radius_pc'],'Ye et al. 22.65 pc','#c44e52','-.')]:
        ax.axvline(value,color=color,ls=style,label=label)
    ax.set(xlabel='3D cluster-centric separation [pc]',ylabel='Source count',title='Stock 2 — geometric extent, not membership')
    ax.legend(loc='upper right'); savefig(fig,folder,'radial_distribution'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
    for cls,color in colors.items():
        v=np.sort(computed[valid&(np.asarray(output['homogenized_class'])==cls)])
        if len(v): ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',color=color,label=f'{cls} (N={len(v)})')
    for factor,style in [(1,'--'),(2,':')]: ax.axvline(factor*rj,color='black',ls=style,label=f'{factor if factor==2 else ""}r_J')
    ax.set(xlabel='3D cluster-centric separation [pc]',ylabel='Fraction within each literature class',ylim=(0,1.02),title='Stock 2 — C / L / T radial distributions')
    ax.legend(loc='lower right'); savefig(fig,folder,'class_radial_distribution'); plt.close(fig)
    check_run_protection(); print(summary)


if __name__=='__main__': main()
