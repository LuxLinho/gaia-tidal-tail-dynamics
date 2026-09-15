"""Project 03D: descriptive spatial PCA and fixed-basis velocity diagnostics."""
from pathlib import Path
import hashlib
import importlib.util

import numpy as np
import scipy
from scipy import stats
from astropy import units as u
from astropy.table import Table

_spec = importlib.util.spec_from_file_location('project03c', Path(__file__).with_name('03_cluster_centered_reference_frame.py'))
c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c)
a, b, plt = c.a, c.b, c.plt
ROOT = c.ROOT
INPUT = ROOT/'data/interim/project03/stock2_cluster_relative_phase_space.ecsv'
REFERENCE = ROOT/'data/interim/project03/stock2_reference_frame.ecsv'
OUTPUT = ROOT/'data/interim/project03'
FIGURES = ROOT/'results/project03/relative_morphology'
S = ['s1_ext_pc', 's2_ext_pc', 's3_ext_pc']
U = ['u1_ext_kms', 'u2_ext_kms', 'u3_ext_kms']
UNITS = {**dict.fromkeys(S+['r_perp_ext_pc'], 'pc'), **dict.fromkeys(U+['u_perp_ext_kms'], 'km / s')}
SIGN_RULE = ('Eigenvalues descending. For e1 and e2, make the largest-absolute Cartesian '
             'component positive (ties use first index X,Y,Z). Set e3=cross(e1,e2) for a '
             'right-handed triad. Rows of basis are e1,e2,e3. Signs are deterministic; '
             'individual axes in exactly degenerate eigenspaces are not uniquely defined.')
PCA_METHOD = ('Sample covariance PCA: subtract each population coordinate mean, use covariance '
              'denominator N-1; no per-axis scaling, source removal, or clipping. '
              'SVD of centred positions; eigenvalues=SVD singular values squared/(N-1).')
PROJECTION = ('Only existing L+T positions define the fixed basis. s=basis @ Project03C_delta_r '
              'and u=basis @ Project03C_delta_v; do NOT subtract the L+T mean for canonical '
              'projection. Zero remains the stored Stock 2 reference. No velocity PCA.')
P_CAVEAT = ('Two-sided SciPy Pearson p-values use the normal uncorrelated null; Spearman '
            'p-values are asymptotic and may be inaccurate for small samples, especially N<=500. '
            'No multiple-comparison adjustment, measurement-error model, or treatment of shared '
            'reference uncertainty. Descriptive diagnostics, not a complete physical significance '
            'test or evidence of gravitational association; no classification uses p-values.')
SUMMARY_UNITS = {**UNITS, **dict.fromkeys(['abs_s1_ext_pc','abs_s2_ext_pc','abs_s3_ext_pc','r_cluster_pc'],'pc'),
                 **dict.fromkeys(['abs_u1_ext_kms','dv_cluster_kms']+c.DECOMPOSITION,'km / s')}
LABELS = {**c.LABELS,
          **dict(zip(S,[r'$s_1$ [pc]',r'$s_2$ [pc]',r'$s_3$ [pc]'])),
          **dict(zip(U,[r'$u_1$ [km/s]',r'$u_2$ [km/s]',r'$u_3$ [km/s]'])),
          'r_perp_ext_pc':r'$r_\perp$ [pc]', 'u_perp_ext_kms':r'$|u_\perp|$ [km/s]',
          'abs_s1_ext_pc':r'$|s_1|$ [pc]', 'abs_u1_ext_kms':r'$|u_1|$ [km/s]'}
PAIRS = [('s1_ext_pc','u1_ext_kms'),('abs_s1_ext_pc','abs_u1_ext_kms'),
         ('r_cluster_pc','dv_cluster_kms'),('r_perp_ext_pc','u_perp_ext_kms'),
         ('s1_ext_pc','dv_cluster_kms')]


def data(table, quantity):
    if quantity.startswith('abs_'):
        return np.abs(a.values(table[quantity[4:]]))
    return a.values(table[quantity])


def populations(table):
    labels = np.asarray(table['homogenized_class'])
    return {'full':np.ones(len(table),dtype=bool), **{g:labels==g for g in ['C','L','T']},
            'L+T':np.isin(labels,['L','T'])}


def preserve_columns(original, result):
    if len(original)!=len(result):
        raise ValueError('Canonical row count changed')
    for q in original.colnames:
        np.testing.assert_equal(original[q].data,result[q].data)
        np.testing.assert_array_equal(np.ma.getmaskarray(original[q]),np.ma.getmaskarray(result[q]))
        if original[q].unit!=result[q].unit:
            raise ValueError(f'Original unit changed: {q}')
    for key,value in original.meta.items():
        if key not in result.meta or result.meta[key]!=value:
            raise ValueError(f'Original metadata changed: {key}')


def validate_input(table, ref):
    """Verify stored 03C reference and arithmetic without recomputing its medians."""
    a.validate_invariants(table,Table.read(a.MASTER_PATH))
    previous = Table.read(c.INPUT)
    preserve_columns(previous,table)
    if len(ref)!=1 or ref['reference_population'][0]!='C' or ref['reference_statistic'][0]!='component-wise median':
        raise ValueError('Unexpected stored reference definition')
    for key,refkey in [('project03c_frame','coordinate_convention'),('project03c_reference','reference_definition')]:
        if table.meta[key]!=ref.meta[refkey]:
            raise ValueError(f'Reference metadata mismatch: {key}')
    if table.meta['project03c_reference_product']!=REFERENCE.name:
        raise ValueError('Reference product mismatch')
    for q in c.REF_POSITION+c.REF_VELOCITY:
        if table.meta[q]!=ref[q][0]:
            raise ValueError(f'Reference value mismatch: {q}')
    for original,derived,refs,n,unit in [(b.POSITION,c.POSITION,c.REF_POSITION,1456,u.pc),
                                       (b.VELOCITY,c.VELOCITY,c.REF_VELOCITY,886,u.km/u.s)]:
        array,valid = c.vectors(table,original)
        offsets,available = c.vectors(table,derived)
        np.testing.assert_array_equal(available,valid)
        if available.sum()!=n:
            raise ValueError(f'Unexpected availability: {derived}')
        for q in derived:
            if table[q].unit!=unit:
                raise ValueError(f'Unexpected unit: {q}')
            np.testing.assert_array_equal(np.isfinite(a.values(table[q])),available)
        for q in refs:
            if ref[q].unit!=unit:
                raise ValueError(f'Unexpected reference unit: {q}')
        np.testing.assert_allclose(offsets[:,valid],array[:,valid]-np.array([ref[q][0] for q in refs])[:,None],atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(a.values(table[derived[3]])[valid],np.linalg.norm(offsets[:,valid],axis=0),atol=1e-12,rtol=1e-12)
    xyz,pos = c.vectors(table,c.POSITION)
    velocity,vel = c.vectors(table,c.VELOCITY)
    np.testing.assert_array_equal(vel,np.isfinite(a.values(table['radial_velocity'])))
    cmask=np.asarray(table['homogenized_class'])=='C'
    if [ref[q][0] for q in ['n_c_total','n_position_contributors','n_velocity_contributors']] != [int(cmask.sum()),int((cmask&pos).sum()),int((cmask&vel).sum())]:
        raise ValueError('Reference contributor count mismatch')
    # Reuse stored radial/tangential values; only validate their semantics.
    radius=data(table,'r_cluster_pc')
    valid=pos&vel&(radius>0)
    for q in c.DECOMPOSITION:
        if table[q].unit!=u.km/u.s:
            raise ValueError(f'Unexpected unit: {q}')
        np.testing.assert_array_equal(np.isfinite(data(table,q)),valid)
    radial=data(table,c.DECOMPOSITION[0])[valid]
    tangential=data(table,c.DECOMPOSITION[1])[valid]
    np.testing.assert_allclose(radial,np.sum(velocity[:,valid]*xyz[:,valid]/radius[valid],axis=0),atol=1e-12)
    np.testing.assert_allclose(radial**2+tangential**2,data(table,'dv_cluster_kms')[valid]**2,atol=1e-12,rtol=1e-12)


def orient_axes(axes):
    result=np.array(axes,dtype=float,copy=True)
    for i in (0,1):
        if result[i,np.argmax(np.abs(result[i]))]<0:
            result[i]*=-1
    result[2]=np.cross(result[0],result[1])
    return result


def spatial_pca(xyz):
    """N x 3 positions in pc. All input vectors are used, without standardization."""
    xyz=np.asarray(xyz,dtype=float)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)<3 or not np.isfinite(xyz).all():
        raise ValueError('PCA requires at least three finite 3D position vectors')
    mean=xyz.mean(axis=0)
    _,singular,axes=np.linalg.svd(xyz-mean,full_matrices=False)
    eigenvalues=singular**2/(len(xyz)-1)
    if eigenvalues.sum()==0:
        raise ValueError('PCA orientation undefined for zero spatial variance')
    axes=orient_axes(axes)
    lengths=np.sqrt(eigenvalues)
    ratios=np.array([lengths[i]/lengths[j] if lengths[j]>0 else np.inf for i,j in [(0,1),(1,2),(0,2)]])
    return dict(n=len(xyz),mean=mean,eigenvalues=eigenvalues,variance_ratios=eigenvalues/eigenvalues.sum(),
                axes=axes,lengths=lengths,axis_ratios=ratios)


def pca_populations(table):
    xyz,_=c.vectors(table,c.POSITION)
    fits={group:spatial_pca(xyz[:,mask].T) for group,mask in populations(table).items()}
    rows=[]
    for group,p in fits.items():
        for i in range(3):
            rows.append(dict(population=group,n=p['n'],component=i+1,
                mean_dx_pc=p['mean'][0],mean_dy_pc=p['mean'][1],mean_dz_pc=p['mean'][2],
                eigenvalue_pc2=p['eigenvalues'][i],explained_variance_ratio=p['variance_ratios'][i],
                axis_length_pc=p['lengths'][i],e_x=p['axes'][i,0],e_y=p['axes'][i,1],e_z=p['axes'][i,2],
                axis_ratio_12=p['axis_ratios'][0],axis_ratio_23=p['axis_ratios'][1],axis_ratio_13=p['axis_ratios'][2],
                canonical_projection_basis=(group=='L+T')))
    output=Table(rows=rows)
    for q in ['mean_dx_pc','mean_dy_pc','mean_dz_pc','axis_length_pc']:
        output[q].unit=u.pc
    output['eigenvalue_pc2'].unit=u.pc**2
    output.meta.update(method=PCA_METHOD,sign_convention=SIGN_RULE,projection=PROJECTION,
                       basis='L+T rows, component ascending; e_x,e_y,e_z are components in original 03C axes')
    return fits,output


def project(table,basis):
    np.testing.assert_allclose(basis@basis.T,np.eye(3),atol=1e-12)
    result=table.copy(copy_data=True)
    for q,unit in UNITS.items():
        if q in result.colnames:
            raise ValueError(f'Derived column already exists: {q}')
        result[q]=np.full(len(table),np.nan)*u.Unit(unit)
    for original,derived,norm in [(c.POSITION,S,'r_perp_ext_pc'),(c.VELOCITY,U,'u_perp_ext_kms')]:
        array,valid=c.vectors(table,original)
        projected=basis@array[:,valid]
        for q,values in zip(derived,projected):
            result[q][valid]=values
        result[norm][valid]=np.hypot(projected[1],projected[2])
    result.meta.update(project03d_projection=PROJECTION,project03d_sign_convention=SIGN_RULE,
                       project03d_pca_product='stock2_morphology_pca.ecsv',project03d_basis=basis.tolist(),
                       project03d_reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
                       project03d_input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest())
    return result


def validate_projection(original,result,basis):
    preserve_columns(original,result)
    if set(result.colnames)-set(original.colnames)!=set(UNITS):
        raise ValueError('Unexpected new columns')
    np.testing.assert_allclose(basis@basis.T,np.eye(3),atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(basis),1.,atol=1e-12)
    for i in (0,1):
        if basis[i,np.argmax(np.abs(basis[i]))]<0:
            raise ValueError('Noncanonical eigenvector sign')
    for original_fields,derived,norm in [(c.POSITION,S,'r_perp_ext_pc'),(c.VELOCITY,U,'u_perp_ext_kms')]:
        array,valid=c.vectors(original,original_fields)
        projections=np.array([data(result,q) for q in derived])
        for q in derived+[norm]:
            np.testing.assert_array_equal(np.isfinite(data(result,q)),valid)
            if result[q].unit!=u.Unit(UNITS[q]):
                raise ValueError(f'Wrong unit: {q}')
        np.testing.assert_allclose(projections[:,valid],basis@array[:,valid],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(basis.T@projections[:,valid],array[:,valid],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(data(result,norm)[valid],np.hypot(projections[1,valid],projections[2,valid]),rtol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(projections[:,valid],axis=0),np.linalg.norm(array[:,valid],axis=0),rtol=1e-12)


def summaries(table):
    def rows(mask,field,group):
        return [dict(group_column=field,group=group,n_total=int(mask.sum()),
                     n_velocity=int(np.isfinite(data(table,'u1_ext_kms')[mask]).sum()),
                     quantity=q,unit=unit,**a.percentile_summary(data(table,q)[mask])) for q,unit in SUMMARY_UNITS.items()]
    overall=Table(rows=rows(np.ones(len(table),dtype=bool),'all','all'))
    groups=Table(rows=[r for field,labels in a.GROUPS.items() for group in labels
                      for r in rows(np.asarray(table[field])==group,field,group)])
    for t in (overall,groups):
        t.meta.update(statistics='Linear percentiles of all finite values; no clipping. Units per row.',projection=PROJECTION)
    return overall,groups


def side_and_radial_summaries(table):
    side_rows,radial_rows=[],[]
    for field,labels in a.GROUPS.items():
        for group in labels:
            mask=np.asarray(table[field])==group
            s=data(table,'s1_ext_pc')[mask]
            for side,selection in [('negative',s<0),('positive',s>0),('zero',s==0)]:
                v=np.abs(s[selection])
                side_rows.append(dict(group_column=field,group=group,side=side,n=int(selection.sum()),
                                      median_abs_s1_pc=float(np.median(v)) if len(v) else np.nan,
                                      max_abs_s1_pc=float(np.max(v)) if len(v) else np.nan))
            radial=data(table,c.DECOMPOSITION[0])[mask]
            finite=np.isfinite(radial)
            radial=radial[finite]
            n=len(radial)
            radial_rows.append(dict(group_column=field,group=group,n_total=int(mask.sum()),n_radial=n,
                n_missing_radial=int(mask.sum())-n,n_outward=int((radial>0).sum()),n_inward=int((radial<0).sum()),
                n_zero=int((radial==0).sum()),fraction_outward=float((radial>0).mean()) if n else np.nan,
                fraction_inward=float((radial<0).mean()) if n else np.nan,
                median_radial_kms=float(np.median(radial)) if n else np.nan,
                p16_radial_kms=float(np.percentile(radial,16)) if n else np.nan,
                p84_radial_kms=float(np.percentile(radial,84)) if n else np.nan,
                median_tangential_kms=a.percentile_summary(data(table,c.DECOMPOSITION[1])[mask])['median']))
    sides,radial=Table(rows=side_rows),Table(rows=radial_rows)
    for q in ['median_abs_s1_pc','max_abs_s1_pc']: sides[q].unit=u.pc
    for q in ['median_radial_kms','p16_radial_kms','p84_radial_kms','median_tangential_kms']: radial[q].unit=u.km/u.s
    sides.meta['definition']='Signs of s1 relative to fixed 03C origin; zero reported separately, no threshold.'
    radial.meta['definition']='Reused Project 03C spatial radial/tangential components. Fractions use n_radial as denominator. Positive means outward, not escaping/unbound.'
    return sides,radial


def correlation(x,y):
    x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    valid=np.isfinite(x)&np.isfinite(y)
    x,y=x[valid],y[valid]
    n=len(x)
    if n<3 or np.ptp(x)==0 or np.ptp(y)==0:
        return dict(n=n,pearson_r=np.nan,pearson_p_two_sided=np.nan,spearman_rho=np.nan,
                    spearman_p_two_sided=np.nan,status='undefined: fewer than 3 pairs or constant input')
    p=stats.pearsonr(x,y,alternative='two-sided')
    s=stats.spearmanr(x,y,alternative='two-sided')
    return dict(n=n,pearson_r=float(p.statistic),pearson_p_two_sided=float(p.pvalue),
                spearman_rho=float(s.statistic),spearman_p_two_sided=float(s.pvalue),status='ok')


def correlations(table):
    output=Table(rows=[dict(population=group,x_quantity=x,y_quantity=y,
                           **correlation(data(table,x)[mask],data(table,y)[mask]))
                      for group,mask in populations(table).items() for x,y in PAIRS])
    output.meta.update(interpretation=P_CAVEAT,scipy_version=scipy.__version__,
                       missing='Only paired finite values; no clipping. Undefined correlations and p-values are NaN.')
    return output


def scatter(table,x,y,field):
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    for color,group in zip(a.COLORS,a.GROUPS[field]):
        group_mask=np.asarray(table[field])==group
        valid=group_mask&np.isfinite(data(table,x))&np.isfinite(data(table,y))
        ax.scatter(data(table,x)[valid],data(table,y)[valid],s=12,alpha=.45,color=color,
                   edgecolors='none',label=f'{group} (N={valid.sum()}/{group_mask.sum()})')
    ax.set(xlabel=LABELS[x],ylabel=LABELS[y],title='Stock 2 — fixed literature L+T spatial PCA basis')
    if x in S and y in S:
        ax.set_aspect('equal',adjustable='box')
    ax.legend(fontsize=8,loc='upper left',bbox_to_anchor=(1.02,1))
    ax.grid(alpha=.18)
    return fig,ax


def ecdf(table,q):
    fig,ax=plt.subplots(figsize=(7,5),layout='constrained')
    for color,group in zip(a.COLORS,a.GROUPS['homogenized_class']):
        mask=np.asarray(table['homogenized_class'])==group
        values=data(table,q)[mask]
        values=np.sort(values[np.isfinite(values)])
        if len(values):
            ax.step(np.r_[values[0],values],np.arange(len(values)+1)/len(values),where='post',
                    color=color,label=f'{group} (valid N={len(values)}/{mask.sum()})')
    ax.set(xlabel=LABELS[q],ylabel='Cumulative fraction within each class',ylim=(0,1.02),
           title='Stock 2 — empirical cumulative distributions')
    ax.legend(fontsize=8,loc='lower right')
    ax.grid(alpha=.18)
    return fig,ax


def axis_visualization(table,basis):
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    xyz,_=c.vectors(table,c.POSITION)
    length=float(np.max(np.abs(basis[0]@xyz)))
    ends=np.outer(basis[0],[-length,length])
    extent=max(float(np.abs(xyz).max()),float(np.abs(ends).max()))*1.05
    for ax,(i,j) in zip(axes,[(0,1),(0,2),(1,2)]):
        c.draw_scatter(ax,table,c.POSITION[i],c.POSITION[j],'homogenized_class')
        ax.plot(ends[i],ends[j],color='black',linestyle='--',linewidth=1.3,
                label='major spatial principal axis of the literature-defined L+T population')
        ax.set(xlim=(-extent,extent),ylim=(-extent,extent))
    axes[1].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.18),ncol=1)
    fig.suptitle('Stock 2 — geometric orientation in original Project 03C coordinates')
    return fig,axes


def generate_plots(table,basis,directory):
    outputs=[]
    def save(fig,stem):
        for ext in ('png','pdf'):
            p=directory/f'{stem}.{ext}'
            fig.savefig(p,dpi=200)
            outputs.append(p)
        plt.close(fig)
    specs=[(S[i],S[j],'homogenized_class') for i,j in [(0,1),(0,2),(1,2)]]
    specs += [('s1_ext_pc','r_perp_ext_pc','homogenized_class')]
    specs += [(x,y,'homogenized_class') for x,y in PAIRS[:4]]
    specs += [('s1_ext_pc','r_perp_ext_pc','catalogue_status')]
    for x,y,field in specs:
        fig,_=scatter(table,x,y,field)
        save(fig,f'{x}_vs_{y}_by_{field}')
    for q in ['s1_ext_pc','r_perp_ext_pc','u1_ext_kms']:
        fig,_=ecdf(table,q)
        save(fig,f'{q}_ecdf_by_homogenized_class')
    fig,_=axis_visualization(table,basis)
    save(fig,'extended_population_pca_axis_original_coordinates')
    return outputs


def protected_hashes():
    hashes=c.protected_hashes()
    paths=list((ROOT/'results/project03/cluster_relative_phase_space').rglob('*'))
    paths += list((ROOT/'data/interim/project03').glob('stock2_cluster_relative*.ecsv'))
    paths += [REFERENCE,Path(c.__file__),ROOT/'tests/test_project03_cluster_reference.py',ROOT/'docs/project03_cluster_reference_frame.md']
    for p in paths:
        if p.is_file(): hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return hashes


def report_text(fits,overall,groups,sides,radial,corr,outputs):
    lines=['Project 03D — Cluster-relative morphology and phase-space coherence','='*72,
           'Canonical rows/unique source IDs: 1456/1456; C/L/T: 940/184/332.',
           'Valid positions: 1456; valid velocities: 886; missing velocities: 570.',
           'Project 03C reference read unchanged; no reference median was recomputed.',
           'C/L/T labels are inherited from literature reconstruction. No membership inference.',
           'L+T PCA describes existing extended-population geometry; PCA does not establish a tidal origin.',
           'Correlations do not establish gravitational association.',PCA_METHOD,SIGN_RULE,PROJECTION,
           '', 'Spatial morphology:']
    for group,p in fits.items():
        lines += [f'{group}: N={p["n"]}; eigenvalues [pc^2]={p["eigenvalues"].tolist()}',
                  f'  explained variance ratios={p["variance_ratios"].tolist()}; axis lengths [pc]={p["lengths"].tolist()}',
                  f'  axis ratios 1/2, 2/3, 1/3={p["axis_ratios"].tolist()}; PCA mean [pc]={p["mean"].tolist()}',
                  f'  e1={p["axes"][0].tolist()}; e2={p["axes"][1].tolist()}; e3={p["axes"][2].tolist()}']
    lines+=['','Fixed L+T basis:']+[f'e{i+1}_ext={axis.tolist()}' for i,axis in enumerate(fits['L+T']['axes'])]
    lines+=['','Key C/L/T projected structure and velocities:']
    for group in ['C','L','T']:
        selected=groups[(groups['group_column']=='homogenized_class')&(groups['group']==group)]
        medians={row['quantity']:row['median'] for row in selected}
        lines.append(f'{group}: median |s1|={medians["abs_s1_ext_pc"]:.6f} pc; '
                     f'median r_perp={medians["r_perp_ext_pc"]:.6f} pc; '
                     f'median relative speed={medians["dv_cluster_kms"]:.6f} km/s; '
                     f'median |u1|={medians["abs_u1_ext_kms"]:.6f} km/s; '
                     f'median transverse speed={medians["u_perp_ext_kms"]:.6f} km/s')
    lines+=['','Key s1-u1 correlations:']
    for row in corr[(corr['x_quantity']=='s1_ext_pc')&(corr['y_quantity']=='u1_ext_kms')]:
        lines.append(f'{row["population"]}: N={row["n"]}; Pearson r={row["pearson_r"]:.6f}; Spearman rho={row["spearman_rho"]:.6f}')
    lines+=['','Key C/L/T radial behavior (fractions of finite radial measurements):']
    for row in radial[radial['group_column']=='homogenized_class']:
        lines.append(f'{row["group"]}: median radial={row["median_radial_kms"]:.6f} km/s; '
                     f'outward={row["fraction_outward"]:.6f}; inward={row["fraction_inward"]:.6f}; N={row["n_radial"]}')
    lines+=['','Overall and class/provenance summaries:']
    for t in (overall,groups):
        for r in t:
            summary=', '.join(f'{q}={r[q]:.6g}' for q in a.PERCENTILES)
            lines.append(f'{r["group_column"]}/{r["group"]} N={r["n_total"]}, N velocity={r["n_velocity"]}: '
                         f'{r["quantity"]} [{r["unit"]}], valid N={r["n_valid"]}: {summary}')
    lines+=['','Positive/negative-side morphology:']
    for r in sides:
        lines.append(f'{r["group_column"]}/{r["group"]} {r["side"]}: N={r["n"]}; median |s1|={r["median_abs_s1_pc"]:.6g} pc; max |s1|={r["max_abs_s1_pc"]:.6g} pc')
    lines+=['','Correlations:',P_CAVEAT]
    for r in corr:
        lines.append(f'{r["population"]}: {r["x_quantity"]} vs {r["y_quantity"]}; N={r["n"]}; '
                     f'Pearson r={r["pearson_r"]:.6g}, p={r["pearson_p_two_sided"]:.6g}; '
                     f'Spearman rho={r["spearman_rho"]:.6g}, p={r["spearman_p_two_sided"]:.6g}')
    lines+=['','Radial behavior (reused 03C; positive outward does not imply escaping or unbound):']
    for r in radial:
        lines.append(f'{r["group_column"]}/{r["group"]}: N radial={r["n_radial"]}; '
                     f'median radial={r["median_radial_kms"]:.6g} km/s; p16/p84={r["p16_radial_kms"]:.6g}/{r["p84_radial_kms"]:.6g}; '
                     f'outward fraction={r["fraction_outward"]:.6g}; inward fraction={r["fraction_inward"]:.6g}; zero N={r["n_zero"]}; '
                     f'median tangential={r["median_tangential_kms"]:.6g} km/s')
    lines+=['','Outputs:']+[str(p.relative_to(ROOT)) for p in outputs]
    lines+=['','Invariants:','rows == 1456: True','unique source_id == 1456: True',
            'C == 940: True','L == 184: True','T == 332: True','position count == 1456: True','velocity count == 886: True',
            'missing-RV rows retain NaN projected velocities: True','Project 03C reference frame unchanged: True',
            'original columns and labels unchanged: True','Project 03A/03B/03C products unchanged: True',
            'no source removed: True','no clipping: True','no new membership field: True','no membership threshold: True',
            'no Galactocentric transformation: True','no orbit integration: True','no actions: True','no potential model: True',
            'PROJECT 03D STATUS: PASS']
    return '\n'.join(lines)+'\n'


def main():
    before=protected_hashes()
    table,ref=Table.read(INPUT),Table.read(REFERENCE)
    validate_input(table,ref)
    fits,pca=pca_populations(table)
    basis=fits['L+T']['axes']
    result=project(table,basis)
    validate_projection(table,result,basis)
    overall,groups=summaries(result)
    sides,radial=side_and_radial_summaries(result)
    corr=correlations(result)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    FIGURES.mkdir(parents=True,exist_ok=True)
    products=[('stock2_relative_morphology',result),('stock2_morphology_pca',pca),
              ('stock2_morphology_summary',overall),('stock2_morphology_group_summary',groups),
              ('stock2_morphology_side_summary',sides),('stock2_morphology_radial_summary',radial),
              ('stock2_morphology_correlations',corr)]
    outputs=[]
    for stem,t in products:
        path=OUTPUT/f'{stem}.ecsv'
        t.write(path,overwrite=True)
        outputs.append(path)
    stored_pca=Table.read(outputs[1])
    ext=stored_pca[stored_pca['population']=='L+T']
    saved_basis=np.array([ext[q] for q in ['e_x','e_y','e_z']]).T
    np.testing.assert_array_equal(saved_basis,basis)
    validate_projection(table,Table.read(outputs[0]),saved_basis)
    outputs+=generate_plots(result,basis,FIGURES)
    if protected_hashes()!=before:
        raise RuntimeError('Protected earlier products changed')
    report_path=FIGURES/'relative_morphology_report.txt'
    outputs.append(report_path)
    report=report_text(fits,overall,groups,sides,radial,corr,outputs)
    report_path.write_text(report,encoding='utf-8')
    print(report)
    return result,pca


if __name__=='__main__':
    main()
