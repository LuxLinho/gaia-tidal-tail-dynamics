"""07A: fixed orthonormal outward/prograde/north frame from existing GC data."""
from geometry_support import *


def build():
    ref,source,zones=read(REFERENCE),read(GC),read(ZONES)
    if len(source)!=1456 or len(zones)!=1456: raise ValueError('Canonical population changed')
    if len(set(source['gaia_dr3_source_id']))!=1456: raise ValueError('Duplicate canonical identities')
    for q in ['gaia_dr3_source_id','homogenized_class','catalogue_status']:
        np.testing.assert_array_equal(source[q],zones[q])
    for label,n in [('C',940),('L',184),('T',332)]:
        if np.sum(source['homogenized_class']==label)!=n: raise ValueError('Literature classes changed')
    if source.meta['project04a_parameters']!=ref.meta['adopted_parameters']: raise ValueError('Frame changed')
    if source.meta['project04a_reference_sha256']!=ref.meta['inputs']['data/interim/project03/stock2_reference_frame.ecsv']: raise ValueError('Centre provenance mismatch')
    B,pos,vel=basis(ref)
    positions=np.column_stack([arr(source,q,u.kpc) for q in ['x_gc_kpc','y_gc_kpc','z_gc_kpc']])
    local=(positions-pos)*1000@B
    radius=np.linalg.norm(local,axis=1)
    np.testing.assert_allclose(radius,arr(zones,'r_cluster_pc',u.pc),rtol=0,atol=1e-8)
    table=zones[['gaia_dr3_source_id','homogenized_class','catalogue_status','r_cluster_pc','spatial_zone']].copy()
    for key,v in zip(FIELDS,local.T): table[key]=v*u.pc
    table['reconstructed_radius_pc']=radius*u.pc
    table.meta=metadata('07A',[GC,REFERENCE,ZONES,JACOBI])
    table.meta.update(basis_columns_in_astropy=B.tolist(),basis_determinant=-1.,
                      reference_position_kpc=pos.tolist(),reference_velocity_kms=vel.tolist(),
                      r_J_pc=float(read(JACOBI)['r_J_now'][0]),max_radius_residual_pc=float(np.max(np.abs(radius-arr(zones,'r_cluster_pc',u.pc)))))
    return table


def main():
    before=snapshot(); table=save(build(),GEOMETRY)
    if not np.isfinite(xyz(table)).all(): raise ValueError('Invalid positions')
    check_hashes(before)
    write_json(RESULTS/'upstream_integrity.json',dict(status='PASS',sha256=before))
    write_json(RESULTS/'frame_metadata.json',table.meta)
    print('07A PASS; radius residual pc:',table.meta['max_radius_residual_pc'])


if __name__=='__main__': main()
