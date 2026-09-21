"""07C: axial angles and descriptive orbital/radial half-spaces."""
from geometry_support import *


def main():
    protect(); table=read(GEOMETRY); points=xyz(table); ref=read(REFERENCE)
    dirs=directions(ref); B,pos,vel=basis(ref)
    rows=[]
    for key,v in dirs.items():
        g=B@v
        rows.append((key,*v,*g))
    vectors=Table(rows=rows,names=['direction','local_radial','local_prograde','local_vertical','astropy_X','astropy_Y','astropy_Z'])
    vectors.meta=metadata('07C',[REFERENCE,GEOMETRY])
    vectors.meta.update(gc_direction='True 3D direction to Galactic origin is -r_ref/|r_ref|; radial_inward is the in-plane -e_R. Both reported, not silently equated.',
                        tangent_vs_velocity_angle_deg=float(acute(dirs['prograde'],dirs['orbital_velocity'])))
    save(vectors,DATA/'stock2_direction_vectors.ecsv')
    shapes=read(AXES); boot=read(BOOT); angle_rows=[]
    for group in groups(table):
        standard=shapes[(shapes['population']==group)&(shapes['method']=='covariance')&(shapes['component']==1)][0]
        robust=shapes[(shapes['population']==group)&(shapes['method']=='spatial_sign')&(shapes['component']==1)][0]
        axis=np.array([standard[q] for q in ['e_radial','e_prograde','e_vertical']])
        robust_axis=np.array([robust[q] for q in ['e_radial','e_prograde','e_vertical']])
        br=boot[boot['population']==group]; bv=np.column_stack([br[q] for q in ['e_radial','e_prograde','e_vertical']])
        for key,v in dirs.items():
            distribution=acute(bv,v); percent=np.percentile(distribution,[2.5,16,50,84,97.5])
            angle_rows.append((group,key,float(acute(axis,v)),float(acute(robust_axis,v)),*percent))
    angles=Table(rows=angle_rows,names=['population','direction','standard_angle_deg','spatial_sign_angle_deg','bootstrap_p025_deg','bootstrap_p16_deg','bootstrap_p50_deg','bootstrap_p84_deg','bootstrap_p975_deg'])
    for q in angles.colnames[2:]: angles[q].unit=u.deg
    angles.meta=metadata('07C major-axis angles',[AXES,BOOT,REFERENCE])
    angles.meta['definition']='arccos(abs(dot(unit axes))) in [0,90] deg; bootstrap quantiles from standard PCA only, conditional on fixed catalogue and frame.'
    save(angles,DATA/'stock2_directional_angles.ecsv')
    orbital=points@dirs['orbital_velocity']; radial=points[:,0]
    output=table[['gaia_dr3_source_id','homogenized_class','catalogue_status','spatial_zone']].copy()
    output['orbital_projection_pc']=orbital*u.pc; output['radial_projection_pc']=radial*u.pc
    tolerance=1e-10
    output['orbital_geometric_side']=np.where(orbital>tolerance,'leading_side',np.where(orbital<-tolerance,'trailing_side','on_plane'))
    output['radial_geometric_side']=np.where(radial>tolerance,'outward_side',np.where(radial<-tolerance,'inward_side','on_plane'))
    output.meta=metadata('07C half-spaces',[GEOMETRY,REFERENCE])
    output.meta.update(definition='Sign of 3D delta-position dot instantaneous unit velocity defines leading/trailing-side geometry. Sign of delta dot e_R defines outward/inward cylindrical radial geometry. Not physical tail membership.',
                       on_plane_tolerance_pc=tolerance)
    save(output,DATA/'stock2_geometric_halfspaces.ecsv')
    counts=[]
    for group,mask in groups(table).items():
        for q in ['orbital_geometric_side','radial_geometric_side']:
            for side in (['leading_side','trailing_side','on_plane'] if q.startswith('orbital') else ['inward_side','outward_side','on_plane']):
                n=int(np.sum(mask&(np.asarray(output[q])==side)))
                counts.append((group,q,side,int(mask.sum()),n,n/mask.sum()))
    count_table=Table(rows=counts,names=['population','partition','side','N_total','N_side','fraction'])
    count_table.meta=output.meta.copy(); save(count_table,DATA/'stock2_geometric_side_counts.ecsv')
    write_json(RESULTS/'directional_geometry/metadata.json',vectors.meta)
    protect(); print('07C PASS; prograde/velocity separation:',vectors.meta['tangent_vs_velocity_angle_deg'])


if __name__=='__main__': main()
