"""06D: fixed-mass circular-equivalent tidal scale along the stored 05 orbit."""
from support import *


def orbital_table(orbit):
    R=arr(orbit,'R_gc_kpc',u.kpc); z=arr(orbit,'z_gc_kpc',u.kpc)
    table=field_table([field_at(float(r),float(h)) for r,h in zip(R,z)])
    table['time_myr']=orbit['time_myr'].copy()
    table['r_J']=jacobi(MASS['value']*u.Msun,table['denominator'].quantity)
    table.meta=metadata('06D',[ORBIT,REFERENCE,JACOBI])
    table.meta['sampling']='All 4001 stored Project 05 points; no re-integration or subsampling.'
    return table


def main():
    check_run_protection(); ref,orbit,spatial=validate_inputs()
    table=orbital_table(orbit)
    table=save(table,DATA/'stock2_orbital_jacobi_radius.ecsv')
    t=arr(table,'time_myr'); rj=arr(table,'r_J',u.pc); R=arr(table,'R_kpc',u.kpc)
    i0=int(np.flatnonzero(t==0)[0]); peri=int(np.argmin(R)); apo=int(np.argmax(R))
    d=arr(table,'denominator')
    if not np.all(np.isfinite(rj)) or np.any(rj<=0): raise ValueError('Invalid orbital Jacobi scale')
    np.testing.assert_allclose(rj[i0],read(JACOBI)['r_J_now'][0],atol=1e-10,rtol=0)
    if np.argmin(rj)!=np.argmax(d): raise ValueError('Strongest denominator must give smallest radius')
    stats=dict(min_pc=float(rj.min()),max_pc=float(rj.max()),median_pc=float(np.median(rj)),now_pc=float(rj[i0]),
               peak_to_peak_over_now=float(np.ptp(rj)/rj[i0]),
               pericentre=dict(time_myr=float(t[peri]),R_kpc=float(R[peri]),r_J_pc=float(rj[peri])),
               apocentre=dict(time_myr=float(t[apo]),R_kpc=float(R[apo]),r_J_pc=float(rj[apo])),
               denominator_min=float(d.min()),denominator_max=float(d.max()),denominator_unit=str(DUNIT),
               strongest_tide_time_myr=float(t[np.argmax(d)]),strongest_tide_smallest_radius=True)
    plt=plotting(); folder=RESULTS/'orbital_variation'
    fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
    ax.plot(t,rj,color='#4c72b0',lw=1.1); ax.scatter(0,rj[i0],color='black',s=25,label='Present')
    ax.axvline(0,color='grey',lw=.7)
    ax.set(xlabel='Time from present [Myr]',ylabel='Circular-equivalent r_J [pc]',title='Stock 2 — fixed 4000 solar masses, midplane approximation')
    ax.legend(); savefig(fig,folder,'jacobi_time'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.8),layout='constrained')
    ax.plot(R,rj,color='#4c72b0',lw=1,label='Stored orbital samples')
    ax.scatter(R[i0],rj[i0],color='black',label='Present',s=25)
    ax.set(xlabel='Galactocentric cylindrical R [kpc]',ylabel='Circular-equivalent r_J [pc]',title='Stock 2 — radial variation of the tidal scale')
    ax.legend(); savefig(fig,folder,'jacobi_R'); plt.close(fig)
    meta=metadata('06A–06D',[REFERENCE,ORBIT,SPATIAL,REF03,MASTER,FIELD,JACOBI])
    meta.update(orbital_variation=stats,canonical_sources=1456,validation='PASS',
                G=dict(value=G_VALUE,unit='pc (km/s)^2 / solMass',source='astropy.constants.G'),
                integration_inherited=orbit.meta['integration'],frame_inherited=orbit.meta['adopted_parameters'],
                potential_inherited=orbit.meta['potential'],
                secondary_literature='Later Stock 2 studies/catalogues may use different tidal-radius definitions. No later numeric radius is adopted; none was independently verified for this project.')
    write_json(RESULTS/'project06_metadata.json',meta)
    write_json(folder/'orbital_variation_summary.json',stats)
    field=read(FIELD); present=read(JACOBI); grid=read(DATA/'stock2_jacobi_mass_sensitivity.ecsv'); counts=read(DATA/'stock2_candidate_tidal_zone_counts.ecsv')
    lines=['Project 06 — Galactic Tidal Field & Jacobi Radius','='*65,
           MODEL['formula'],MODEL['omega_definition'],MODEL['approximation'],MODEL['vertical_treatment'],
           f'Potential: {MODEL["name"]}, galpy 1.12.0, ro=8 kpc, vo=220 km/s.',
           f'Mass: approximately 4000 solMass, literature baseline; {MASS["reference"]}, DOI {MASS["doi"]}.',
           MASS['definition'],MASS['uncertainty_status'],MASS['provenance_status'],
           f'G={G_VALUE:.12g} pc (km/s)^2 / solMass','', 'Present field (actual vs midplane explicitly separated):']
    lines += [f'{q}: {field[q][0]:.12g} {field[q].unit}' for q in field.colnames]
    lines += ['', 'Present Jacobi comparison:']+[f'{q}: {present[q][0]:.12g} {present[q].unit}' for q in present.colnames]
    lines += ['', 'Mass sensitivity (NOT uncertainty interval), r_J proportional to M^(1/3):']
    lines += [f'{row["mass"]:.0f} solMass -> {row["r_J"]:.9f} pc' for row in grid]
    lines += ['', '3D geometric zones (fractions among calculable positions):']
    lines += [f'{row["group"]}: {row["spatial_zone"]}: {row["N_zone"]}/{row["N_calculable"]} = {row["fraction_of_calculable"]:.6f}; missing={row["N_unavailable"]}' for row in counts]
    lines += ['', 'Fixed-mass orbital variation:',json.dumps(stats,indent=2),'',
              'Strongest effective midplane tidal denominator corresponds to smallest r_J: PASS.',
              'All canonical 1456 sources retained, labels unchanged. Project 05 reused unchanged.',
              'No per-candidate orbit integration, membership cuts, mass-loss evolution, or potential robustness.',
              'Outside r_J does not establish unboundness, escape, tidal-tail membership, or disruption history.',
              'The 22.65 pc comparison is not a calibration target and is not necessarily statistically independent of the mass.',
              'The approximation neglects vertical modulation of r_J and noncircular time-dependent rotating-frame effects.',
              'The relatively small vertical heights and eccentricity motivate this diagnostic approximation, but do not make it exact.',
              'No mass uncertainty or candidate distance uncertainty is propagated; 3D radii inherit earlier parallax assumptions.',
              'PROJECT 06 NUMERICAL VALIDATION: PASS; see verification record for full test-suite status.']
    (RESULTS/'project06_summary.txt').write_text('\n'.join(lines)+'\n')
    check_run_protection(); print('\n'.join(lines))


if __name__=='__main__': main()
