"""06B: literature-baseline Jacobi scale and explicit mass-sensitivity grid."""
from support import *


def main():
    check_run_protection(); validate_inputs()
    field=read(FIELD); d=field['denominator'].quantity[0]
    radius=jacobi(MASS['value']*u.Msun,d)
    ye=MASS['comparison_tidal_radius_pc']
    summary=Table()
    summary['mass']=[MASS['value']]*u.Msun
    summary['G']=[G_VALUE]*u.pc*(u.km/u.s)**2/u.Msun
    summary['denominator']=[d.value]*d.unit
    summary['r_J_now']=[radius.value]*u.pc
    summary['literature_tidal_radius']=[ye]*u.pc
    summary['signed_difference']=[radius.value-ye]*u.pc
    summary['absolute_difference']=[abs(radius.value-ye)]*u.pc
    summary['fractional_difference']=[(radius.value-ye)/ye]*u.dimensionless_unscaled
    summary.meta=metadata('06B',[FIELD,REFERENCE])
    save(summary,JACOBI)
    grid=Table()
    grid['mass']=MASS_GRID*u.Msun
    grid['r_J']=jacobi(grid['mass'].quantity,d)
    grid.meta=metadata('06B mass sensitivity',[FIELD])
    grid.meta['interpretation']='Sensitivity experiment only, NOT an uncertainty interval; fixed D implies r_J proportional to M^(1/3).'
    save(grid,DATA/'stock2_jacobi_mass_sensitivity.ecsv')
    plt=plotting(); fig,ax=plt.subplots(figsize=(7,4.8),layout='constrained')
    ax.plot(MASS_GRID,grid['r_J'],'o-',color='#4c72b0')
    ax.scatter([4000],[radius.value],color='black',zorder=4,label='Literature mass baseline')
    ax.set(xlabel='Cluster mass [solar masses]',ylabel='Circular-equivalent Jacobi scale [pc]',title='Stock 2 — mass sensitivity (not an uncertainty interval)')
    ax.legend(); savefig(fig,RESULTS/'jacobi_radius','mass_sensitivity'); plt.close(fig)
    check_run_protection()
    print(summary); print(grid)


if __name__=='__main__': main()
