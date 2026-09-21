"""06A: actual local field plus consistent midplane Jacobi ingredients."""
from support import *


def main():
    before=upstream_snapshot()
    ref,orbit,spatial=validate_inputs()
    R=float(ref['R_gc_kpc'][0]); z=float(ref['z_gc_kpc'][0])
    table=field_table([field_at(R,z)])
    table.meta=metadata('06A',[REFERENCE,ORBIT,REF03])
    table=save(table,FIELD)
    check_hashes(before)
    write_json(RESULTS/'upstream_integrity.json',dict(status='PASS',sha256=before))
    write_json(RESULTS/'tidal_field/metadata.json',table.meta)
    print(table)


if __name__=='__main__': main()
