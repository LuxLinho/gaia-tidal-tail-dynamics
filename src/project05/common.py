"""Shared IO and immutable upstream provenance for Project 05."""
from pathlib import Path
import hashlib
import importlib.util
import json
import platform
import astropy
import galpy
import numpy as np
import scipy
from astropy import units as u
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/interim/project05'
RESULTS = ROOT / 'results/project05'
REF03 = ROOT / 'data/interim/project03/stock2_reference_frame.ecsv'
REF04 = ROOT / 'data/interim/project04/stock2_galactocentric_reference.ecsv'
POPULATION = ROOT / 'data/interim/project04/stock2_galactocentric_phase_space.ecsv'
REFERENCE = DATA / 'stock2_cluster_reference.ecsv'
ORBIT = DATA / 'stock2_galactic_orbit.ecsv'
POSITION = ['x_gc_kpc', 'y_gc_kpc', 'z_gc_kpc']
VELOCITY = ['vx_gc_kms', 'vy_gc_kms', 'vz_gc_kms']
VERSIONS = {m.__name__: m.__version__ for m in (astropy, galpy, np, scipy)}
VERSIONS['python'] = platform.python_version()


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


p04a = module('project05_upstream04a', 'src/project04/01_galactocentric_context.py')
p04b = module('project05_upstream04b', 'src/project04/02_reference_orbit.py')


def read(path):
    return Table.read(path, format='ascii.ecsv')


def array(table, key, unit=None):
    return np.asarray(table[key].quantity.to_value(unit or table[key].unit), dtype=float)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(paths):
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def save(table, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write(path, format='ascii.ecsv', overwrite=True)
    return read(path)


def population_invariants():
    t = read(POPULATION)
    if len(t) != 1456 or len(set(t['gaia_dr3_source_id'])) != 1456:
        raise ValueError('Canonical population must contain 1456 unique sources')
    for key, expected in [('homogenized_class', {'C': 940, 'L': 184, 'T': 332}),
                          ('catalogue_status', {'shared': 885, 'kos_only': 178, 'risbud_only': 393})]:
        actual = {str(k): int(v) for k, v in zip(*np.unique(t[key], return_counts=True))}
        if actual != expected:
            raise ValueError(f'Population labels changed: {key}: {actual}')
    n6d = int(np.all(np.isfinite(np.column_stack([array(t, q) for q in VELOCITY])), axis=1).sum())
    if n6d != 886:
        raise ValueError('Expected 886 available 6D states, without redefining canonical sample')
    return {'canonical_unique_sources': 1456, 'full_6d_sources': n6d,
            'class_counts': {'C': 940, 'L': 184, 'T': 332},
            'catalogue_counts': {'shared': 885, 'kos_only': 178, 'risbud_only': 393}}


def protected_hashes():
    paths = []
    for directory in ('data', 'results', 'src', 'tests', 'docs'):
        for p in (ROOT / directory).rglob('*'):
            if p.is_file() and 'project05' not in str(p.relative_to(ROOT)) and '__pycache__' not in p.parts:
                paths.append(p)
    return provenance(paths)


def check_protected(before):
    for name, expected in before.items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'Upstream file changed: {name}')
