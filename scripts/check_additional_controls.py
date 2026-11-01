#!/usr/bin/env python3
"""Run exact scalar/SG controls and the frozen generic-nonlinear kernel gate.

Use in a disposable reproduction copy: the original check writes its JSON
beside its source. No PDE campaign, installation or network is required.
"""
from __future__ import annotations
import json, math, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    qa = ROOT / 'qa'
    qa.mkdir(exist_ok=True)
    checks = []
    scripts = [
        ('exact_controls', 'regression/R1_R2/R1/check_exact_controls.py'),
        ('kernel_G1', 'legacy/src/kernel/test_kernel.py'),
        ('sg_initial_section', 'regression/JNM/check_sg_initial_section.py'),
    ]
    for name, relative in scripts:
        with (qa / f'{name}.log').open('w') as log:
            result = subprocess.run([sys.executable, str(ROOT / relative)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        checks.append({'check': name, 'returncode': result.returncode,
                       'log': f'{name}.log'})
        if result.returncode:
            raise SystemExit(f'STOP: {name}; inspect {qa / (name + ".log")}')
    data = json.loads((ROOT / 'regression/R1_R2/R1/exact_controls.json').read_text())
    if data['linear_identity'] != 'PASS' or data['sg_algebra'] != 'PASS':
        raise SystemExit('STOP: scalar or SG symbolic identity failed')
    for row in data['c0_exact_family']:
        numbers = [float(row[key]) for key in ['h', 'mass',
            'max_midpoint_residual', 'omega_h', 'h4_frequency_coefficient']]
        if not all(math.isfinite(value) for value in numbers):
            raise SystemExit('STOP: nonfinite exact c=0 control')
        if abs(float(row['mass']) - 2.0) > 1e-14:
            raise SystemExit('STOP: exact c=0 mass failed')
        if float(row['max_midpoint_residual']) > 1e-50:
            raise SystemExit('STOP: exact c=0 midpoint residual failed')
    for row in data['sg_quadrature']:
        if not all(math.isfinite(value) for value in row.values()) or row['relative_error'] > 1e-6:
            raise SystemExit('STOP: independent SG quadrature failed')
    phase_data = json.loads((qa / 'sg_initial_section.log').read_text())
    if phase_data['quadrature_checks'] != 'PASS':
        raise SystemExit('STOP: independent SG initial-section control failed')
    for row in phase_data['rows']:
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise SystemExit('STOP: nonfinite SG initial-section control')
    record = {'status': 'PASS', 'checks': checks,
              'qualification': 'Exact identities and finite-dimensional kernel checks; '
                               'not uniform nonlinear PDE stability or reference certification.'}
    (qa / 'ADDITIONAL_CONTROLS.json').write_text(json.dumps(record, indent=2) + '\n')
    print('ADDITIONAL_CONTROLS_PASS: scalar cumulants, exact c=0 midpoint, SG initial sections, kernel G1/H2')


if __name__ == '__main__':
    main()
