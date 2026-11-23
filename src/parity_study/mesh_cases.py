"""Consistent catalogue names and amplitudes for matched mesh experiments."""
import numpy as np


def symmetric_cases(config):
    anchor = config['null']['A']
    amplitudes = config['null'].get('amplitudes', [anchor])
    if (not amplitudes or len(set(amplitudes)) != len(amplitudes)
            or anchor not in amplitudes or any(not np.isfinite(a) or a <= 0 for a in amplitudes)):
        raise ValueError('Symmetric-control amplitudes must be unique positive values including null.A')
    return [('null_symmetric' if a == anchor else f'null_A{a:g}_symmetric', a) for a in amplitudes]


def case_parameters(config):
    cases = {}
    if config['null']['seeds']:
        cases['null_za'] = (0., 0.)
        cases.update({name:(a, 0.) for name,a in symmetric_cases(config)})
    if config['scan']['seeds']:
        for a in config['scan']['amplitudes']:
            cases[f'scan_A{a:g}_plus'] = (a, 1.)
            cases[f'scan_A{a:g}_minus'] = (a, -1.)
    return cases
