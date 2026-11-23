"""Explicit, non-overlapping random-seed ranges for new mesh ensembles."""


def seed_start(config):
    start = config.get('seed_start', 0)
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError('seed_start must be a nonnegative integer')
    if start and config['field'].get('model') != 'mesh_za_helical':
        raise ValueError('Nonzero seed_start is supported only for mesh ensembles')
    return start


def seed_ids(config, count):
    first = seed_start(config)
    return range(first, first + count)
