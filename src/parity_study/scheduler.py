"""Slurm resources for the ensemble and its validation/report stages."""


def resource_args(scheduler, stage):
    defaults = {
        'validate': dict(partition='debug-cpu', cpus=2, memory='4G', time='00:15:00'),
        'summary': dict(partition='public-cpu', cpus=1, memory='2G', time='00:15:00'),
    }
    resources = (dict(defaults[stage], **scheduler.get(stage, {}))
                 if stage in defaults else scheduler)
    return ['--partition', resources['partition'], '--cpus-per-task', str(resources['cpus']),
            '--mem', resources['memory'], '--time', resources['time']]
