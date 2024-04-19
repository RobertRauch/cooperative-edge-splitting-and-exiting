config = {
    "result_dir": "/results/RB/236-180/200/{}/run_{}/ep_{}/",
    "result_name": "{}/",
    "repeat": 10,
    "repeat_start_idx": 0,
    "episodes": 1,
    "comment": "Test Config",
    'requirements_search': None,
    'simulation': {
        'dt': 1,
        'simulation_type': 'ours',
        'steps': 500,
        'training_steps': 300,
        'gui_enabled': False,
        'gui_timeout': 0,
    },
    'vehicles': [
        {
            'count': 10,
            'speed_ms': 11,
            'speed_variation': 2,
            'ips': 23.1e9,
            'model_location': "",
            'task': {
                'ttl': 0.01,
                'generation_period': 0.01,
                'generation_delay_range': 0
            },
            'requirements': {
                'latency': 0.01,
                'accuracy': 0.8
            },
            'weights': {
                'latency': 1,
                'accuracy': 1
            }
        },
    ],
    'base_stations': {
        "count": 1,
        "min_radius": 30,
        "ips": 41.29e12,
        "bandwidth": 100e6,
        "resource_blocks": 2000,
        "tx_frequency": 2e9,
        "tx_power": 0.1,
        "coverage_radius": 500
    }
}