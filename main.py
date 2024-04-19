from simulations.sc_ee_simulation.main_our import main_repeat as main_repeat_our
from simulations.sc_ee_simulation.main_our import main_grid_search as main_grid_search_our
from simulations.sc_ee_simulation.configs.configs import config
from simulations.sc_ee_simulation.config import SimulationConfig, SimulationType

import warnings

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    config_data = SimulationConfig(**config)
    if config_data.simulation.simulation_type == SimulationType.ours:
        main_repeat_our(config_data, config) if config_data.requirements_search is None else main_grid_search_our(config_data, config)