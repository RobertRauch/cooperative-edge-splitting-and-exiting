import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
import time
from typing import Any, Dict
import attr
import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars
from src.IISMotion import IISMotion
from src.common.SimulationClock import getDateTime
from src.common.SimulationClock import updateSimulationClock


@attr.s(frozen=True, kw_only=True)
class SimulationInfo:
    simulation_datetime: datetime = attr.ib()
    simulation_dt: float = attr.ib()  # in seconds
    simulation_step: int = attr.ib(default=0)
    simulation_name: str = attr.ib()
    training_steps: int = attr.ib(default=-1)
    new_day: bool = attr.ib(default=False)
    simulation_index: int = attr.ib(default=None)
    episode_index: int = attr.ib(default=None)


@attr.s(frozen=True)
class SimulationContext:
    sim_info: SimulationInfo = attr.ib()
    data: Dict[str, Any] = attr.ib(factory=lambda: {})


@attr.s(kw_only=True)
class IISMotionSimulation(ABC):
    """Class for simulation initialization and simulation running"""

    iismotion: IISMotion = attr.ib()
    number_of_ticks: int = attr.ib()
    simulation_dt: float = attr.ib()
    name: str = attr.ib()
    simulation_index: int = attr.ib(default=None)
    episode_index: int = attr.ib(default=None)
    training_steps: int = attr.ib(default=-1)
    gui_enabled: bool = attr.ib(default=False)
    gui_timeout: float = attr.ib(default=0.5)
    stepping: bool = attr.ib(default=False)
    _log: structlog.stdlib.BoundLogger = attr.ib(
        factory=structlog.get_logger, init=False
    )

    @abstractmethod
    def simulation_step(self, context: SimulationContext) -> None:
        """Perform simulation step"""

    def _create_context(
        self,
        current_context: SimulationContext,
        sim_step: int,
        sim_datetime: datetime,
        new_day: bool,
    ) -> SimulationContext:
        data = current_context.data if current_context else {}
        return SimulationContext(
            sim_info=SimulationInfo(
                simulation_step=sim_step,
                simulation_datetime=sim_datetime,
                simulation_name=self.name,
                new_day=new_day,
                simulation_dt=self.simulation_dt,
                simulation_index=self.simulation_index,
                training_steps=self.training_steps,
                episode_index=self.episode_index,
            ),
            data=data,
        )

    async def _simulation(self) -> None:
        """Main method of simulation run by run_simulation() method"""
        # self._log.info("iismotion_simulation.start")
        start = time.time()
        context = None
        for step in range(0, self.number_of_ticks):
            sim_datetime = getDateTime()
            bind_contextvars(step=step, datetime=sim_datetime)

            if self.stepping:
                input("Press enter to do step")

            new_day = updateSimulationClock(self.iismotion.secondsPerTick)

            # self._log.info("iismotion_simulation.step")

            stepStart = time.time()
            # do simulation step
            context = self._create_context(context, step, getDateTime(), new_day)
            self.simulation_step(context)
            stepEnd = time.time()
            self.iismotion.sendUpdateToFrontend()

            # self._log.debug(
            #     f"iismotion_simulation.step_time_took", time=stepEnd - stepStart
            # )

            if self.gui_enabled == True:
                await asyncio.sleep(self.gui_timeout)

            end = time.time()

        elapsed = end - start
        clear_contextvars()
        # self._log.info("iismotion_simulation.end", elapsed=elapsed)

    def run_simulation(self) -> None:
        """Run simulation"""
        if self.gui_enabled:
            loop = asyncio.get_event_loop()
            loop.create_task(self._simulation())
            loop.run_until_complete(self.iismotion.frontend.start_server)
            loop.run_forever()
        else:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self._simulation())
