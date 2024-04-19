from typing import List, Optional
import pytest
from mec.entities import UE, BaseStation
from mec.processing import task_processing as uut
from mec.tasks.task import Computing, Task, TaskComputationEntity
from mec.mec_types import EntityID
import attr

from src.common.Location import Location


@attr.s
class BaseStationTest(BaseStation):
    _id: int = attr.ib()
    _location: Location = attr.ib()

    @property
    def id(self) -> int:
        return self._id

    @property
    def location(self) -> Location:
        return self._location


@attr.s
class UETest(UE):
    _id: int = attr.ib()
    _location: Location = attr.ib()

    @property
    def id(self) -> int:
        return self._id

    @property
    def location(self) -> Location:
        return self._location

    def tasks_to_compute(self) -> List[Task]:
        return []


def dummy_task_factory(
    ttl: float = 0.5,
    request_size: float = 10,
    response_size: float = 10,
    instruction_count: float = 100,
    task_owner_id: EntityID = 1,
    target_id: Optional[EntityID] = 2,
    created_timestep: float = 0,
    data: dict = {},
    used_timestep_interval: Optional[float] = None,
    used_timestep: Optional[float] = None,
) -> Task:
    task = Task(
        ttl=ttl,
        request_size=request_size,
        response_size=response_size,
        instruction_count=instruction_count,
        task_owner_id=task_owner_id,
        target_id=target_id,
        created_timestep=created_timestep,
        data=data,
    )
    if used_timestep_interval is not None:
        task.used_timestep_interval = used_timestep_interval
    if used_timestep is not None:
        task.used_timestep = used_timestep
    return task


def dummy_computing_factory(
    task: Task,
    remaining_computation: Optional[float] = None,
) -> Task:
    computing = Computing(task=task)
    if remaining_computation is not None:
        computing.remaining_computation = remaining_computation
    return computing

# def dummy_message_factory(
#     task: Task,

# )

@pytest.mark.parametrize(
    ("comp_entity", "available_computing_power", "duration_sim_timestamp", "expected_result", "expected_computing"),
    [
        (
            dummy_task_factory().computing,
            100,
            0.5,
            uut.ProcessResult(used_computing=100, task_computed=True, time_spent=0.5),
            dummy_computing_factory(
                dummy_task_factory(
                    used_timestep_interval=0.5,
                ),
                remaining_computation=0,
            ),
        ),
        (
            dummy_task_factory(instruction_count=120).computing,
            100,
            0.5,
            uut.ProcessResult(used_computing=100, task_computed=False, time_spent=0.5),
            dummy_computing_factory(
                dummy_task_factory(
                    instruction_count=120,
                    used_timestep_interval=0.5,
                ),
                remaining_computation=20.0,
            ),
        ),
        (
            dummy_task_factory(instruction_count=120, used_timestep_interval=0.2).computing,
            60,
            0.3,
            uut.ProcessResult(used_computing=60, task_computed=False, time_spent=0.3),
            dummy_computing_factory(
                dummy_task_factory(
                    instruction_count=120,
                    used_timestep_interval=0.5,
                ),
                remaining_computation=60,
            )
        ),
        (
            dummy_task_factory(instruction_count=70, used_timestep_interval=0.1).computing,
            80,
            0.4,
            uut.ProcessResult(used_computing=70, task_computed=True, time_spent=pytest.approx(0.35)),
            dummy_computing_factory(
                dummy_task_factory(
                    instruction_count=70,
                    used_timestep_interval=pytest.approx(0.45),
                ),
                remaining_computation=0,
            )
        ),
        (
            dummy_task_factory(instruction_count=120, used_timestep_interval=0.4).computing,
            100,
            0.0,
            uut.ProcessResult(used_computing=0, task_computed=False, time_spent=0),
            dummy_task_factory(instruction_count=120, used_timestep_interval=0.4).computing
        ),
    ]
)
def test_process_task_computation(
    comp_entity: TaskComputationEntity,
    expected_result: uut.ProcessResult,
    expected_computing: TaskComputationEntity,
    available_computing_power: int,
    duration_sim_timestamp: float,
) -> None:
    result = uut.process_task_computation(available_computing_power, comp_entity, duration_sim_timestamp)
    assert result == expected_result
    assert comp_entity == expected_computing


def test_process_radio_round_robin() -> None:
    bs = BaseStationTest(0, Location(1,1),
        ips=100,
        bandwidth=100,
        resource_blocks=100,
        tx_frequency=4,
        tx_power=0.6,
        coverage_radius=100,
    )
    ues = [
        UETest(1, Location(1,1)),
        # TestUE(2, Location(1,1)),
        # TestUE(3, Location(1,1)),
    ]
    sim_dt = 0.5
    def calculate_avg_MBps(ue: UE) -> float:
        return {
            1: 40,
            2: 15,
            3: 20,
        }[ue.id]

    tasks = [
        dummy_task_factory(target_id=0, created_timestep=0.2),
        dummy_task_factory(target_id=0, created_timestep=0.2),
    ]
    messages = lambda _: [t.request for t in tasks]
    ues[0].tasks_sending = [t.request for t in tasks]

    uut._proces_radio_round_robin(bs, ues, sim_dt, calculate_avg_MBps, messages)

    assert bs.tasks_computing[0].task.request.remaining_computation == 0
    assert bs.tasks_computing[0].task.used_timestep_interval == 0.45
    assert ues[0].tasks_sending[0].remaining_computation == 8
    assert ues[0].tasks_sending[0].task.used_timestep_interval == 0.5


