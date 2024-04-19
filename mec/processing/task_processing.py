from collections import defaultdict
from enum import Enum
from typing import Callable, Dict, Iterable, List, NamedTuple, Protocol

from structlog.stdlib import BoundLogger

from mec.entities import UE, BaseStation, MecEntity
from mec.mec_types import EntityID
from mec.tasks.task import Computing, Message, Task, TaskComputationEntity, ComputingUE


class _QueueAssignment(Enum):
    bs = 1
    ue = 2


class BSProtocol(Protocol):
    id: EntityID
    ips: int
    tasks_computing: List[Computing]
    tasks_sending: List[Message]


class ProcessResult(NamedTuple):
    used_computing: float
    task_computed: bool
    time_spent: float


DeadlineCallback = Callable[[Task], None]
ReservationCalculator = Callable[[EntityID], float]


def deadline_reached(task: Task) -> bool:
    return task.ttl <= round(task.lived_timestep, 6)


def process_radio_uplink_round_robin(
    bs: BaseStation,
    connected_ues: List[UE],
    sim_dt: float,
    calculate_available_MBps: Callable[[UE], float],
    log: BoundLogger,
    on_deadline: DeadlineCallback = lambda _: None,
) -> None:
    _proces_radio_round_robin(
        bs,
        connected_ues,
        sim_dt,
        calculate_available_MBps,
        lambda ue: [m for m in ue.tasks_sending if m.is_uplink],
        _QueueAssignment.bs,
        log,
        "uplink",
        on_deadline,
    )


def process_radio_downlink_round_robin(
    bs: BaseStation,
    connected_ues: List[UE],
    sim_dt: float,
    calculate_available_MBps: Callable[[UE], float],
    log: BoundLogger,
    on_deadline: DeadlineCallback = lambda _: None,
) -> None:
    _proces_radio_round_robin(
        bs,
        connected_ues,
        sim_dt,
        calculate_available_MBps,
        lambda ue: [
            m for m in bs.tasks_sending if m.is_downlink and m.destination_id == ue.id
        ],
        _QueueAssignment.ue,
        log,
        "downlink",
        on_deadline,
    )


def process_ue_tasks_computation_round_robin(
    ue: UE,
    sim_dt: float,
    log: BoundLogger,
    quantum: float = 0.1,
    on_deadline: DeadlineCallback = lambda _: None,
) -> None:
    if ue.used_timestep_interval >= sim_dt:
        return
    instructions_per_dt = ue.ips*sim_dt
    log = log.bind(ue_id=ue.id)

    tasks = list(ue.tasks_computing)
    # tasks: Dict[EntityID, List[Computing]] = defaultdict(lambda: [])
    groups_count = len(ue.tasks_computing)
    if groups_count == 0:
        log.debug("ue_computation.no_tasks_to_compute")
        return

    inst_per_round = instructions_per_dt * quantum
    # dt_used = 0

    while ue.used_timestep_interval < sim_dt and not len(ue.tasks_computing) == 0:
        if should_wait(tasks, ue.used_timestep_interval):
            waiting_timestamp = get_waiting_timestamp_ue(tasks, ue.used_timestep_interval, sim_dt)
            ue.used_timestep_interval = waiting_timestamp
        for task in tasks:
            if task.remaining_computation <= 0 or ue.used_timestep_interval >= sim_dt \
                    or task.task.used_timestep_interval > ue.used_timestep_interval:
                continue
            available_instructions = inst_per_round if (ue.used_timestep_interval * ue.ips) + inst_per_round < instructions_per_dt \
                                                            else instructions_per_dt - (ue.used_timestep_interval * ue.ips)
            if available_instructions < 0:
                raise Exception(f"this should not happen ({available_instructions})")

            result = process_task_computation_cpu(
                ue.ips,
                available_instructions,
                task,
                ue.used_timestep_interval,
                sim_dt,
            )
            ue.used_timestep_interval += result.time_spent
            computation_queue_assignment(
                result.task_computed,
                task,
                ue.tasks_computing,
                ue.tasks_sending,
                log,
                ue,
                on_deadline,
            )


def process_tasks_computation_fifo(
    cpu: MecEntity,
    sim_dt: float,
    log: BoundLogger,
    is_vehicle: bool,
    on_deadline: DeadlineCallback = lambda _: None,
) -> None:
    if cpu.used_timestep_interval >= sim_dt:
        return
    instructions_per_dt = cpu.ips * sim_dt
    log = log.bind(ue_id=cpu.id)
    tasks = list(cpu.tasks_computing)
    tasks.sort(key=lambda x: x.task.used_timestep_interval, reverse=False)

    for task in tasks:
        if task.task.used_timestep_interval > cpu.used_timestep_interval:
            cpu.used_timestep_interval = task.task.used_timestep_interval
        available_instructions = round(instructions_per_dt * (sim_dt - round(cpu.used_timestep_interval, 10)))

        if round(instructions_per_dt * (sim_dt - round(cpu.used_timestep_interval, 5))) < 0:
            raise Exception(f"this should not happen ({available_instructions}) ({instructions_per_dt}) ({sim_dt}) ({cpu.used_timestep_interval})")
        if available_instructions < 0:
            available_instructions = 0

        task.task.used_timestep_interval = cpu.used_timestep_interval
        # if waiting time was longer than TTL
        if task.task.ttl <= task.task.lived_timestep:
            task.task.used_timestep_interval = 0
            task.task.used_timestep = task.task.ttl + task.task.created_timestep

        result = process_task_computation_cpu(
            cpu.ips,
            available_instructions,
            task,
            cpu.used_timestep_interval,
            sim_dt,
        )
        cpu.used_timestep_interval += result.time_spent
        if "computation" not in task.task.data.keys():
            task.task.data["computation"] = {}
            task.task.data["computation"]["vehicle"] = 0
            task.task.data["computation"]["server"] = 0
        task.task.data["computation"]["vehicle" if is_vehicle else "server"] += result.used_computing

        computation_queue_assignment(
            result.task_computed,
            task,
            cpu.tasks_computing,
            cpu.tasks_sending,
            log,
            cpu if is_vehicle else None,
            on_deadline,
        )


def should_wait(tasks: [ComputingUE], cur_dt: float) -> bool:
    result = True
    for task in tasks:
        if task.task.used_timestep_interval < cur_dt and task.remaining_computation > 0:
            result = False
    return result


def get_waiting_timestamp_ue(tasks: [ComputingUE], cur_dt: float, sim_dt: float) -> float:
    result = sim_dt
    for task in tasks:
        if task.task.used_timestep_interval <= cur_dt and task.remaining_computation > 0:
            return cur_dt
        if cur_dt < task.task.used_timestep_interval < result and task.remaining_computation > 0:
            result = task.task.used_timestep_interval
    return result


def process_bs_tasks_computation_round_robin_per_task_owner(
    bs: BaseStation,
    sim_dt: float,
    log: BoundLogger,
    # Quantum ... 0-1 - % of sim_dt
    quantum: float = 0.1,
    on_deadline: DeadlineCallback = lambda _: None,
) -> None:
    instructions_per_dt = bs.ips * sim_dt
    log = log.bind(bs_id=bs.id)
    # group tasks
    ueid_tasks_mapping: Dict[EntityID, List[Computing]] = defaultdict(lambda: [])
    per_owner_dt_used: Dict[EntityID, float] = defaultdict()
    for t in bs.tasks_computing:
        ueid_tasks_mapping[t.task.task_owner_id].append(t)
        per_owner_dt_used[t.task.task_owner_id] = 0

    groups_count = len(ueid_tasks_mapping)
    if groups_count == 0:
        log.debug("bs_computation.no_tasks_to_compute")
        return

    inst_per_round = instructions_per_dt * quantum
    dt_used = 0

    while dt_used < sim_dt:
        available_instructions = inst_per_round if (dt_used * bs.ips) + inst_per_round < instructions_per_dt \
                                                        else instructions_per_dt - (dt_used * bs.ips)
        if check_tasks_queue(ueid_tasks_mapping):
            break

        tasks = get_next_task_queue(ueid_tasks_mapping, per_owner_dt_used, dt_used, quantum, sim_dt)
        # If task doesnt exist wait for next task
        if not tasks:
            per_owner_dt_used = dict.fromkeys(per_owner_dt_used, 0)
            # for key, value in per_owner_dt_used.items():
            new_dt = get_waiting_timestamp_bs(ueid_tasks_mapping, dt_used, sim_dt)
            dt_used = new_dt
            tasks = get_next_task_queue(ueid_tasks_mapping, per_owner_dt_used, dt_used, quantum, sim_dt)
            if not tasks:
                break
        for task in tasks:
            if available_instructions == 0:
                break
            if available_instructions < 0:
                raise Exception("This shouldnt happen")
            result = process_task_computation_cpu(
                        bs.ips,
                        available_instructions,
                        task,
                        dt_used,
                        sim_dt,
                    )
            dt_used += result.time_spent
            per_owner_dt_used[task.task.task_owner_id] += result.time_spent
            available_instructions -= result.used_computing
            computation_queue_assignment(
                        result.task_computed,
                        task,
                        bs.tasks_computing,
                        bs.tasks_sending,
                        log,
                        on_deadline,
                    )


def get_waiting_timestamp_bs(tasks: Dict[EntityID, List[Computing]], used_dt: float, sim_dt: float) -> float:
    if check_tasks_queue(tasks):
        return sim_dt
    waiting_time = sim_dt
    for owner in tasks.values():
        for t in owner:
            if t.task.used_timestep_interval <= used_dt and t.remaining_computation > 0:
                return used_dt
            if used_dt < t.task.used_timestep_interval < waiting_time and t.remaining_computation > 0:
                waiting_time = t.task.used_timestep_interval
    return waiting_time


def get_next_task_queue(tasks: Dict[EntityID, List[Computing]], per_owner_dt_used: Dict[EntityID, float],
                  used_dt: float, quantum: float,
                  sim_dt: float) -> Computing | None:
    if check_tasks_queue(tasks):
        return None
    for idx, (owner_id, owner_tasks) in enumerate(tasks.items()):
        if per_owner_dt_used[owner_id] >= sim_dt*quantum:
            continue
        for t in owner_tasks:
            if t.task.used_timestep_interval <= used_dt and t.remaining_computation > 0:
                return owner_tasks
    return None


def check_tasks_queue(tasks: Dict[EntityID, List[Computing]]) -> bool:
    if not tasks or len(tasks) == 0:
        return True
    result = True
    for task in tasks.values():
        if task and len(task) > 0:
            return False
    return result


def process_bs_tasks_computation_round_robin(
    bs: BaseStation,
    sim_dt: float,
    ips_reservation: ReservationCalculator,
    log: BoundLogger,
    on_deadline: DeadlineCallback = lambda _: None,
) -> None:
    instructions_per_dt = bs.ips * sim_dt
    log = log.bind(bs_id=bs.id)
    # group tasks
    ueid_tasks_mapping: Dict[EntityID, List[Computing]] = defaultdict(lambda: [])
    for t in bs.tasks_computing:
        ueid_tasks_mapping[t.task.task_owner_id].append(t)

    if len(ueid_tasks_mapping) == 0:
        log.debug("bs_computation.no_tasks_to_compute")
        return
    # NOTE: if not all of the available
    # instractions for ue are used, they are lost (not shared with other ues computation)

    for ue_id, tasks in ueid_tasks_mapping.items():

        available_instructions = min(
            ips_reservation(ue_id) * sim_dt, instructions_per_dt
        )
        computing_remaining_timestamp = sim_dt

        for computing_task in tasks:
            log = log.bind(
                ue_id=computing_task.task.task_owner_id, task_id=computing_task.task.id
            )

            if available_instructions < 0:
                raise Exception(f"this should not happen ({available_instructions})")

            computing_remaining_timestamp = min(
                computing_remaining_timestamp,
                sim_dt - computing_task.task.used_timestep_interval,
            )

            computing_task.task.used_timestep_interval = (
                sim_dt - computing_remaining_timestamp
            )

            result = process_task_computation(
                available_instructions * (computing_remaining_timestamp / sim_dt),
                computing_task,
                computing_remaining_timestamp,
            )

            log.debug("bs_computation.process_result", result=result)

            computing_remaining_timestamp -= result.time_spent
            computation_queue_assignment(
                result.task_computed,
                computing_task,
                bs.tasks_computing,
                bs.tasks_sending,
                log,
                on_deadline,
            )


def computation_queue_assignment(
    computation_processed: bool,
    computation: Computing,
    origin_queue: List[Computing],
    sending_queue: List[Message],
    log: BoundLogger,
    current_ue: UE = None,
    on_deadline: DeadlineCallback = lambda _: None,
) -> None:
    if deadline_reached(computation.task):
        # log.info("bs_queue_assignment.deadline_reached", task=computation.task)
        _pop_computation_entity(origin_queue, computation)
        on_deadline(computation.task)
        return

    if not computation_processed:
        return

    # remove from tasks_sending_queue
    _pop_computation_entity(origin_queue, computation)

    if computation.task.computing.remaining_computation == 0 \
            and computation.task.computing_ue.remaining_computation == 0 and current_ue:
        current_ue.accept_computed_task(computation.task)
        return

    if type(computation) == ComputingUE:
        sending_queue.append(computation.task.request)
    else:
        sending_queue.append(computation.task.response)


def message_ue_queue_assignment(
    message_processed: bool,
    message: Message,
    origin_queue: List[Message],
    current_ue: UE,
    log: BoundLogger,
    on_deadline: DeadlineCallback = lambda _: None,
) -> None:
    if deadline_reached(message.task):
        # log.info("ue_queue_assignment.deadline_reached", task=message.task)
        _pop_computation_entity(origin_queue, message)
        on_deadline(message.task)
        return

    if not message_processed:
        return

    if message.destination_id is None:
        log.warning("ue_queue_assignment.no_destination_id")
        return

    # remove from tasks_sending_queue
    _pop_computation_entity(origin_queue, message)

    if message.task.task_owner_id == current_ue.id:
        if message.task.computing_ue.remaining_computation > 0:
            current_ue.tasks_computing.append(message.task.computing_ue)
        else:
            current_ue.accept_computed_task(message.task)
    else:
        # forward task
        raise NotImplementedError(
            "This functionality of forwarding task on UE is not implemented"
        )


def message_bs_queue_assignment(
    message_processed: bool,
    message: Message,
    origin_queue: List[Message],
    current_bs: BaseStation,
    log: BoundLogger,
    on_deadline: DeadlineCallback = lambda _: None,
) -> None:
    if deadline_reached(message.task):
        _pop_computation_entity(origin_queue, message)
        on_deadline(message.task)
        return

    if not message_processed:
        return

    if message.destination_id is None:
        return

    _pop_computation_entity(origin_queue, message)

    if message.destination_id == current_bs.id:
        current_bs.tasks_computing.append(message.task.computing)
    else:
        raise NotImplementedError(
            "This functionality of forwarding task on BS is not implemented"
        )


def _proces_radio_round_robin(
    bs: BaseStation,
    connected_ues: List[UE],
    sim_dt: float,
    calculate_available_MBps: Callable[[UE], float],
    messages: Callable[[UE], Iterable[Message]],
    queue_assignment: _QueueAssignment,
    log: BoundLogger,
    connection_type: str,
    on_deadline: DeadlineCallback = lambda _: None,
) -> None:
    for ue in connected_ues:
        throughput = calculate_available_MBps(ue)
        ue.step_data[connection_type] = throughput
        available_MB_per_dt = throughput * sim_dt
        computing_remaining_timestamp = sim_dt
        for message in messages(ue):
            log = log.bind(ue_id=ue.id, task_id=message.task.id)
            if available_MB_per_dt < 0:
                raise Exception(f"this should not happen ({available_MB_per_dt})")

            computing_remaining_timestamp = min(
                computing_remaining_timestamp,
                sim_dt - message.task.used_timestep_interval,
            )

            message.task.used_timestep_interval = sim_dt - computing_remaining_timestamp
            if message.task.lived_timestep < message.task.ttl:
                result = process_task_computation(
                    available_MB_per_dt * (computing_remaining_timestamp / sim_dt),
                    message,
                    computing_remaining_timestamp,
                    connection_type
                )
                computing_remaining_timestamp -= result.time_spent
            else:
                message.task.used_timestep_interval = 0
                message.task.used_timestep = message.task.ttl + message.task.created_timestep
                result = ProcessResult(0, False, 0)

            if queue_assignment == _QueueAssignment.bs:
                message_bs_queue_assignment(
                    result.task_computed,
                    message,
                    ue.tasks_sending,
                    bs,
                    log,
                    on_deadline,
                )
            if queue_assignment == _QueueAssignment.ue:
                message_ue_queue_assignment(
                    result.task_computed,
                    message,
                    bs.tasks_sending,
                    ue,
                    log,
                    on_deadline,
                )


def process_task_computation_cpu(
    ips: float,
    available_computing_power: float,
    comp_entity: TaskComputationEntity,
    used_dt: float,
    sim_dt: float,
) -> ProcessResult:
    time_to_ttl = round(comp_entity.task.ttl - comp_entity.task.lived_timestep, 10)
    comp_dt = comp_entity.remaining_computation/ips
    if time_to_ttl < 0:
        raise Exception("Lived time is lower than ttl")
    elif time_to_ttl == 0: 
        return ProcessResult(0, False, 0)

    entity_remaining_computation = time_to_ttl*ips if time_to_ttl < comp_dt else comp_entity.remaining_computation
    if available_computing_power < entity_remaining_computation:
        comp_entity.remaining_computation -= available_computing_power
        comp_entity.task.used_timestep_interval += available_computing_power/ips
        return ProcessResult(available_computing_power, False, (available_computing_power/ips))

    computing_time = (
        entity_remaining_computation
        / ips
    )
    comp_entity.remaining_computation -= entity_remaining_computation
    comp_entity.task.used_timestep_interval += computing_time

    # task was computed
    return ProcessResult(entity_remaining_computation, comp_entity.remaining_computation <= 0, computing_time)


def process_task_computation(
    available_computing_power: float,
    comp_entity: TaskComputationEntity,
    duration_sim_timestamp: float,
    computation_type: str = None,
) -> ProcessResult:
    if duration_sim_timestamp <= 0 or available_computing_power <= 0:
        return ProcessResult(0, False, 0)

    entity_remaining_computation = comp_entity.remaining_computation
    if computation_type is not None and computation_type not in comp_entity.task.data.keys():
        comp_entity.task.data[computation_type] = 0

    if available_computing_power < entity_remaining_computation:
        computation_size = 0
        computation_duration = 0
        if comp_entity.task.lived_timestep + duration_sim_timestamp > comp_entity.task.ttl:
            computation_duration = comp_entity.task.ttl - comp_entity.task.lived_timestep
            computation_size = (available_computing_power/duration_sim_timestamp)*computation_duration
        else:
            computation_size = available_computing_power
            computation_duration = duration_sim_timestamp
        entity_remaining_computation -= computation_size
        comp_entity.remaining_computation = entity_remaining_computation
        comp_entity.task.used_timestep_interval += computation_duration
        if computation_type is not None:
            comp_entity.task.data[computation_type] += computation_duration
        return ProcessResult(computation_size, False, computation_duration)

    computing_time = (
        entity_remaining_computation
        / available_computing_power
        * duration_sim_timestamp
    )
    
    if comp_entity.task.lived_timestep + computing_time > comp_entity.task.ttl:
        computation_duration = comp_entity.task.ttl - comp_entity.task.lived_timestep
        computation_size = (available_computing_power/duration_sim_timestamp)*computation_duration
        comp_entity.remaining_computation -= computation_size
        computing_time = computation_duration
        entity_remaining_computation = computation_size
    else:
        comp_entity.remaining_computation = 0
    if computation_type is not None:
        comp_entity.task.data[computation_type] += computing_time
    comp_entity.task.used_timestep_interval += computing_time

    return ProcessResult(entity_remaining_computation, True, computing_time)


def _pop_computation_entity(
    l: List[TaskComputationEntity], ce: TaskComputationEntity
) -> None:
    l[:] = [c for c in l if c.task.id != ce.task.id]
