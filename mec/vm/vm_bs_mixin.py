from typing import Dict
from mec.mec_types import EntityID
import attr

### NOT COMPLETED

@attr.s
class VirtualMachine:
    startup_sec: float = attr.ib()
    shut_down_sec: float = attr.ib()
    running_required_hw: float = attr.ib()


@attr.s
class VMStat:
    started_at: float = attr.ib()
    running_at: float = attr.ib()
    vm: VirtualMachine = attr.ib()


class VMNotEnoughHWResourcesError(Exception):
    pass

class UE_VM_Mixin:
    def __init__(self) -> None:
        self.__vms: Dict[EntityID, VMStat] = {}

    def vm_start(self, ue_id: EntityID, vm: VirtualMachine, current_timestamp: float) -> None:
        """starts VM for user `ue_id`"""
        virtual = self.__vms.get(ue_id, None)

        if virtual is not None:
            return

        # if there is not running virtual, start virtual only if there are
        # available resources
        # TODO somehow handle free resources and used resources
        if self.free_resources < vm.running_required_hw:
            raise VMNotEnoughHWResourcesError()

        self.__vms[ue_id] = VMStat(
            started_at=current_timestamp, vm=vm, running_at=current_timestamp+vm.startup_sec
        )
        self._used_hw_resources += vm.running_required_hw

    def vm_shut_down(self, ue_id: EntityID) -> None:
        """shut down virtual for the ue_id"""
        # TODO: increase resources
        self.__vms.pop(ue_id, None)

    def vm_is_running(self, ue_id: EntityID, current_timestamp: float) -> bool:
        vm_stat = self.__vms.get(ue_id, None)
        if vm_stat is None:
            return False
        return current_timestamp >= vm_stat.started_at