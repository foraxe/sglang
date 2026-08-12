"""coTensor POSIX-FD lifecycle adapter for SGLang DWDP.

This module is imported only when ``--dwdp-vmm-backend cotensor`` is selected.
It deliberately keeps DWDP layout, tensors, streams, and prefetch scheduling in
SGLang while replacing CUDA allocation/import/map/unmap ownership.
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Dict, List, Optional, Tuple

import cotensor
import torch
import torch.distributed as dist

from sglang.srt.cuda_vmm_utils import (
    align_down,
    align_up,
    exchange_posix_fds,
    get_device_granularity,
    tensor_from_pointer,
)
from sglang.srt.layers.moe.dwdp.layout import (
    DwdpExpertLayout,
    EdgeInfo,
    LayerWeightSpecs,
    MnnvlHandleSet,
    PageAlignedLayout,
)
from sglang.srt.layers.moe.dwdp.page_pool import (
    DEFAULT_PAGE_SIZE_MULTIPLIER,
    compute_slot_sizes,
)

logger = logging.getLogger(__name__)


def _close_fds(fds) -> None:
    for fd in fds:
        try:
            os.close(fd)
        except OSError:
            pass


def _dtype_name(dtype: torch.dtype) -> str:
    names = {
        torch.uint8: "uint8",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }
    try:
        return names[dtype]
    except KeyError as error:
        raise TypeError(f"coTensor DWDP does not support dtype {dtype}") from error


def _tensor_slice_from_view(
    view,
    *,
    byte_offset: int,
    byte_count: int,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a shaped tensor and the byte tensor retaining *view*."""

    storage = view.tensor("uint8", [view.size])
    data = storage.narrow(0, byte_offset, byte_count)
    tensor = data.view(dtype).reshape(shape)
    return tensor, storage


def _copy_local_weights_to_endpoints(
    sorted_keys: List[Tuple[int, str]],
    local_params: Dict[Tuple[int, str], torch.Tensor],
    layer_weight_specs: LayerWeightSpecs,
    layout: DwdpExpertLayout,
    device_id: int,
):
    granularity = get_device_granularity(device_id)
    endpoints = {}
    sizes = {}
    slabs = {}

    for layer_idx, name in sorted_keys:
        param = local_params[(layer_idx, name)]
        spec = layer_weight_specs[layer_idx][name]

        local_start_bytes = layout.local_expert_start * spec.expert_bytes
        local_end_bytes = layout.local_expert_end * spec.expert_bytes
        page_start = align_down(local_start_bytes, granularity)
        page_end = align_up(local_end_bytes, granularity)
        phys_size = page_end - page_start
        data_offset = local_start_bytes - page_start

        slab = cotensor.Slab.allocate(phys_size, device_id, True)
        slab.copy_from(param, data_offset)
        endpoint = slab.retain_endpoint()

        key = (layer_idx, name)
        slabs[key] = slab
        endpoints[key] = endpoint
        sizes[key] = phys_size

    torch.cuda.empty_cache()
    return MnnvlHandleSet(handles=endpoints, sizes=sizes), slabs


class CoTensorDWDPTransport:
    def __init__(self):
        self._handle_set: Optional[MnnvlHandleSet] = None
        self._peer_views: Dict[Tuple[int, int, str], torch.Tensor] = {}
        self._local_slabs = {}
        self._peer_endpoints = []
        self._peer_slots = []
        self._peer_mappings = []
        self._peer_storage_tensors = []
        self._released = False
        self._setup_committed = False
        self._original_params = None

    @classmethod
    def create(
        cls,
        layer_weight_specs: LayerWeightSpecs,
        local_params: Dict[Tuple[int, str], torch.Tensor],
        group: dist.ProcessGroup,
        layout: DwdpExpertLayout,
        device_id: int,
    ) -> "CoTensorDWDPTransport":
        transport = cls()
        sorted_keys = sorted(local_params.keys())
        transport._handle_set, transport._local_slabs = (
            _copy_local_weights_to_endpoints(
                sorted_keys,
                local_params,
                layer_weight_specs,
                layout,
                device_id,
            )
        )
        transport._original_params = local_params
        try:
            transport._import_peer_views(
                sorted_keys, layer_weight_specs, group, layout, device_id
            )
        except BaseException:
            transport.release()
            raise
        dist.barrier(group=group.device_group)
        logger.info(
            "coTensor DWDP transport complete: rank=%d/%d, %d local endpoints, "
            "%d peer views (POSIX fd)",
            layout.dwdp_rank,
            layout.dwdp_size,
            len(sorted_keys),
            len(transport._peer_views),
        )
        return transport

    def commit(self) -> None:
        """Commit setup only after SGLang has rebound every model Parameter."""
        self._original_params = None
        self._setup_committed = True

    def _import_peer_views(
        self,
        sorted_keys: List[Tuple[int, str]],
        layer_weight_specs: LayerWeightSpecs,
        group: dist.ProcessGroup,
        layout: DwdpExpertLayout,
        device_id: int,
    ) -> None:
        cpu_group = group.cpu_group
        local_descriptors = [
            self._handle_set.get_handle(li, name).export_fd()
            for li, name in sorted_keys
        ]
        local_fds = [descriptor.fileno() for descriptor in local_descriptors]
        peer_fds = {}

        key_counts = [None] * layout.dwdp_size
        dist.all_gather_object(key_counts, len(sorted_keys), group=cpu_group)
        if any(count != len(sorted_keys) for count in key_counts):
            raise RuntimeError(
                f"Mismatched DWDP weight handle counts across ranks: {key_counts}"
            )

        try:
            peer_fds = exchange_posix_fds(
                cpu_group,
                layout.dwdp_rank,
                layout.dwdp_size,
                local_fds,
                key_counts,
            )

            granularity = get_device_granularity(device_id)
            for key_idx, (layer_idx, name) in enumerate(sorted_keys):
                spec = layer_weight_specs[layer_idx][name]
                for peer_rank in range(layout.dwdp_size):
                    if peer_rank == layout.dwdp_rank:
                        continue

                    peer_start, peer_end = layout.peer_ranges[peer_rank]
                    peer_start_bytes = peer_start * spec.expert_bytes
                    peer_end_bytes = peer_end * spec.expert_bytes
                    peer_page_start = align_down(peer_start_bytes, granularity)
                    peer_page_end = align_up(peer_end_bytes, granularity)
                    peer_phys_size = peer_page_end - peer_page_start
                    peer_data_offset = peer_start_bytes - peer_page_start

                    fd = peer_fds[(peer_rank, key_idx)]
                    endpoint = cotensor.Endpoint.import_fd(fd, peer_phys_size)
                    slot = cotensor.Slot.reserve(peer_phys_size, granularity)
                    mapping = endpoint.map(
                        slot,
                        0,
                        0,
                        peer_phys_size,
                        device_id,
                        cotensor.AccessMode.READ_WRITE,
                    )
                    peer_shape = (peer_end - peer_start,) + spec.full_shape[1:]
                    peer_tensor, storage = _tensor_slice_from_view(
                        mapping,
                        byte_offset=peer_data_offset,
                        byte_count=peer_end_bytes - peer_start_bytes,
                        shape=peer_shape,
                        dtype=spec.dtype,
                    )

                    self._peer_endpoints.append(endpoint)
                    self._peer_slots.append(slot)
                    self._peer_mappings.append(mapping)
                    self._peer_storage_tensors.append(storage)
                    self._peer_views[(peer_rank, layer_idx, name)] = peer_tensor
        finally:
            for descriptor in local_descriptors:
                descriptor.close()
            _close_fds(peer_fds.values())

    @property
    def handle_set(self) -> MnnvlHandleSet:
        assert self._handle_set is not None
        return self._handle_set

    @property
    def peer_views(self) -> Dict[Tuple[int, int, str], torch.Tensor]:
        return self._peer_views

    def release(self) -> None:
        if self._released:
            return

        self._peer_views.clear()
        self._peer_storage_tensors.clear()
        gc.collect()
        live = [mapping.live_tensors for mapping in self._peer_mappings]
        if any(live):
            raise RuntimeError(
                f"coTensor DWDP peer mappings retain tensor aliases: {live}"
            )
        for mapping in self._peer_mappings:
            mapping.unbind()
        if self._peer_mappings:
            del mapping
        self._peer_mappings.clear()
        gc.collect()
        for endpoint in self._peer_endpoints:
            endpoint.close()
        self._peer_endpoints.clear()
        self._peer_slots.clear()

        if self._handle_set is not None:
            for endpoint in self._handle_set.handles.values():
                endpoint.close()
            self._handle_set = None
        self._local_slabs.clear()
        self._original_params = None
        self._released = True


class _CoTensorPagePool:
    def __init__(self, slot_sizes: List[int], device_id: int, page_size: int):
        self.page_size = page_size
        self._slabs = []
        self._endpoints = []
        for size in slot_sizes:
            page_count = align_up(size, page_size) // page_size
            slabs = [
                cotensor.Slab.allocate(page_size, device_id, True)
                for _ in range(page_count)
            ]
            self._slabs.append(slabs)
            self._endpoints.append([slab.retain_endpoint() for slab in slabs])

    def map(self, slot_index: int, slot, slot_offset: int, size: int, pool_offset: int):
        if size % self.page_size != 0 or pool_offset % self.page_size != 0:
            raise ValueError("coTensor DWDP page-pool mapping is not page-aligned")
        first_page = pool_offset // self.page_size
        mappings = []
        for index in range(size // self.page_size):
            endpoint = self._endpoints[slot_index][first_page + index]
            mapping = endpoint.map(
                slot,
                slot_offset + index * self.page_size,
                0,
                self.page_size,
                self._slabs[slot_index][first_page + index].device,
                cotensor.AccessMode.READ_WRITE,
            )
            mappings.append(mapping)
        return mappings

    def release(self) -> None:
        for endpoints in self._endpoints:
            for endpoint in endpoints:
                endpoint.close()
        self._endpoints.clear()
        self._slabs.clear()


class CoTensorWeightBuffer:
    def __init__(
        self,
        layer_weight_specs: LayerWeightSpecs,
        handles: MnnvlHandleSet,
        local_start: int,
        local_end: int,
        dwdp_size: int,
        device_id: int,
    ):
        self._layer_weight_specs = layer_weight_specs
        self._handles = handles
        self._local_start = local_start
        self._local_end = local_end
        self._dwdp_size = dwdp_size
        self._device_id = device_id
        self._granularity = get_device_granularity(device_id)
        self._pool_page_size = DEFAULT_PAGE_SIZE_MULTIPLIER * self._granularity
        self._page_pool: Optional[_CoTensorPagePool] = None
        self._moe_layer_indices = sorted(layer_weight_specs.keys())
        self._layouts: Dict[int, Dict[str, PageAlignedLayout]] = {}
        self._tensors: Dict[int, Dict[str, torch.Tensor]] = {}
        self._remote_slices = {}
        self._slots = {}
        self._views = {}
        self._view_roots = {}
        self._released = False

    @classmethod
    def create(
        cls,
        layer_weight_specs: LayerWeightSpecs,
        handles: MnnvlHandleSet,
        local_start: int,
        local_end: int,
        dwdp_size: int,
        device_id: int,
    ) -> "CoTensorWeightBuffer":
        buf = cls(
            layer_weight_specs,
            handles,
            local_start,
            local_end,
            dwdp_size,
            device_id,
        )
        for layer_idx, weight_specs in layer_weight_specs.items():
            buf._layouts[layer_idx] = {}
            for name, spec in weight_specs.items():
                buf._layouts[layer_idx][name] = PageAlignedLayout.compute(
                    expert_bytes=spec.expert_bytes,
                    num_experts=spec.num_experts,
                    local_start=local_start,
                    local_end=local_end,
                    granularity=buf._granularity,
                    handle_phys_size=handles.get_size(layer_idx, name),
                    pool_granularity=buf._pool_page_size,
                )

        assignments = {
            layer_idx: buf.buffer_index_for_layer(layer_idx)
            for layer_idx in layer_weight_specs
        }
        slot_sizes = compute_slot_sizes(buf._layouts, assignments)
        buf._page_pool = _CoTensorPagePool(slot_sizes, device_id, buf._pool_page_size)
        try:
            for layer_idx in buf._moe_layer_indices:
                buf._setup_layer(layer_idx)
        except BaseException:
            buf.release()
            raise
        return buf

    def _setup_layer(self, layer_idx: int) -> None:
        weight_layouts = self._layouts[layer_idx]
        weight_specs = self._layer_weight_specs[layer_idx]
        buffer_slot = self.buffer_index_for_layer(layer_idx)
        pool_offset = 0

        self._tensors[layer_idx] = {}
        self._remote_slices[layer_idx] = {}
        self._slots[layer_idx] = []
        self._views[layer_idx] = []
        self._view_roots[layer_idx] = []

        for name, layout in weight_layouts.items():
            spec = weight_specs[name]
            endpoint = self._handles.get_handle(layer_idx, name)
            slot = cotensor.Slot.reserve(layout.total_size, self._granularity)
            self._slots[layer_idx].append(slot)

            if layout.pre_size > 0:
                views = self._page_pool.map(
                    buffer_slot, slot, 0, layout.pre_size, pool_offset
                )
                self._views[layer_idx].extend(views)
                self._view_roots[layer_idx].extend(
                    view.tensor("uint8", [view.size]) for view in views
                )
                pool_offset += layout.pre_size

            local_view = endpoint.map(
                slot,
                layout.pre_size,
                0,
                layout.mnnvl_size,
                self._device_id,
                cotensor.AccessMode.READ_WRITE,
            )
            self._views[layer_idx].append(local_view)
            self._view_roots[layer_idx].append(
                local_view.tensor("uint8", [local_view.size])
            )

            if layout.post_size > 0:
                views = self._page_pool.map(
                    buffer_slot,
                    slot,
                    layout.pre_size + layout.mnnvl_size,
                    layout.post_size,
                    pool_offset,
                )
                self._views[layer_idx].extend(views)
                self._view_roots[layer_idx].extend(
                    view.tensor("uint8", [view.size]) for view in views
                )
                pool_offset += layout.post_size

            full_tensor = tensor_from_pointer(
                slot.address + layout.pre_padding,
                layout.num_experts * layout.expert_bytes,
                shape=spec.full_shape,
                dtype=spec.dtype,
                device_id=self._device_id,
            )
            self._tensors[layer_idx][name] = full_tensor

            slices = []
            if self._local_start > 0:
                slices.append((full_tensor[: self._local_start], 0, self._local_start))
            if self._local_end < spec.num_experts:
                slices.append(
                    (full_tensor[self._local_end :], self._local_end, spec.num_experts)
                )
            self._remote_slices[layer_idx][name] = slices

    def get_full_tensor(self, layer_idx: int, name: str) -> torch.Tensor:
        return self._tensors[layer_idx][name]

    def get_remote_slices(self, layer_idx: int, name: str):
        return self._remote_slices[layer_idx][name]

    def get_edge_info(self, layer_idx: int, name: str) -> EdgeInfo:
        return self._layouts[layer_idx][name].get_edge_info()

    def get_layout(self, layer_idx: int, name: str) -> PageAlignedLayout:
        return self._layouts[layer_idx][name]

    @property
    def layer_indices(self) -> List[int]:
        return list(self._moe_layer_indices)

    @property
    def local_start(self) -> int:
        return self._local_start

    @property
    def local_end(self) -> int:
        return self._local_end

    @property
    def device_id(self) -> int:
        return self._device_id

    def weight_names(self, layer_idx: int) -> List[str]:
        return list(self._layer_weight_specs[layer_idx].keys())

    def buffer_index_for_layer(self, layer_idx: int) -> int:
        if layer_idx in self._moe_layer_indices:
            return self._moe_layer_indices.index(layer_idx) % 2
        return layer_idx % 2

    def release(self) -> None:
        if self._released:
            return
        self._remote_slices.clear()
        self._tensors.clear()
        self._view_roots.clear()
        gc.collect()
        live = [view.live_tensors for views in self._views.values() for view in views]
        if any(live):
            raise RuntimeError(
                f"coTensor DWDP composite mappings retain tensor aliases: {live}"
            )
        for views in self._views.values():
            for view in views:
                view.unbind()
        if self._views:
            del view
        self._views.clear()
        gc.collect()
        self._slots.clear()
        if self._page_pool is not None:
            self._page_pool.release()
            self._page_pool = None
        self._released = True
