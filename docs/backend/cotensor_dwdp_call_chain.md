# coTensor-backed DWDP lifecycle boundary

## Native call chain at the pinned base

1. `ModelRunner.maybe_init_dwdp()` creates `DwdpManager` after model loading.
2. `DwdpManager.setup()` partitions experts without changing their logical
   layout, then creates `DWDPTransport`.
3. `_copy_local_weights_to_handles()` creates a shareable CUDA VMM allocation,
   maps it temporarily, copies each local expert tensor, unmaps the temporary
   VA, and retains the allocation handle.
4. `export_shareable_handles()` prefers FABRIC and falls back to POSIX FDs;
   `exchange_posix_fds()` transports FDs with `SCM_RIGHTS`.
5. `import_peer_handle()` imports each peer allocation. A `VmmReservation` maps
   it into a peer view used as the prefetch source.
6. `WeightBuffer` reserves the stable full-expert VA and maps the local retained
   allocation between two page-pool regions. `PagePool` backs the remote expert
   destination regions used by the two alternating buffers.
7. `DWDPWeightManager` keeps the same peer tensors, full-expert tensors,
   copy-stream operations, and two-slot event protocol.
8. `DWDPWeightManager.release()` releases `WeightBuffer` before `DWDPTransport`;
   reservations unmap before allocation handles are released.

## Candidate boundary

`--dwdp-vmm-backend cotensor` replaces steps 3-6 and the corresponding teardown
with move-only coTensor `Slab`, `Endpoint`, `Slot`, and `View` owners. The
transport still carries POSIX FDs, and SGLang still owns rank coordination,
tensor shapes, expert layout, prefetch copies, streams, and events.

The Python binding must expose a non-owning CUDA tensor while retaining its
`View`; closing an Endpoint must not invalidate a live View, and closing a View
must be rejected until SGLang has quiesced the associated stream work.

