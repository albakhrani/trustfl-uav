"""Stable private Win32 memory-query bindings, with no numerical dependencies.

Each imported copy owns its DLL function objects. Prototypes are initialized once
under a lock and published only after initialization succeeds. Calls use separate
output buffers, so the sampler and budget check may query concurrently.
"""
from __future__ import annotations

import ctypes
import threading


# Fixed-width Win32 types also make the ABI inspectable on non-Windows hosts.
_DWORD = ctypes.c_uint32
_BOOL = ctypes.c_int32
_HANDLE = ctypes.c_void_p


class ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [('cb', _DWORD), ('PageFaultCount', _DWORD)] + [
        (name, ctypes.c_size_t) for name in (
            'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
            'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]


class MemoryStatus(ctypes.Structure):
    _fields_ = [('length', _DWORD), ('load', _DWORD)] + [
        (name, ctypes.c_uint64) for name in (
            'total_physical', 'available_physical', 'total_pagefile',
            'available_pagefile', 'total_virtual', 'available_virtual',
            'available_extended_virtual')]


class _Bindings:
    def __init__(self):
        # Do not use ctypes.windll: its cached function objects are shared with
        # unrelated modules that may assign incompatible pointer prototypes.
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.psapi = ctypes.WinDLL('psapi', use_last_error=True)
        self.current_process = self.kernel.GetCurrentProcess
        self.current_process.argtypes = []
        self.current_process.restype = _HANDLE
        self.process_info = self.psapi.GetProcessMemoryInfo
        self.process_info.argtypes = [
            _HANDLE, ctypes.POINTER(ProcessMemoryCounters), _DWORD]
        self.process_info.restype = _BOOL
        self.memory_status = self.kernel.GlobalMemoryStatusEx
        self.memory_status.argtypes = [ctypes.POINTER(MemoryStatus)]
        self.memory_status.restype = _BOOL
        self.open_process = self.kernel.OpenProcess
        self.open_process.argtypes = [_DWORD, _BOOL, _DWORD]
        self.open_process.restype = _HANDLE
        self.close_handle = self.kernel.CloseHandle
        self.close_handle.argtypes = [_HANDLE]
        self.close_handle.restype = _BOOL


_bindings = None
_bindings_lock = threading.Lock()


def _get_bindings():
    global _bindings
    with _bindings_lock:
        if _bindings is None:
            # Assignment happens only after every prototype is ready. If the
            # loader raises, a later memory query can try initialization again.
            _bindings = _Bindings()
        return _bindings


def _process_counters(bindings, handle):
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not bindings.process_info(handle, ctypes.byref(counters), counters.cb):
        return None
    return counters


def process_memory():
    """Return the current process working set and lifetime peak, in bytes."""
    bindings = _get_bindings()
    counters = _process_counters(bindings, bindings.current_process())
    if counters is None:
        raise OSError('GetProcessMemoryInfo failed')
    # GetCurrentProcess returns a pseudo-handle; it must not be closed.
    return {'rss_bytes': int(counters.WorkingSetSize),
            'lifetime_peak_rss_bytes': int(counters.PeakWorkingSetSize)}


def physical_memory_bytes(failure_message='GlobalMemoryStatusEx failed'):
    """Return total physical RAM, retaining each caller's failure message."""
    bindings = _get_bindings()
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not bindings.memory_status(ctypes.byref(status)):
        raise OSError(failure_message)
    return int(status.total_physical)


def process_rss_bytes(pid):
    """Query an owned child PID, preserving None on unavailable working set."""
    bindings = _get_bindings()
    handle = bindings.open_process(0x0400 | 0x0010, False, int(pid))
    if not handle:
        return None
    try:
        counters = _process_counters(bindings, handle)
        return None if counters is None else int(counters.WorkingSetSize)
    finally:
        # Preserve prior CloseHandle handling: always call it for an owned
        # handle; a false return does not replace the memory query's result.
        bindings.close_handle(handle)
