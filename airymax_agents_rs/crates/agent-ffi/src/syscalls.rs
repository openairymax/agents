// SPDX-FileCopyrightText: 2026 SPHARX Ltd.
// SPDX-License-Identifier: AGPL-3.0-or-later OR Apache-2.0
//! agentrt syscall FFI 绑定。
//!
//! 签名严格匹配 `agentrt/atoms/syscall/include/syscalls.h`：
//!   - `airy_err_t` → `i32`（C 侧为 enum，0=success）
//!   - 所有字符串参数为 `*const c_char`，需以 NULL 结尾
//!   - 输出参数为 `*mut *mut c_char`，调用者须用 `airy_sys_free` 释放

use libc::{c_char, c_int, c_void};
use std::ffi::{CStr, CString};
use std::ptr;
use thiserror::Error;

/// agentrt syscall 错误代码（与 C 侧 `error.h` 对齐，0=success）。
pub type AiryErr = c_int;

pub const AIRY_SUCCESS: AiryErr = 0;
pub const AIRY_ERR_INVALID_PARAM: AiryErr = 1002;
pub const AIRY_ERR_OUT_OF_MEMORY: AiryErr = 3001;
pub const AIRY_ERR_NOT_FOUND: AiryErr = 4004;
pub const AIRY_ERR_STATE_ERROR: AiryErr = 4005;

/// FFI 调用错误。
#[derive(Debug, Error)]
pub enum SyscallError {
    #[error("FFI call returned error code {0}")]
    Code(AiryErr),

    #[error("null pointer returned")]
    NullPointer,

    #[error("string contains interior NUL byte")]
    NulError(#[from] std::ffi::NulError),

    #[error("UTF-8 conversion failed: {0}")]
    Utf8(#[from] std::str::Utf8Error),
}

pub type FfiResult<T> = std::result::Result<T, SyscallError>;

extern "C" {
    fn airy_syscalls_init() -> AiryErr;
    fn airy_syscalls_cleanup();
    fn airy_sys_free(ptr: *mut c_void);

    fn airy_sys_memory_write(
        data: *const c_void,
        len: usize,
        metadata: *const c_char,
        out_record_id: *mut *mut c_char,
    ) -> AiryErr;

    fn airy_sys_memory_get(
        record_id: *const c_char,
        out_data: *mut *mut c_void,
        out_len: *mut usize,
    ) -> AiryErr;

    fn airy_sys_memory_search(
        query: *const c_char,
        limit: u32,
        out_record_ids: *mut *mut *mut c_char,
        out_scores: *mut *mut f32,
        out_count: *mut usize,
    ) -> AiryErr;

    fn airy_sys_memory_delete(record_id: *const c_char) -> AiryErr;

    fn airy_sys_agent_spawn(spec: *const c_char, out_agent_id: *mut *mut c_char) -> AiryErr;
    fn airy_sys_agent_terminate(agent_id: *const c_char) -> AiryErr;
    fn airy_sys_agent_invoke(
        agent_id: *const c_char,
        input: *const c_char,
        input_len: usize,
        out_output: *mut *mut c_char,
    ) -> AiryErr;
    fn airy_sys_agent_list(out_ids: *mut *mut *mut c_char, out_count: *mut usize) -> AiryErr;
}

/// FFI 资源句柄，负责初始化与清理。
pub struct SyscallHandle;

impl SyscallHandle {
    /// 初始化系统调用层。
    ///
    /// # Safety
    /// 调用方须保证未重复初始化，且在调用任何 syscall 前调用。
    pub unsafe fn init() -> FfiResult<Self> {
        let rc = airy_syscalls_init();
        if rc == AIRY_SUCCESS {
            Ok(Self)
        } else {
            Err(SyscallError::Code(rc))
        }
    }

    /// 清理系统调用层资源。
    ///
    /// # Safety
    /// 必须在 init 成功后调用一次，且之后不得再调用任何 syscall。
    pub unsafe fn cleanup(&self) {
        airy_syscalls_cleanup();
    }
}

fn check(rc: AiryErr) -> FfiResult<()> {
    if rc == AIRY_SUCCESS {
        Ok(())
    } else {
        Err(SyscallError::Code(rc))
    }
}

/// 将 C 字符串指针接管为 Rust `String`，并经 `airy_sys_free` 释放。
///
/// # Safety
/// `ptr` 必须是 `airy_sys_*` 返回的、由 `airy_sys_free` 管理的堆分配 C 字符串。
unsafe fn take_c_string(ptr: *mut c_char) -> FfiResult<String> {
    if ptr.is_null() {
        return Err(SyscallError::NullPointer);
    }
    let s = CStr::from_ptr(ptr).to_str()?.to_owned();
    airy_sys_free(ptr as *mut c_void);
    Ok(s)
}

/// 写入记忆记录，返回 record_id。
pub fn memory_write(data: &[u8], metadata: Option<&str>) -> FfiResult<String> {
    let metadata_c = match metadata {
        Some(s) => Some(CString::new(s)?),
        None => None,
    };
    let mut out: *mut c_char = ptr::null_mut();
    let rc = unsafe {
        airy_sys_memory_write(
            data.as_ptr() as *const c_void,
            data.len(),
            metadata_c.as_ref().map_or(ptr::null(), |c| c.as_ptr()),
            &mut out,
        )
    };
    check(rc)?;
    unsafe { take_c_string(out) }
}

/// 按 record_id 读取记忆数据。
pub fn memory_get(record_id: &str) -> FfiResult<Vec<u8>> {
    let rid = CString::new(record_id)?;
    let mut out_data: *mut c_void = ptr::null_mut();
    let mut out_len: usize = 0;
    let rc = unsafe { airy_sys_memory_get(rid.as_ptr(), &mut out_data, &mut out_len) };
    check(rc)?;
    if out_data.is_null() {
        return Err(SyscallError::NullPointer);
    }
    let bytes = unsafe { std::slice::from_raw_parts(out_data as *const u8, out_len) }.to_vec();
    unsafe { airy_sys_free(out_data) };
    Ok(bytes)
}

/// 关键词检索，返回 (record_id, score) 列表。
pub fn memory_search(query: &str, limit: u32) -> FfiResult<Vec<(String, f32)>> {
    let q = CString::new(query)?;
    let mut out_ids: *mut *mut c_char = ptr::null_mut();
    let mut out_scores: *mut f32 = ptr::null_mut();
    let mut out_count: usize = 0;
    let rc = unsafe {
        airy_sys_memory_search(
            q.as_ptr(),
            limit,
            &mut out_ids,
            &mut out_scores,
            &mut out_count,
        )
    };
    check(rc)?;

    let mut out = Vec::with_capacity(out_count);
    for i in 0..out_count {
        unsafe {
            let id_ptr = *out_ids.add(i);
            if !id_ptr.is_null() {
                let id = CStr::from_ptr(id_ptr).to_str()?.to_owned();
                airy_sys_free(id_ptr as *mut c_void);
                let score = *out_scores.add(i);
                out.push((id, score));
            }
        }
    }
    if !out_ids.is_null() {
        unsafe { airy_sys_free(out_ids as *mut c_void) };
    }
    if !out_scores.is_null() {
        unsafe { airy_sys_free(out_scores as *mut c_void) };
    }
    Ok(out)
}

/// 删除记忆记录。
pub fn memory_delete(record_id: &str) -> FfiResult<()> {
    let rid = CString::new(record_id)?;
    let rc = unsafe { airy_sys_memory_delete(rid.as_ptr()) };
    check(rc)
}

/// 派生 Agent，返回 agent_id。
pub fn agent_spawn(spec: &str) -> FfiResult<String> {
    let s = CString::new(spec)?;
    let mut out: *mut c_char = ptr::null_mut();
    let rc = unsafe { airy_sys_agent_spawn(s.as_ptr(), &mut out) };
    check(rc)?;
    unsafe { take_c_string(out) }
}

/// 终止 Agent。
pub fn agent_terminate(agent_id: &str) -> FfiResult<()> {
    let s = CString::new(agent_id)?;
    let rc = unsafe { airy_sys_agent_terminate(s.as_ptr()) };
    check(rc)
}

/// 调用 Agent，返回 output 字符串。
pub fn agent_invoke(agent_id: &str, input: &str) -> FfiResult<String> {
    let aid = CString::new(agent_id)?;
    let input_bytes = input.as_bytes();
    let mut out: *mut c_char = ptr::null_mut();
    let rc = unsafe {
        airy_sys_agent_invoke(
            aid.as_ptr(),
            input_bytes.as_ptr() as *const c_char,
            input_bytes.len(),
            &mut out,
        )
    };
    check(rc)?;
    unsafe { take_c_string(out) }
}

/// 列出所有 Agent ID。
pub fn agent_list() -> FfiResult<Vec<String>> {
    let mut out_ids: *mut *mut c_char = ptr::null_mut();
    let mut out_count: usize = 0;
    let rc = unsafe { airy_sys_agent_list(&mut out_ids, &mut out_count) };
    check(rc)?;

    let mut out = Vec::with_capacity(out_count);
    for i in 0..out_count {
        unsafe {
            let id_ptr = *out_ids.add(i);
            if !id_ptr.is_null() {
                let id = CStr::from_ptr(id_ptr).to_str()?.to_owned();
                airy_sys_free(id_ptr as *mut c_void);
                out.push(id);
            }
        }
    }
    if !out_ids.is_null() {
        unsafe { airy_sys_free(out_ids as *mut c_void) };
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 仅校验常量与 C 侧 `error.h` 一致性（不调用 FFI，避免依赖 libagentrt.so）。
    #[test]
    fn error_codes_match_c() {
        assert_eq!(AIRY_SUCCESS, 0);
        assert_eq!(AIRY_ERR_INVALID_PARAM, 1002);
        assert_eq!(AIRY_ERR_OUT_OF_MEMORY, 3001);
        assert_eq!(AIRY_ERR_NOT_FOUND, 4004);
        assert_eq!(AIRY_ERR_STATE_ERROR, 4005);
    }
}
