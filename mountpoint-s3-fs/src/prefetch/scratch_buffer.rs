//! A small free-list of reusable heap buffers for `do_read`'s multi-part straddle path.
//!
//! On the multi-part read path, `do_read` must copy the first part out of its pool buffer so
//! the pool buffer is not pinned across the next allocation (see [`super::cursor`]). Doing that
//! copy with a fresh `Bytes::copy_from_slice` allocates and frees a heap buffer on every straddle
//! read. This module instead hands out a buffer drawn from a free-list and returns it to the list
//! when the resulting [`bytes::Bytes`] is dropped, so the underlying allocation is reused rather
//! than malloc/free-d each time.
//!
//! Concurrent `do_read` calls are bounded by the number of FUSE worker threads (default 16), so
//! the free-list holds at most that many buffers at steady state. Each buffer is at most the FUSE
//! read size (<= 1 MiB on Linux), bounding resident memory at ~16 MiB.

use std::sync::OnceLock;

use bytes::Bytes;

use crate::sync::{Arc, Mutex};

/// Process-global scratch pool shared by all cursors' `do_read` straddle copies.
///
/// A single shared free-list lets buffers freed by one FUSE worker thread be reused by another,
/// which is what bounds the resident set to ~(worker threads) buffers. Threading an explicit
/// `Arc<ScratchPool>` through the prefetcher would be cleaner for a production change; this global
/// keeps the prototype's diff small.
static GLOBAL: OnceLock<ScratchPool> = OnceLock::new();

/// Returns the process-global scratch pool, initializing it on first use.
pub fn global() -> &'static ScratchPool {
    GLOBAL.get_or_init(ScratchPool::new)
}

/// Buffers larger than this are not recycled — they are handed back to the allocator on drop.
/// The FUSE read size is <= 1 MiB on Linux, so straddle copies never exceed this in practice; the
/// cap only guards against an unexpectedly large request bloating the free-list.
const MAX_RECYCLED_CAPACITY: usize = 1024 * 1024;

/// Cap on the number of idle buffers retained. Bounded by FUSE worker-thread count in practice;
/// this is a hard backstop so a burst cannot grow the list without limit.
const MAX_IDLE_BUFFERS: usize = 32;

/// A reusable heap buffer free-list. Cheap to clone (`Arc` inside).
#[derive(Debug, Clone, Default)]
pub struct ScratchPool {
    idle: Arc<Mutex<Vec<Vec<u8>>>>,
}

impl ScratchPool {
    pub fn new() -> Self {
        Self::default()
    }

    /// Copy `data` into a recycled (or freshly allocated) buffer and return it as `Bytes`.
    ///
    /// When the returned `Bytes` is dropped, the backing buffer is returned to this pool for
    /// reuse (subject to the capacity and count caps).
    pub fn copy_from_slice(&self, data: &[u8]) -> Bytes {
        let mut buf = self.take_idle().unwrap_or_default();
        buf.clear();
        buf.extend_from_slice(data);
        Bytes::from_owner(ScratchBuffer {
            buf,
            pool: self.clone(),
        })
    }

    fn take_idle(&self) -> Option<Vec<u8>> {
        self.idle.lock().unwrap().pop()
    }

    fn return_buffer(&self, buf: Vec<u8>) {
        if buf.capacity() == 0 || buf.capacity() > MAX_RECYCLED_CAPACITY {
            return;
        }
        let mut idle = self.idle.lock().unwrap();
        if idle.len() < MAX_IDLE_BUFFERS {
            idle.push(buf);
        }
    }
}

/// Owner handed to `Bytes::from_owner`. Derefs to the copied bytes; on drop, returns its buffer to
/// the pool for reuse.
#[derive(Debug)]
struct ScratchBuffer {
    buf: Vec<u8>,
    pool: ScratchPool,
}

impl AsRef<[u8]> for ScratchBuffer {
    fn as_ref(&self) -> &[u8] {
        &self.buf
    }
}

impl Drop for ScratchBuffer {
    fn drop(&mut self) {
        let buf = std::mem::take(&mut self.buf);
        self.pool.return_buffer(buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_data_faithfully() {
        let pool = ScratchPool::new();
        let data = b"hello world";
        let bytes = pool.copy_from_slice(data);
        assert_eq!(&bytes[..], data);
    }

    #[test]
    fn recycles_backing_allocation() {
        let pool = ScratchPool::new();

        // First use allocates, then returns to the pool on drop.
        let bytes = pool.copy_from_slice(&[1u8; 4096]);
        drop(bytes);
        assert_eq!(pool.idle.lock().unwrap().len(), 1);

        // Second use draws from the free-list rather than allocating.
        let bytes = pool.copy_from_slice(&[2u8; 4096]);
        assert_eq!(pool.idle.lock().unwrap().len(), 0);
        assert_eq!(&bytes[..], &[2u8; 4096][..]);
    }

    #[test]
    fn does_not_recycle_oversized_buffers() {
        let pool = ScratchPool::new();
        let bytes = pool.copy_from_slice(&vec![0u8; MAX_RECYCLED_CAPACITY + 1]);
        drop(bytes);
        assert_eq!(pool.idle.lock().unwrap().len(), 0);
    }

    #[test]
    fn caps_idle_buffer_count() {
        let pool = ScratchPool::new();
        // Hold more live buffers than the cap, then drop them all.
        let live: Vec<_> = (0..MAX_IDLE_BUFFERS + 8).map(|_| pool.copy_from_slice(&[7u8; 64])).collect();
        drop(live);
        assert_eq!(pool.idle.lock().unwrap().len(), MAX_IDLE_BUFFERS);
    }

    #[test]
    fn data_survives_pool_drop() {
        // The `Bytes` must remain valid even if the originating pool handle is dropped.
        let bytes = {
            let pool = ScratchPool::new();
            pool.copy_from_slice(b"outlives pool")
        };
        assert_eq!(&bytes[..], b"outlives pool");
    }
}
