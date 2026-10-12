//! [otzaria-mcp local patch] A read-only, lock-free MmapDirectory, lifted from
//! smart_search_bench.rs so the MCP can open the app's live index without writing to it.
use std::io;
use std::path::Path;
use std::sync::Arc;
use tantivy::directory::error::{
    DeleteError, LockError, OpenDirectoryError, OpenReadError, OpenWriteError,
};
use tantivy::directory::{
    Directory, DirectoryLock, FileHandle, FileSlice, Lock, MmapDirectory, WatchCallback,
    WatchHandle, WritePtr,
};

/// An `MmapDirectory` that refuses every write and takes no lock file.
#[derive(Clone, Debug)]
pub(crate) struct ReadOnlyDirectory(MmapDirectory);

impl ReadOnlyDirectory {
    pub(crate) fn open(path: &Path) -> Result<Self, OpenDirectoryError> {
        MmapDirectory::open(path).map(Self)
    }
}

fn refused() -> io::Error {
    io::Error::new(io::ErrorKind::PermissionDenied, "otzaria-mcp opens the index read-only")
}

impl Directory for ReadOnlyDirectory {
    fn get_file_handle(&self, path: &Path) -> Result<Arc<dyn FileHandle>, OpenReadError> {
        self.0.get_file_handle(path)
    }

    fn open_read(&self, path: &Path) -> Result<FileSlice, OpenReadError> {
        self.0.open_read(path)
    }

    fn delete(&self, path: &Path) -> Result<(), DeleteError> {
        Err(DeleteError::IoError {
            io_error: Arc::new(refused()),
            filepath: path.to_path_buf(),
        })
    }

    fn exists(&self, path: &Path) -> Result<bool, OpenReadError> {
        self.0.exists(path)
    }

    fn open_write(&self, path: &Path) -> Result<WritePtr, OpenWriteError> {
        Err(OpenWriteError::wrap_io_error(refused(), path.to_path_buf()))
    }

    fn atomic_read(&self, path: &Path) -> Result<Vec<u8>, OpenReadError> {
        self.0.atomic_read(path)
    }

    fn atomic_write(&self, _path: &Path, _data: &[u8]) -> io::Result<()> {
        Err(refused())
    }

    fn sync_directory(&self) -> io::Result<()> {
        Ok(())
    }

    // The reader's meta lock guards against the owner's garbage collection; a benchmark
    // leaves the owner's lock file alone and accepts that race.
    fn acquire_lock(&self, _lock: &Lock) -> Result<DirectoryLock, LockError> {
        Ok(DirectoryLock::from(Box::new(())))
    }

    fn watch(&self, _watch_callback: WatchCallback) -> tantivy::Result<WatchHandle> {
        Ok(WatchHandle::empty())
    }
}
