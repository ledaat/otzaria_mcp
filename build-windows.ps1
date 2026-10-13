# Build otzaria-mcp.exe on Windows. Needs Rust (rustup, MSVC toolchain) and the
# Visual Studio C++ Build Tools (for the bundled SQLite and other C code).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
cargo build --release -p otzaria-mcp
$exe = Join-Path $PSScriptRoot "target\release\otzaria-mcp.exe"
New-Item -ItemType Directory -Force "C:\Programs\cli" | Out-Null
Copy-Item $exe "C:\Programs\cli\otzaria-mcp-windows-x86_64.exe" -Force
Write-Host "Installed C:\Programs\cli\otzaria-mcp-windows-x86_64.exe"
& "C:\Programs\cli\otzaria-mcp-windows-x86_64.exe" --version
