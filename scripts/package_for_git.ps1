$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$releaseRoot = Join-Path $root "release"
$target = Join-Path $releaseRoot "maternal-mortality-dashboard-git-ready"
$zipPath = Join-Path $releaseRoot "maternal-mortality-dashboard-git-ready.zip"

if (Test-Path $target) {
    Remove-Item -Recurse -Force $target
}
if (Test-Path $zipPath) {
    Remove-Item -Force $zipPath
}
if (!(Test-Path $releaseRoot)) {
    New-Item -ItemType Directory -Path $releaseRoot | Out-Null
}

New-Item -ItemType Directory -Path $target | Out-Null

$topFiles = @(".env.example", ".gitignore", "README.md", "requirements.txt", "pyproject.toml")
foreach ($file in $topFiles) {
    Copy-Item -Path (Join-Path $root $file) -Destination (Join-Path $target $file)
}

Copy-Item -Recurse -Force -Path (Join-Path $root "scripts") -Destination (Join-Path $target "scripts")
Copy-Item -Recurse -Force -Path (Join-Path $root "src") -Destination (Join-Path $target "src")
Copy-Item -Recurse -Force -Path (Join-Path $root "tests") -Destination (Join-Path $target "tests")

$dataTarget = Join-Path $target "data"
$logsTarget = Join-Path $target "logs"
New-Item -ItemType Directory -Path $dataTarget | Out-Null
New-Item -ItemType Directory -Path $logsTarget | Out-Null

foreach ($sub in @("raw", "interim", "processed", "external")) {
    $subPath = Join-Path $dataTarget $sub
    New-Item -ItemType Directory -Path $subPath | Out-Null
    Copy-Item -Path (Join-Path $root ("data/" + $sub + "/.gitkeep")) -Destination (Join-Path $subPath ".gitkeep")
}
Copy-Item -Path (Join-Path $root "logs/.gitkeep") -Destination (Join-Path $logsTarget ".gitkeep")

Get-ChildItem -Path $target -Recurse -Directory -Force |
    Where-Object { $_.Name -in @("__pycache__", ".pytest_cache", ".ruff_cache") } |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

Get-ChildItem -Path $target -Recurse -File -Force |
    Where-Object { $_.Extension -eq ".pyc" } |
    Remove-Item -Force -ErrorAction SilentlyContinue

Compress-Archive -Path (Join-Path $target "*") -DestinationPath $zipPath

Write-Output ("TARGET_FOLDER=" + $target)
Write-Output ("ZIP_FILE=" + $zipPath)
Write-Output ("ZIP_SIZE_MB=" + [Math]::Round((Get-Item $zipPath).Length / 1MB, 2))
