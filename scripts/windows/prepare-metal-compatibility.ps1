#Requires -Version 7.2
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$KextDirectory,
    [ValidateSet('Diagnostic','Global')][string]$Routing = 'Diagnostic'
)
. "$PSScriptRoot/start.ps1" -FunctionsOnly
Assert-ReimsVmStopped
$manifestPath = Join-Path $VmRoot 'state/opencore.json'
$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
$vmConfigPath = Join-Path $VmRoot 'state/config.json'
if (Test-Path -LiteralPath $vmConfigPath) {
    $selected = Get-Content -Raw -LiteralPath $vmConfigPath | ConvertFrom-Json
    if (![string]::Equals([IO.Path]::GetFullPath($selected.opencore),
        [IO.Path]::GetFullPath($manifest.image), [StringComparison]::OrdinalIgnoreCase)) {
        throw 'The selected VM uses a different OpenCore image. Select its matching preparation manifest first.'
    }
}
foreach ($path in @($manifest.config, $manifest.image, $manifest.ocvalidate)) {
    [void](Assert-ReimsArtifactPath $path)
}
$backup = Join-Path $VmRoot ('state/metal-compat-backup-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $backup | Out-Null
Copy-Item -LiteralPath $manifest.config -Destination (Join-Path $backup 'config.plist')
Copy-Item -LiteralPath $manifest.image -Destination (Join-Path $backup 'OpenCore.img')
Copy-Item -LiteralPath $manifestPath -Destination (Join-Path $backup 'opencore.json')
$candidate = Join-Path $backup 'candidate.plist'
$python = Get-ReimsTool 'python.exe'
try {
    Invoke-ReimsTool $python @((Join-Path $PSScriptRoot 'metal-compat-config.py'),
        '--source', $manifest.config, '--target', $candidate,
        '--bundle', (Get-Item -LiteralPath $KextDirectory).FullName,
        '--repository', $ProjectRoot, '--routing', $Routing) | Out-Null
    Invoke-ReimsTool $manifest.ocvalidate @($candidate) | Out-Null
    Copy-Item -LiteralPath $candidate -Destination $manifest.config -Force
    & "$PSScriptRoot/prepare-opencore.ps1" -Version $manifest.version -Rebuild | Out-Null
} catch {
    Copy-Item -LiteralPath (Join-Path $backup 'config.plist') -Destination $manifest.config -Force
    Copy-Item -LiteralPath (Join-Path $backup 'OpenCore.img') -Destination $manifest.image -Force
    Copy-Item -LiteralPath (Join-Path $backup 'opencore.json') -Destination $manifestPath -Force
    throw
}
"Saved experimental Metal routing: $Routing (Darwin 25.6.0 only)."
"Previous OpenCore config and image: $backup"
"Guest graphics preferences are applied separately by REIMSTOOLS/setup.command."
