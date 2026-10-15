#Requires -Version 7.2
[CmdletBinding()]
param(
    [string]$Qemu,
    [switch]$DryRun,
    [switch]$Paused,
    [switch]$Headless,
    [switch]$WithoutInstaller,
    [switch]$DiagnosticTcg,
    [switch]$FunctionsOnly
)
. "$PSScriptRoot/environment.ps1"
. "$PSScriptRoot/usb.ps1"

function Assert-ReimsArtifactPath([string]$Path) {
    $full = [IO.Path]::GetFullPath($Path)
    $prefix = $ProjectRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    $vmPrefix = $VmRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    if (!$full.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase) -and !$full.StartsWith($vmPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Artifact must stay under the project: $full"
    }
    return $full
}

function Write-ReimsJson([string]$Path, $Value) {
    $Path = Assert-ReimsArtifactPath $Path
    New-Item -ItemType Directory -Force -Path (Split-Path $Path) | Out-Null
    ConvertTo-Json -InputObject $Value -Depth 20 | Set-Content -LiteralPath "$Path.new" -Encoding utf8NoBOM
    Move-Item -LiteralPath "$Path.new" -Destination $Path -Force
}

function Get-ReimsTool([string]$Name, [string]$ExplicitPath) {
    if ($ExplicitPath) { return (Get-Item -LiteralPath $ExplicitPath -ErrorAction Stop).FullName }
    $built = Join-Path $ProjectRoot "build/qemu/$Name"
    if ($Name -like 'qemu-*' -and (Test-Path -LiteralPath $built -PathType Leaf)) { return $built }
    $command = Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($command) { return $command.Source }
    $candidates = @((Join-Path $MsysRoot "ucrt64/bin/$Name"), (Join-Path $MsysRoot "mingw64/bin/$Name"))
    if ($Name -eq '7z.exe') { $candidates += Join-Path $env:ProgramFiles '7-Zip/7z.exe' }
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    }
    throw "Missing native tool $Name. Install it or pass its path explicitly."
}

function Invoke-ReimsTool([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed ($LASTEXITCODE): $($Arguments -join ' ')" }
}

function Get-ReimsVmConfig {
    $path = Join-Path $VmRoot 'state/config.json'
    if (!(Test-Path -LiteralPath $path)) { throw 'Run prepare-vm.ps1 first.' }
    $config = Get-Content -Raw -LiteralPath $path | ConvertFrom-Json
    if ($config.schema -ne 1) { throw 'Unknown VM configuration schema.' }
    foreach ($key in @('base_disk', 'active_disk', 'opencore', 'firmware_code', 'firmware_vars', 'rom')) {
        [void](Assert-ReimsArtifactPath $config.$key)
    }
    # A validation VM may share a read-only base chain inside the project.
    foreach ($key in @('active_disk', 'opencore', 'firmware_vars')) {
        $vmPrefix = $VmRoot.TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar
        if (![IO.Path]::GetFullPath($config.$key).StartsWith($vmPrefix,[StringComparison]::OrdinalIgnoreCase)) {
            throw "VM input must belong to the selected VM folder: $key"
        }
    }
    if ($config.active_disk -eq $config.base_disk) { throw 'Refusing to boot or modify the base disk directly; use a qcow2 overlay.' }
    if ($config.qmp_port -lt 1024 -or $config.qmp_port -gt 65535 -or $config.ssh_port -lt 1024 -or
        $config.ssh_port -gt 65535 -or $config.qmp_port -eq $config.ssh_port) { throw 'Invalid localhost QMP/SSH ports.' }
    return $config
}

function Get-ReimsVmProcess($Runtime) {
    $process = Get-Process -Id $Runtime.pid -ErrorAction SilentlyContinue
    if (!$process) { return $null }
    if ($process.StartTime.ToUniversalTime().Ticks -ne [long]$Runtime.start_ticks -or
        ![string]::Equals($process.Path, $Runtime.executable, [StringComparison]::OrdinalIgnoreCase)) {
        throw "PID $($Runtime.pid) belongs to another process; refusing to control it."
    }
    return $process
}

function Get-ReimsRuntime {
    $path = Join-Path $VmRoot 'state/runtime.json'
    if (Test-Path -LiteralPath $path) { return Get-Content -Raw -LiteralPath $path | ConvertFrom-Json }
    return $null
}

function Assert-ReimsVmStopped {
    $runtime = Get-ReimsRuntime
    if ($runtime -and (Get-ReimsVmProcess $runtime)) { throw 'Stop this VM with stop.ps1 -Exit before changing its disks or firmware.' }
}

function Invoke-ReimsQmp($Runtime, [string]$Command, [hashtable]$Arguments = @{}) {
    if (!(Get-ReimsVmProcess $Runtime)) { throw 'The recorded VM is no longer running.' }
    $client = [Net.Sockets.TcpClient]::new()
    try {
        if (!$client.ConnectAsync('127.0.0.1', [int]$Runtime.qmp_port).Wait(5000)) { throw 'QMP connection timed out.' }
        $stream = $client.GetStream()
        $stream.ReadTimeout = 5000
        $stream.WriteTimeout = 5000
        $reader = [IO.StreamReader]::new($stream, [Text.Encoding]::UTF8, $false, 1024, $true)
        $writer = [IO.StreamWriter]::new($stream, [Text.UTF8Encoding]::new($false), 1024, $true)
        $writer.AutoFlush = $true
        $writer.NewLine = "`n"
        $line = $reader.ReadLine()
        if (!$line -or !(($line | ConvertFrom-Json).PSObject.Properties['QMP'])) { throw 'Not a QMP greeting.' }
        foreach ($request in @(@{ execute = 'qmp_capabilities' }, @{ execute = $Command; arguments = $Arguments })) {
            $request.id = [guid]::NewGuid().ToString()
            $writer.WriteLine(($request | ConvertTo-Json -Depth 10 -Compress))
            do {
                $line = $reader.ReadLine()
                if ($null -eq $line) { throw "QMP closed during $($request.execute)." }
                $response = $line | ConvertFrom-Json
            } while (!$response.PSObject.Properties['id'] -or $response.id -ne $request.id)
            if ($response.PSObject.Properties['error']) { throw "QMP $($request.execute): $($response.error.desc)" }
        }
        return ,$response.return
    } finally { $client.Dispose() }
}

function ConvertTo-ReimsQemuPath([string]$Path) { return $Path.Replace('\', '/').Replace(',', ',,') }

function Get-ReimsNetworkMode($Config) {
    if ($Config.PSObject.Properties['network']) { return $Config.network.mode }
    return 'nat'
}

function Get-ReimsNetworkArgument($Config) {
    switch (Get-ReimsNetworkMode $Config) {
        nat { return "user,id=net0,ipv6=off,hostfwd=tcp:127.0.0.1:$($Config.ssh_port)-:22" }
        bridge {
            $name = $Config.network.adapter
            if ([string]::IsNullOrWhiteSpace($name) -or $name -match '[\r\n]') { throw 'Select a TAP adapter.' }
            return 'tap,id=net0,ifname=' + $name.Replace(',', ',,')
        }
        default { throw 'Unknown network mode.' }
    }
}

function Get-ReimsTapAdapters {
    # QEMU opens an Ethernet TAP device; Wintun is an IP tunnel and cannot substitute.
    return @(Get-NetAdapter -ErrorAction Stop | Where-Object InterfaceDescription -Match 'TAP-Windows|TAP-Win32|TAP.*Adapter')
}

function ConvertTo-ReimsWindowsArgument([string]$Value) {
    $escaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
    return '"' + $escaped + '"'
}

function Get-ReimsWhpxStatus {
    $system = Get-CimInstance Win32_ComputerSystem
    $feature = Get-CimInstance Win32_OptionalFeature -Filter "Name='HypervisorPlatform'"
    return [ordered]@{
        hypervisor_present = [bool]$system.HypervisorPresent
        feature_install_state = if ($feature) { [int]$feature.InstallState } else { $null }
        ready = [bool]($system.HypervisorPresent -and $feature -and $feature.InstallState -eq 1)
    }
}

function Get-ReimsQemuArguments($Config, [string]$RunDirectory, [switch]$Paused, [switch]$Headless,
    [switch]$WithoutInstaller, [switch]$DiagnosticTcg) {
    $code = ConvertTo-ReimsQemuPath $Config.firmware_code
    $vars = ConvertTo-ReimsQemuPath $Config.firmware_vars
    $oc = ConvertTo-ReimsQemuPath $Config.opencore
    $disk = ConvertTo-ReimsQemuPath $Config.active_disk
    $serial = ConvertTo-ReimsQemuPath (Join-Path $RunDirectory 'serial.log')
    $audio = if ($Config.PSObject.Properties['audio']) { $Config.audio } else { 'none' }
    if ($audio -notin @('none','dsound')) { throw 'Unknown Windows audio backend.' }
    $gfx = [ordered]@{
        driver = 'apple-gfx-pci'; id = 'gfx'; bus = 'pcie.0'; addr = '0x4'
        displays = 1; resolution = $Config.resolution; romfile = $Config.rom.Replace('\', '/')
        'display-modes' = @("$($Config.resolution)@60", '1280x720@60', '1920x1080@60', '2560x1440@60', '3840x2160@60') | Select-Object -Unique
    }
    # macOS needs QEMU's measured TSC leaf; Hyper-V passthrough reported 6.4 GHz for a 3.187 GHz clock.
    $qemuArguments = @(
        '-L', (ConvertTo-ReimsQemuPath (Join-Path $ProjectRoot 'vendor/qemu/pc-bios')),
        '-name', 'reims-windows', '-machine', 'q35', '-accel', $(if ($DiagnosticTcg) { 'tcg' } else { 'whpx,hyperv=off' }),
        '-cpu', $(if ($DiagnosticTcg) { 'max,vendor=GenuineIntel' } else { $Config.cpu }), '-smp', "$($Config.cpus),sockets=1,cores=$($Config.cpus),threads=1", '-m', "$($Config.memory_mib)M",
        '-global', 'ICH9-LPC.acpi-pci-hotplug-with-bridge-support=off',
        '-global', 'ICH9-LPC.disable_s3=1', '-global', 'ICH9-LPC.disable_s4=1',
        '-device', 'pcie-root-port,bus=pcie.0,addr=0x1,slot=1,x-speed=16,x-width=32',
        '-device', 'isa-applesmc,osk=ourhardworkbythesewordsguardedpleasedontsteal(c)AppleComputerInc',
        '-drive', "if=pflash,format=raw,readonly=on,file=$code",
        '-drive', "if=pflash,format=raw,file=$vars", '-smbios', 'type=2',
        '-device', 'ich9-ahci,id=sata,addr=0x2',
        '-drive', "id=opencore,if=none,format=raw,snapshot=on,file=$oc",
        '-device', 'ide-hd,bus=sata.2,drive=opencore,bootindex=1',
        '-drive', "id=disk,if=none,format=qcow2,file=$disk",
        '-device', 'ide-hd,bus=sata.3,drive=disk,rotation_rate=1',
        '-netdev', (Get-ReimsNetworkArgument $Config),
        '-device', "virtio-net-pci,netdev=net0,id=net0,mac=$($Config.mac),addr=0x3", '-vga', 'none',
        '-device', (ConvertTo-Json -InputObject $gfx -Depth 4 -Compress),
        '-device', 'qemu-xhci,id=xhci,addr=0x5,p2=8,p3=6,conditional-intr-mapping=on',
        '-device', 'usb-kbd,id=kbd,bus=xhci.0', '-device', 'usb-tablet,id=tablet,bus=xhci.0',
        '-audiodev', "$audio,id=snd0,in.voices=0", '-device', 'usb-audio,id=audio,audiodev=snd0,bus=xhci.0',
        '-display', 'none', '-monitor', 'none', '-serial', "file:$serial", '-msg', 'timestamp=on',
        '-qmp', "tcp:127.0.0.1:$($Config.qmp_port),server=on,wait=off"
    )
    if ($Config.PSObject.Properties['microphone'] -and $Config.microphone) {
        $qemuArguments += @('-audiodev','dsound,id=mic0,in.voices=1',
            '-device','usb-audio,id=microphone,input=on,audiodev=mic0,bus=xhci.0')
    }
    foreach ($usb in @(Get-ReimsUsbSelections $Config)) {
        $qemuArguments += @('-device', (Get-ReimsUsbArgument $usb | ConvertTo-Json -Compress))
    }
    if (!$WithoutInstaller) {
        $media = ConvertTo-ReimsQemuPath $Config.iso
        $protection = if ($Config.media_kind -eq 'disk') { 'snapshot=on' } else { 'media=cdrom,readonly=on' }
        $qemuArguments += '-drive', "id=installer,if=none,format=raw,$protection,file=$media"
        $qemuArguments += '-device', $(if ($Config.media_kind -eq 'disk') { 'ide-hd,bus=sata.4,drive=installer' } else { 'ide-cd,bus=sata.4,drive=installer' })
    }
    $toolsImage = Join-Path $VmRoot 'tools/ReimsTools.img'
    if (Test-Path -LiteralPath $toolsImage) {
        $qemuArguments += '-drive', ('id=tools,if=none,format=raw,snapshot=on,file=' + (ConvertTo-ReimsQemuPath $toolsImage))
        $qemuArguments += '-device', 'ide-hd,id=tools,bus=sata.5,drive=tools'
    }
    if (!$DiagnosticTcg) { $qemuArguments += '-no-shutdown' }
    if ($Paused -or $DiagnosticTcg) { $qemuArguments += '-S' }
    return $qemuArguments
}

if ($FunctionsOnly) { return }
$config = Get-ReimsVmConfig
if (!$Qemu) { $Qemu = Join-Path $ProjectRoot 'build/qemu/qemu-system-x86_64.exe' }
$Qemu = [IO.Path]::GetFullPath($Qemu)
$run = Join-Path $VmRoot ("logs/" + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
$qemuArgs = Get-ReimsQemuArguments $config $run -Paused:$Paused -Headless:$Headless -WithoutInstaller:$WithoutInstaller -DiagnosticTcg:$DiagnosticTcg
$runtime = Get-ReimsRuntime
if (!$DryRun -and $runtime -and (Get-ReimsVmProcess $runtime)) {
    $status = Invoke-ReimsQmp $runtime 'query-status'
    if ($status.status -ne 'shutdown') {
        throw "VM is $($status.status). Use qmp.ps1 to control it; start only powers on a shut-down VM."
    }
    $previous = Get-Content -Raw -LiteralPath (Join-Path $runtime.run_directory 'command.json') | ConvertFrom-Json
    # Launch-time -S does not constrain a retained VM's next power-on.
    $original = @(Remove-ReimsUsbLaunchArguments $previous.arguments | Where-Object { $_ -ne '-S' })
    $expected = @(Remove-ReimsUsbLaunchArguments (Get-ReimsQemuArguments $config $runtime.run_directory -Headless:$Headless -WithoutInstaller:$WithoutInstaller -DiagnosticTcg:$DiagnosticTcg) | Where-Object { $_ -ne '-S' })
    $selectedUsb=@(Get-ReimsUsbSelections $config | ForEach-Object id | Sort-Object)
    $peripherals=Invoke-ReimsQmp $runtime qom-list @{path='/machine/peripheral'}
    $liveUsb=@($peripherals | Where-Object name -Like 'hostusb-*' | ForEach-Object name | Sort-Object)
    if (($selectedUsb -join ',') -ne ($liveUsb -join ',')) { throw 'USB selections differ from the retained VM. Exit QEMU before starting.' }
    if (![string]::Equals($runtime.executable, $Qemu, [StringComparison]::OrdinalIgnoreCase) -or
        $previous.headless -ne [bool]$Headless -or
        (ConvertTo-Json -InputObject $original -Compress) -ne (ConvertTo-Json -InputObject $expected -Compress)) {
        throw 'Launch settings changed. Run stop.ps1 -Exit before starting with different disks, firmware, devices or options.'
    }
    Invoke-ReimsQmp $runtime 'system_reset' | Out-Null
    $deadline = [DateTime]::UtcNow.AddSeconds(10)
    do {
        $status = Invoke-ReimsQmp $runtime 'query-status'
        if ($status.status -eq 'prelaunch') { break }
        if ($status.status -ne 'shutdown' -or [DateTime]::UtcNow -gt $deadline) {
            throw "VM reset did not reach prelaunch: $($status.status)."
        }
        Start-Sleep -Milliseconds 100
    } while ($true)
    if (!$Paused) { Invoke-ReimsQmp $runtime 'cont' | Out-Null }
    "Powered on the existing QEMU process, PID $($runtime.pid). Logs: $($runtime.run_directory)"
    return
}
$whpx = Get-ReimsWhpxStatus
$launch = [ordered]@{
    executable = $Qemu; arguments = $qemuArgs; whpx = $whpx
    headless = [bool]$Headless; diagnostic_only = [bool]$DiagnosticTcg
}
if ($DryRun) {
    Write-ReimsJson (Join-Path $ProjectRoot '.work/diagnostics/vm-launch.json') $launch
    $launch | ConvertTo-Json -Depth 10
    return
}
Assert-ReimsVmStopped
Assert-ReimsUsbSafe @(Get-ReimsUsbSelections $config)
if (!$DiagnosticTcg -and !$whpx.ready) {
    throw 'WHPX unavailable. Enable Windows Hypervisor Platform in an elevated Windows session and reboot. No VM was started; no TCG fallback.'
}
foreach ($file in @($Qemu, $config.active_disk, $config.opencore, $config.firmware_code, $config.firmware_vars, $config.rom)) {
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing VM input: $file" }
}
$qemuImg = Get-ReimsTool 'qemu-img.exe'
$chain = @((Invoke-ReimsTool $qemuImg @('info', '--backing-chain', '--output=json', $config.active_disk)) -join "`n" | ConvertFrom-Json)
if ($chain.Count -lt 2 -or $chain[0].format -ne 'qcow2' -or
    [IO.Path]::GetFullPath($chain[-1].filename) -ne $config.base_disk) { throw 'Active disk is not an overlay of this VM base disk.' }
if (!$WithoutInstaller -and !(Test-Path -LiteralPath $config.iso -PathType Leaf)) { throw 'Installer is missing.' }
$accelerators = Invoke-ReimsTool $Qemu @('-accel', 'help')
if (!$DiagnosticTcg -and 'whpx' -notin ($accelerators | ForEach-Object { $_.Trim() })) { throw 'This QEMU was not built with WHPX.' }
$properties = Invoke-ReimsTool $Qemu @('-device', 'apple-gfx-pci,help')
if (($properties -join "`n") -notmatch 'resolution') { throw 'This QEMU lacks the reims apple-gfx-pci ABI.' }
if ((Get-ReimsNetworkMode $config) -eq 'bridge' -and $config.network.adapter -notin @(Get-ReimsTapAdapters | ForEach-Object Name)) {
    throw 'The configured TAP adapter is missing. Install TAP-Windows6, create a Windows bridge, or switch to NAT.'
}
$ports = @($config.qmp_port)
if ((Get-ReimsNetworkMode $config) -eq 'nat') { $ports += $config.ssh_port }
foreach ($port in $ports) {
    $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, [int]$port)
    try { $listener.Start() } finally { $listener.Stop() }
}
New-Item -ItemType Directory -Force -Path $run | Out-Null
Write-ReimsJson (Join-Path $run 'command.json') $launch
$env:REIMS_VGPU_LOG = 'info'
$env:REIMS_VGPU_STATS = '1'
if ($config.PSObject.Properties['usb_backend'] -and $config.usb_backend -eq 'usbdk') { $env:REIMS_USB_BACKEND = 'usbdk' }
else { Remove-Item Env:REIMS_USB_BACKEND -ErrorAction SilentlyContinue }
if ($Headless) { $env:REIMS_VGPU_WINDOW = '0' } else { Remove-Item Env:REIMS_VGPU_WINDOW -ErrorAction SilentlyContinue }
$process = Start-Process -FilePath $Qemu -ArgumentList (($qemuArgs | ForEach-Object { ConvertTo-ReimsWindowsArgument $_ }) -join ' ') -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $run 'stdout.log') -RedirectStandardError (Join-Path $run 'stderr.log') -PassThru
$runtime = [ordered]@{
    pid = $process.Id; start_ticks = $process.StartTime.ToUniversalTime().Ticks; executable = $Qemu
    qmp_port = $config.qmp_port; run_directory = $run; disk = $config.active_disk
    diagnostic_only = [bool]$DiagnosticTcg
}
Write-ReimsJson (Join-Path $VmRoot 'state/runtime.json') $runtime
try {
    $deadline = [DateTime]::UtcNow.AddSeconds(20)
    do {
        if (!(Get-ReimsVmProcess $runtime)) { throw "QEMU exited. Read $run/stderr.log" }
        try { $status = Invoke-ReimsQmp $runtime 'query-status'; break } catch {
            if ([DateTime]::UtcNow -gt $deadline) { throw }
            Start-Sleep -Milliseconds 250
        }
    } while ($true)
    Write-ReimsJson (Join-Path $run 'qmp-status.json') $status
} catch {
    $owned = Get-ReimsVmProcess $runtime
    if ($owned) { Stop-Process -InputObject $owned -Force }
    throw
}
"Started PID $($process.Id), QMP 127.0.0.1:$($config.qmp_port), SSH 127.0.0.1:$($config.ssh_port). Logs: $run"
