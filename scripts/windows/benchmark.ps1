#Requires -Version 7.2
[CmdletBinding()]
param([ValidateRange(1, 3600)][int]$Seconds = 60, [string]$GuestUser, [string]$IdentityFile,
    [switch]$FrameTrace, [switch]$GpuMetrics)
. "$PSScriptRoot/start.ps1" -FunctionsOnly
function Get-ReimsFrameTraceSummary([string[]]$Lines) {
    $times=@(foreach($line in $Lines) {
        if ($line -match '^(\S+)\s+apple_gfx_new_frame\s*$') {
            $time=[DateTimeOffset]::MinValue
            if ([DateTimeOffset]::TryParse($Matches[1],[ref]$time)) { $time.UtcDateTime.Ticks }
        }
    })
    $intervals=@(); $backwards=0
    for($i=1;$i -lt $times.Count;$i++) {
        $ms=($times[$i]-$times[$i-1])/[double][TimeSpan]::TicksPerMillisecond
        if($ms -lt 0){$backwards++}else{$intervals+=$ms}
    }
    $sorted=@($intervals | Sort-Object)
    $quantile={param($p) if($sorted.Count){$sorted[[math]::Max(0,[math]::Ceiling($p*$sorted.Count)-1)]}else{$null}}
    return @{frames=$times.Count;clock_regressions=$backwards;
        mean_ms=$(if($sorted.Count){($intervals | Measure-Object -Average).Average}else{$null});
        median_ms=(& $quantile .5);p95_ms=(& $quantile .95);p99_ms=(& $quantile .99);
        max_ms=(& $quantile 1);metric='QEMU guest-frame callbacks, not physical monitor scanout'}
}
if ($GuestUser -and $GuestUser -notmatch '^[A-Za-z_][A-Za-z0-9_.-]*$') { throw 'Invalid guest user name.' }
$runtime = Get-ReimsRuntime
if (!$runtime -or !(Get-ReimsVmProcess $runtime)) { throw 'Start the VM before sampling it.' }
if ($runtime.diagnostic_only) { throw 'A paused TCG diagnostic is not a performance benchmark.' }
$config = Get-ReimsVmConfig
$status = Invoke-ReimsQmp $runtime 'query-status'
if (!$status.running) { throw 'VM is paused; no running-VM benchmark can be collected.' }
$directory = Join-Path $ProjectRoot ('.work/diagnostics/benchmark-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
New-Item -ItemType Directory -Path $directory | Out-Null
$samples = [Collections.Generic.List[object]]::new()
$stderr = Join-Path $runtime.run_directory 'stderr.log'
$logOffset = if (Test-Path -LiteralPath $stderr) { (Get-Item -LiteralPath $stderr).Length } else { 0 }
$smi=if($GpuMetrics){Get-ReimsTool 'nvidia-smi.exe'}else{$null}
if($FrameTrace){
    $launch=Get-Content -Raw (Join-Path $runtime.run_directory 'command.json') | ConvertFrom-Json
    if('-d' -in $launch.arguments -or '-D' -in $launch.arguments){throw 'Frame tracing requires the standard launch without a custom QEMU log mask.'}
    Invoke-ReimsQmp $runtime human-monitor-command @{'command-line'='log trace:apple_gfx_new_frame'} | Out-Null
}
$clock = [Diagnostics.Stopwatch]::StartNew()
$previous = (Get-ReimsVmProcess $runtime).TotalProcessorTime.TotalSeconds
$previousTime = 0.0
try { while ($clock.Elapsed.TotalSeconds -lt $Seconds) {
    Start-Sleep -Milliseconds 1000
    $process = Get-ReimsVmProcess $runtime
    if (!$process) { throw "VM exited during benchmark. Partial diagnostics: $directory" }
    $elapsed = $clock.Elapsed.TotalSeconds
    $cpu = $process.TotalProcessorTime.TotalSeconds
    $gpu=$null; $vram=$null
    if($smi){
        $reading=@(& $smi --id=0 --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>$null)
        if($LASTEXITCODE -eq 0 -and $reading.Count -eq 1 -and $reading[0] -match '^\s*(\d+)\s*,\s*(\d+)\s*$'){
            $gpu=[int]$Matches[1]; $vram=[int]$Matches[2]
        }
    }
    $samples.Add([pscustomobject]@{
        elapsed_seconds = [math]::Round($elapsed, 3)
        cpu_core_percent = [math]::Round(100 * ($cpu - $previous) / ($elapsed - $previousTime), 2)
        working_set_bytes = $process.WorkingSet64; private_bytes = $process.PrivateMemorySize64
        nvidia_adapter_gpu_percent=$gpu; nvidia_adapter_vram_mib=$vram
    })
    $samples[-1] | Export-Csv -LiteralPath (Join-Path $directory 'host-process.csv') -Append -NoTypeInformation
    $previous = $cpu
    $previousTime = $elapsed
} } finally {
    if($FrameTrace -and (Get-ReimsVmProcess $runtime)){
        Invoke-ReimsQmp $runtime human-monitor-command @{'command-line'='log none'} | Out-Null
        Invoke-ReimsQmp $runtime trace-event-set-state @{name='apple_gfx_new_frame';enable=$false} | Out-Null
    }
}
$summary = [ordered]@{
    schema = 1; seconds = $clock.Elapsed.TotalSeconds; sample_count = $samples.Count; accelerator = 'whpx'
    metric = 'host QEMU process CPU and memory during the user workload; 100% CPU is one logical core'
    average_cpu_core_percent = ($samples | Measure-Object cpu_core_percent -Average).Average
    peak_working_set_bytes = ($samples | Measure-Object working_set_bytes -Maximum).Maximum
    guest_probe = 'not requested'; run_directory = $runtime.run_directory
    display_stats = 'no samples reported during this interval'
    gpu_metrics=$(if($smi){'NVIDIA adapter 0, includes other host applications; unavailable samples are null'}else{'not requested'})
}
if (Test-Path -LiteralPath $stderr) {
    $file = [IO.File]::Open($stderr, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
    try {
        [void]$file.Seek($logOffset, [IO.SeekOrigin]::Begin)
        $reader = [IO.StreamReader]::new($file)
        $lines=@($reader.ReadToEnd() -split "`r?`n")
        $stats = @($lines | Where-Object { $_ -match '^reims-vgpu stats:' })
        if($FrameTrace){
            $frames=@($lines | Where-Object {$_ -match 'apple_gfx_new_frame'})
            $frames | Set-Content (Join-Path $directory 'frame-events.log') -Encoding utf8NoBOM
            $summary.frame_intervals=Get-ReimsFrameTraceSummary $frames
        }
        if ($stats.Count) {
            $stats | Set-Content -LiteralPath (Join-Path $directory 'display-stats.log') -Encoding utf8NoBOM
            $summary.display_stats = 'raw per-second guest presents, window draws and window GPU times in display-stats.log'
        }
    } finally { $file.Dispose() }
}
if ($GuestUser) {
    $ssh = Get-ReimsTool 'ssh.exe'
    $sshArgs = @('-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10',
        '-o', ('UserKnownHostsFile=' + (Join-Path $VmRoot 'state/known_hosts')), '-p', "$($config.ssh_port)")
    if ($IdentityFile) { $sshArgs += '-i', $IdentityFile }
    $sshArgs += "$GuestUser@127.0.0.1", 'sw_vers; uname -m; sysctl hw.ncpu hw.memsize; system_profiler SPDisplaysDataType'
    $probe = Invoke-ReimsTool $ssh $sshArgs
    $probe | Set-Content -LiteralPath (Join-Path $directory 'guest-info.txt') -Encoding utf8NoBOM
    $summary.guest_probe = 'read-only inventory over an existing SSH key and known host'
}
Write-ReimsJson (Join-Path $directory 'summary.json') $summary
"Host process samples: $directory. This does not measure guest GPU throughput or prove screen correctness."
