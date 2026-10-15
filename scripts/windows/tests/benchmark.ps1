#Requires -Version 7.2
$ErrorActionPreference='Stop'
$path=Join-Path $PSScriptRoot '../benchmark.ps1'
$ast=[scriptblock]::Create((Get-Content -Raw $path)).Ast
$function=$ast.Find({param($n) $n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Get-ReimsFrameTraceSummary'},$true)
. ([scriptblock]::Create($function.Extent.Text))
$result=Get-ReimsFrameTraceSummary @(
    '2026-10-14T08:00:00.000000Z apple_gfx_new_frame ',
    '2026-10-14T08:00:00.010000Z apple_gfx_new_frame ',
    '2026-10-14T08:00:00.030000Z apple_gfx_new_frame ',
    '2026-10-14T08:00:00.060000Z apple_gfx_new_frame ')
if($result.frames -ne 4 -or $result.mean_ms -ne 20 -or $result.median_ms -ne 20 -or $result.p95_ms -ne 30 -or $result.p99_ms -ne 30){throw 'Frame interval statistics are incorrect'}
$result=Get-ReimsFrameTraceSummary @('reims-vgpu stats: idle','apple_gfx_new_frame')
if($result.frames -ne 0 -or $null -ne $result.mean_ms){throw 'Missing timestamps must not fabricate frames'}
$result=Get-ReimsFrameTraceSummary @('2026-10-14T08:00:01Z apple_gfx_new_frame','2026-10-14T08:00:00Z apple_gfx_new_frame')
if($result.clock_regressions -ne 1 -or $null -ne $result.mean_ms){throw 'Backward clock was not recorded'}
'Frame trace parsing, quantiles and missing/backward timestamps passed.'
