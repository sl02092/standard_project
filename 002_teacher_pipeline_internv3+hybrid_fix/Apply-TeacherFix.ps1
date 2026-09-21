<#
.SYNOPSIS
    Applies patch_teacher_noleak.py to every step40_/step46_ pipeline under a
    folder tree, writing NEW *_noleak.py / *_alwaysdino.py files beside each
    original. Originals are never modified.

.DESCRIPTION
    Put this file and patch_teacher_noleak.py together in ONE folder, then run
    it. It walks the tree, skips anything already patched, calls the patcher on
    each file in place, and prints a per-file summary plus a final count.

    Nothing is written unless every anchor in a file is found exactly once and
    the patched result compiles, so a file that has drifted from the frozen
    versions fails loudly rather than being half-edited.

.EXAMPLE
    # dry run first -- shows what would change, writes nothing
    .\Apply-TeacherFix.ps1 -Root 'C:\Users\scott\Desktop\dissertation\090_teacher_pipeline_fix' -Check

.EXAMPLE
    # do it
    .\Apply-TeacherFix.ps1 -Root 'C:\Users\scott\Desktop\dissertation\090_teacher_pipeline_fix'

.EXAMPLE
    # collect every patched file into one folder, ready to upload
    .\Apply-TeacherFix.ps1 -Root '...\090_teacher_pipeline_fix' -Collect '...\090_teacher_pipeline_fix\_upload'

.NOTES
    If PowerShell refuses to run this:
        powershell -ExecutionPolicy Bypass -File .\Apply-TeacherFix.ps1 -Root '...'
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $Root,

    # Filename tag baked into the patched files' output names. MUST stay the
    # same across a resumed run, or the job will not find its own partial
    # output and will start from zero.
    [string] $Tag = 'noleak',

    # Report only; write nothing.
    [switch] $Check,

    # Optionally also copy every patched .py (and its .sub) into one flat
    # folder, so you have a single directory to upload.
    [string] $Collect
)

$ErrorActionPreference = 'Stop'

# ── locate python ────────────────────────────────────────────────────
$python = $null
foreach ($cand in @('python', 'py', 'python3')) {
    $cmd = Get-Command $cand -ErrorAction SilentlyContinue
    if ($cmd) { $python = $cmd.Source; break }
}
if (-not $python) {
    Write-Host "ERROR: no python found on PATH (tried python, py, python3)." -ForegroundColor Red
    exit 1
}

# ── locate the patcher, next to this script ──────────────────────────
$patcher = Join-Path $PSScriptRoot 'patch_teacher_noleak.py'
if (-not (Test-Path $patcher)) {
    Write-Host "ERROR: patch_teacher_noleak.py not found next to this script." -ForegroundColor Red
    Write-Host "       Expected at: $patcher" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $Root)) {
    Write-Host "ERROR: -Root does not exist: $Root" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "  python   : $python"
Write-Host "  patcher  : $patcher"
Write-Host "  root     : $Root"
Write-Host "  tag      : $Tag"
if ($Check) { Write-Host "  mode     : CHECK (nothing will be written)" -ForegroundColor Yellow }
Write-Host ""

# ── find the targets ─────────────────────────────────────────────────
# Already-patched files are skipped so the script is safe to re-run.
$targets = Get-ChildItem -Path $Root -Recurse -File |
    Where-Object {
        # One complete statement per line. PowerShell ends an expression at a
        # newline unless the line itself is unfinished, so a line may not BEGIN
        # with -and / -or, and a comment may not sit inside a parenthesised
        # expression. Both of those are what broke the previous version.
        $isPy  = ($_.Extension -eq '.py') -and (($_.Name -like 'step40_*') -or ($_.Name -like 'step46_*'))
        $isSub = ($_.Extension -eq '.sub')
        $done  = ($_.Name -like '*_noleak.*') -or ($_.Name -like '*_alwaysdino.*')
        ($isPy -or $isSub) -and (-not $done)
    } |
    Sort-Object FullName

if (-not $targets) {
    Write-Host "No step40_/step46_ .py files found under that root." -ForegroundColor Yellow
    exit 1
}

Write-Host ("Found {0} file(s) to patch (.py pipelines + .sub submit files):" -f $targets.Count)
foreach ($t in $targets) {
    Write-Host ("   {0}" -f $t.FullName.Substring($Root.Length).TrimStart('\'))
}
Write-Host ""

# ── patch each one, in place (output lands beside the original) ──────
$ok = 0; $fail = 0; $failed = @()
foreach ($t in $targets) {
    $argList = @($patcher, $t.FullName, '--tag', $Tag)
    if ($Check) { $argList += '--check' }

    # 2>&1 so the patcher's own FAIL lines are visible, not swallowed
    $out = & $python @argList 2>&1
    $rc  = $LASTEXITCODE

    $rel = $t.FullName.Substring($Root.Length).TrimStart('\')
    if ($rc -eq 0) {
        $ok++
        Write-Host "OK    $rel" -ForegroundColor Green
    } else {
        $fail++; $failed += $rel
        Write-Host "FAIL  $rel" -ForegroundColor Red
    }
    $out | Where-Object { $_ -notmatch '^\s*\d+ patched' -and $_ -notmatch '^\s*$' } |
        ForEach-Object { Write-Host "        $_" -ForegroundColor DarkGray }
}

Write-Host ""
Write-Host ("{0} patched, {1} failed" -f $ok, $fail)
if ($fail -gt 0) {
    Write-Host ""
    Write-Host "These files did not match the expected frozen source and were LEFT ALONE:" -ForegroundColor Red
    $failed | ForEach-Object { Write-Host "   $_" -ForegroundColor Red }
    Write-Host "Diff one of them against a file that patched cleanly before editing by hand." -ForegroundColor Red
}

# ── optional: gather everything into one upload folder ───────────────
if ($Collect -and -not $Check -and $ok -gt 0) {
    New-Item -ItemType Directory -Force -Path $Collect | Out-Null
    $patched = Get-ChildItem -Path $Root -Recurse -File -Filter '*.py' |
        Where-Object { $_.Name -like '*_noleak.py' -or $_.Name -like '*_alwaysdino.py' }
    foreach ($p in $patched) { Copy-Item $p.FullName -Destination $Collect -Force }

    # patched .sub files come along too -- these already point at the
    # _noleak / _alwaysdino scripts.
    $subs = Get-ChildItem -Path $Root -Recurse -File -Filter '*.sub' |
        Where-Object { $_.Name -like '*_noleak.sub' -or $_.Name -like '*_alwaysdino.sub' }
    foreach ($s in $subs) { Copy-Item $s.FullName -Destination $Collect -Force }

    Write-Host ""
    Write-Host ("Collected {0} patched .py and {1} .sub file(s) into:" -f $patched.Count, $subs.Count)
    Write-Host "   $Collect"
    Write-Host ""
    Write-Host "  Check each .sub's --time against its NEW script before submitting:" -ForegroundColor Yellow
    Write-Host "  always-DINO removes the no-box GT shortcut, so step46 object frames" -ForegroundColor Yellow
    Write-Host "  now all run the full two-model path and the job may take longer." -ForegroundColor Yellow
}

Write-Host ""
if ($Check) {
    Write-Host "Check complete. Re-run without -Check to write the patched files."
} else {
    Write-Host "Next: run a 10-frame prompt test on one dataset before queueing anything --"
    Write-Host "      set TEACHER_PROMPT_TEST=1 and run the _noleak script."
}

exit $(if ($fail -gt 0) { 1 } else { 0 })
