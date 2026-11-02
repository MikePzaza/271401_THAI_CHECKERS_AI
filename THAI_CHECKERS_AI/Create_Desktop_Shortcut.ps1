$ErrorActionPreference = 'Stop'

# Keep this script ASCII-only so Windows PowerShell 5.1 cannot corrupt its text
# when the file is read using a legacy system code page.
$launcher = Join-Path $PSScriptRoot 'RUN_WINDOWS.bat'
$icon = Join-Path $PSScriptRoot 'assets\AI_ADOM_VS_YOU.ico'

if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
    throw "RUN_WINDOWS.bat was not found: $launcher"
}
if (-not (Test-Path -LiteralPath $icon -PathType Leaf)) {
    throw "Game icon was not found: $icon"
}

# DesktopDirectory is the physical filesystem folder. Desktop can be a shell
# namespace on some Windows configurations, so prefer DesktopDirectory.
$desktopCandidates = @()
$desktopCandidates += [Environment]::GetFolderPath([Environment+SpecialFolder]::DesktopDirectory)
if (-not [string]::IsNullOrWhiteSpace($env:OneDrive)) {
    $desktopCandidates += (Join-Path $env:OneDrive 'Desktop')
}
if (-not [string]::IsNullOrWhiteSpace($env:USERPROFILE)) {
    $desktopCandidates += (Join-Path $env:USERPROFILE 'Desktop')
}
$desktopCandidates += [Environment]::GetFolderPath([Environment+SpecialFolder]::Desktop)

$desktop = $null
foreach ($candidate in $desktopCandidates) {
    if (-not [string]::IsNullOrWhiteSpace($candidate) -and
        (Test-Path -LiteralPath $candidate -PathType Container)) {
        $desktop = $candidate
        break
    }
}
if ([string]::IsNullOrWhiteSpace($desktop)) {
    throw 'Could not find a physical Desktop folder. Please check Windows Desktop/OneDrive settings.'
}

# Use an ASCII-only filename to avoid invalid question-mark characters if the
# system PowerShell console/code page cannot represent Thai filename text.
$shortcutName = 'AI ADOM VS YOU.lnk'
$linkPath = Join-Path $desktop $shortcutName

$shell = New-Object -ComObject WScript.Shell
$link = $shell.CreateShortcut($linkPath)
$link.TargetPath = $env:ComSpec
$link.Arguments = '/c ""' + $launcher + '""'
$link.WorkingDirectory = $PSScriptRoot
$link.IconLocation = $icon + ',0'
$link.Description = 'AI ADOM VS YOU - Thai Checkers 8x8 with MG400'
$link.Save()

if (-not (Test-Path -LiteralPath $linkPath -PathType Leaf)) {
    throw "Windows did not create the shortcut file: $linkPath"
}

Write-Host ''
Write-Host 'Shortcut created successfully.' -ForegroundColor Green
Write-Host ('Desktop folder: ' + $desktop)
Write-Host ('Shortcut file: ' + $linkPath)
Write-Host 'Shortcut name: AI ADOM VS YOU'
Write-Host 'Keep the game folder in its current location after creating the shortcut.' -ForegroundColor Yellow
